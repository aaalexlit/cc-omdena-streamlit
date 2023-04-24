from newspaper import Article
from transformers import AutoTokenizer, pipeline, RobertaForSequenceClassification, AutoModelForSequenceClassification
import streamlit as st
import streamlit.components.v1 as components
from codetiming import Timer
import requests
import subprocess
import pandas as pd

import jsonlines
from annotated_text import annotated_text
from fastcoref import spacy_component
import spacy
import nltk
import torch
import gc
import os
import time
from pathlib import Path
import re
from itertools import zip_longest

if 'rand_num' not in st.session_state:
    st.session_state.rand_num = int(time.time() * 1000)

CLAIMS_FILE = f"claims_{st.session_state.rand_num}.jsonl"
CORPUS_FILE = f"corpus_{st.session_state.rand_num}.jsonl"
PREDS_FILE = f"preds_{st.session_state.rand_num}.jsonl"
GW_STANCE_TEST_PREFIX = f"test_{st.session_state.rand_num}"
GW_STANCE_PRED_PREFIX = f"pred_{st.session_state.rand_num}"

label_highlight_color = {'CONTRADICT': '#faa',
                         'REFUTES': '#faa',
                         'NOT_ENOUGH_INFO': '#808080',
                         'SUPPORT': '#afa',
                         'SUPPORTS': '#afa'}

label_mapping = {0: 'disagree', 1: 'neutral', 2: 'agree'}


@st.cache_resource
def download_multivers_models():
    nltk.download('punkt')
    # make multivers download needed model
    predict_with_multivers()


@st.cache_resource
def get_climate_sentence_detection_model():
    return RobertaForSequenceClassification.from_pretrained('kruthof/climateattention-10k-upscaled',
                                                            num_labels=2)


@st.cache_resource
def get_spacy_nlp():
    return spacy.load('en_core_web_sm',
                      enable=['tok2vec', 'senter'],
                      config={"nlp": {"disabled": []}})


@st.cache_resource
def get_spacy_coref():
    nlp = spacy.load('en_core_web_sm',
                     exclude=["parser", "lemmatizer", "ner", "textcat"])
    nlp.add_pipe("fastcoref")
    return nlp


@st.cache_resource
def get_climatebert_tokenizer():
    return AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-f")


@st.cache_resource
def get_claimbuster_model():
    return AutoModelForSequenceClassification.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")


@st.cache_resource
def get_claimbuster_tokenizer():
    return AutoTokenizer.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")


def is_about_climate(texts: [str]):
    if torch.cuda.is_available():
        device = 0
        batch_size = 128
    else:
        device = -1
        batch_size = 1

    model = get_climate_sentence_detection_model()
    tokenizer = get_climatebert_tokenizer()
    pipe = pipeline("text-classification", model=model,
                    tokenizer=tokenizer, device=device,
                    truncation=True, padding=True)
    labels = []
    probs = []
    for out in pipe(texts, batch_size=batch_size):
        labels.append(out['label'])
        probs.append(out['score'])
    torch.cuda.empty_cache()
    return labels, probs


def filter_climate_related(sentences):
    labels, _ = is_about_climate(sentences)
    return [doc for label, doc in zip(labels, sentences) if label == 'Yes']


def is_claim(sentences):
    tokenizer = get_claimbuster_tokenizer()
    model = get_claimbuster_model()
    if torch.cuda.is_available():
        device = 0
        batch_size = 128
    else:
        device = -1
        batch_size = 1

    pipe = pipeline("text-classification", model=model,
                    tokenizer=tokenizer, device=device,
                    truncation=True, padding=True)
    labels, probs = [], []
    for out in pipe(sentences, batch_size=batch_size):
        labels.append(out['label'])
        probs.append(out['score'])

    return list(map(lambda l: model.config.label2id[l], labels)), probs


def filter_claims(sentences: [str]):
    if 'claim_prob_threshold' in st.session_state:
        threshold = st.session_state.claim_prob_threshold
    else:
        threshold = 0.5
    labels, probs = is_claim(sentences)
    # for the time being will return Unimportant Factual Statement (UFS) and Check-worthy Factual Statement (CFS)
    return [sentence for sentence, label, prob in zip(sentences, labels, probs) if
            label in [1, 2] and prob > threshold]


def get_text_from_url(news_article_url):
    article = Article(news_article_url)
    article.download()
    article.parse()
    article.nlp()
    return f"{article.title}. {article.text}"


def get_text_from_input(text: str):
    if text.startswith('http'):
        text = get_text_from_url(text)
    return text


def predict_gw_stance(input_sentences, filter=True):
    if filter:
        sentences = filter_climate_related(input_sentences)
    else:
        sentences = input_sentences

    if not sentences:
        return None
    test_filename = f'{GW_STANCE_TEST_PREFIX}.tsv'
    eval_filename = f'final_model/no-dev/eval_results_{GW_STANCE_TEST_PREFIX}_.txt'
    preds_filename = f'final_model/no-dev/{GW_STANCE_PRED_PREFIX}_{GW_STANCE_TEST_PREFIX}.tsv'
    try:
        df = pd.DataFrame(sentences)
        df["lab"] = "neutral"
        df["weight"] = 1.0
        df.to_csv(test_filename, sep='\t', index=False, header=False)

        cmd = "python GWStance/3_stance_detection/2_Stance_model/predict.py " \
              "final_model/config.json " \
              "final_model/no-dev " \
              f"--input-prefix {GW_STANCE_TEST_PREFIX} " \
              f"--output-prefix {GW_STANCE_PRED_PREFIX} " \
              "--data-dir ./ " \
              "--transformers-dir /opt/bitnami/python/lib/python3.8/site-packages/transformers"
        subprocess.call(cmd, shell=True, executable='/bin/bash')

        input_df = pd.read_csv(test_filename, sep='\t', header=None, names=["text", "fake1", "fake2"])
        preds_df = pd.read_csv(preds_filename, sep='\t')

        res_df = input_df.join(preds_df)[["text", "predicted"]]

        res_df['predicted'] = res_df['predicted'].apply(lambda x: label_mapping[x])
        res_df = res_df.reset_index(drop=True)
    except:
        return "some problem encountered during the analysis"
    finally:
        Path.unlink(Path(test_filename), missing_ok=True)
        Path.unlink(Path(eval_filename), missing_ok=True)
        Path.unlink(Path(preds_filename), missing_ok=True)

    return res_df


def predict_with_multivers():
    gc.collect()
    cmd = f'source mult/bin/activate; python multivers/multivers/predict.py \
        --checkpoint_path=multivers/checkpoints/fever_sci.ckpt \
        --input_file="{CLAIMS_FILE}" \
        --corpus_file="{CORPUS_FILE}" \
        --output_file="{PREDS_FILE}"'
    subprocess.call(cmd, shell=True, executable='/bin/bash')


@Timer(text="get_abstracts_matching_claims elapsed time: {seconds:.0f} s")
def get_abstracts_matching_claims(threshold=0.6):
    api_url = f"http://{os.getenv('EVIDENCE_API_IP')}/api/abstract/evidence/batch"

    request_body = get_request_body(threshold)
    params = {'re_rank': st.session_state.re_rank if 're_rank' in st.session_state else True}
    response = requests.post(api_url, json=request_body, params=params)
    return response.json()


@Timer(text="get_verified_against_phrases elapsed time: {seconds:.0f} s")
def get_verified_against_phrases(threshold=0.7):
    path = 'phrase'
    if 'retrieve_abstracts' in st.session_state and st.session_state.retrieve_abstracts:
        path = 'abstract'

    api_url = f"http://{os.getenv('EVIDENCE_API_IP')}/api/{path}/verify/batch"
    request_body = get_request_body(threshold)
    params = {'re_rank': st.session_state.re_rank if 're_rank' in st.session_state else True,
              'include_title': st.session_state.include_title if 'include_title' in st.session_state else True,
              'filter_nei': (not st.session_state.show_nei) if 'show_nei' in st.session_state else True}
    response = requests.post(api_url, json=request_body, params=params)
    return response.json()


def get_request_body(threshold):
    request_body = {
        "claims": st.session_state.filtered_input_sentences,
        "threshold": st.session_state.similarity_threshold if 'similarity_threshold' in st.session_state else threshold,
        "top_k": st.session_state.top_k if 'top_k' in st.session_state else 10
    }
    return request_body


def split_into_sentences(text):
    nlp = get_spacy_nlp()
    sentences = [sent.text for sent in nlp(text).sents]
    # replace end of line with space
    return [' '.join(inp_sent.rsplit('\n')) for inp_sent in sentences]


def convert_evidences_from_abstracts_to_multivers_format():
    claims = st.session_state.filtered_input_sentences
    evidence_abstracts = get_abstracts_matching_claims()
    with jsonlines.open(CLAIMS_FILE, 'w') as claims_writer, \
            jsonlines.open(CORPUS_FILE, 'w') as corpus_writer:
        doc_id = 0

        for claim_id, cur_evidences in enumerate(evidence_abstracts):
            doc_ids = []
            for i, cur_evidence in enumerate(cur_evidences):
                sentences = split_into_sentences(cur_evidence["text"])
                evidence_abstract = {
                    'doc_id': doc_id,
                    'title': cur_evidence["title"],
                    'abstract': sentences,
                    'doi': cur_evidence["doi"],
                    'year': cur_evidence["year"],
                    'citation_count': cur_evidence["citation_count"],
                    'influential_citation_count': cur_evidence["influential_citation_count"]
                }
                corpus_writer.write(evidence_abstract)
                doc_ids.append(doc_id)
                doc_id += 1
            claim_doc = {
                'id': claim_id,
                'claim': claims[claim_id],
                'doc_ids': doc_ids
            }
            claims_writer.write(claim_doc)


def get_verified_claims():
    claim_id_evidence_ids = {}
    try:
        with jsonlines.open(PREDS_FILE, 'r') as preds_reader:
            for pred_line in preds_reader.iter():
                if pred_line["evidence"]:
                    evidences = []
                    for key in pred_line["evidence"]:
                        d = {'evidence_id': key,
                             'label': pred_line["evidence"][key]['label'],
                             'sentences': pred_line["evidence"][key]['sentences']
                             }
                        evidences.append(d)
                    claim_id_evidence_ids[pred_line["id"]] = {"evidences": evidences}

        for claim_id in claim_id_evidence_ids:
            with jsonlines.open(CLAIMS_FILE, 'r') as claims_reader:
                for line_num, claim_line in enumerate(claims_reader.iter()):
                    if line_num == claim_id:
                        claim_id_evidence_ids[claim_id]['claim_text'] = claim_line['claim']
                        claim_id_evidence_ids[claim_id]['claim_id'] = claim_line['id']

        for claim_id in claim_id_evidence_ids:
            evidences = claim_id_evidence_ids[claim_id]['evidences']
            for evidence in evidences:
                with jsonlines.open(CORPUS_FILE, 'r') as corpus_reader:
                    for line_num, corpus_line in enumerate(corpus_reader.iter()):
                        if str(line_num) == evidence['evidence_id']:
                            evidence['evidence_title'] = corpus_line['title']
                            evidence['evidence_text'] = corpus_line['abstract']
                            evidence['doi'] = corpus_line['doi']
                            evidence['year'] = corpus_line['year']
                            evidence['citation_count'] = corpus_line['citation_count']
                            evidence['influential_citation_count'] = corpus_line['influential_citation_count']
                            sentences_text = []
                            for sent_num in evidence['sentences']:
                                sentences_text.append(corpus_line['abstract'][sent_num])
                            evidence['sentences_text'] = sentences_text
    except:
        return "Something went wrong while verifying with MultiVerS"
    finally:
        Path.unlink(Path(PREDS_FILE), missing_ok=True)
        Path.unlink(Path(CLAIMS_FILE), missing_ok=True)
        Path.unlink(Path(CORPUS_FILE), missing_ok=True)
    return claim_id_evidence_ids


# Create the Streamlit app
def main():
    st.set_page_config(page_title="Verify news article using different verification models",
                       page_icon=":earth_americas:",
                       layout='wide')

    pre_download_used_models()

    add_sidebar()
    tab_sci_veri, tab_faq, tab_how_to = set_header_and_tabs()

    with tab_sci_veri:
        if 'input_text' in st.session_state:
            init_value = st.session_state.input_text
        else:
            init_value = ''
        st.text_area("Enter Text or URL of a media article about Climate",
                     placeholder="CO2 is not the cause of our current warming trend",
                     key='original_input_text',
                     on_change=on_input_text_change,
                     value=init_value)

        display_text_to_analyze()
        st.subheader("Options")
        options_col_left, options_col_right = st.columns(2)
        with options_col_left:
            filter_climate_checkbox = st.checkbox('Filter climate related sentences', on_change=on_filter_state_change,
                                                  args=('verified_with_multivers', 'verified_with_climatebert'))

            filter_claims_checkbox = st.checkbox('Extract check worthy claims', on_change=on_filter_state_change,
                                                 args=('verified_with_multivers', 'verified_with_climatebert'))
        with options_col_right:
            with st.expander("Advanced options"):
                add_advanced_options()
        st.write("---")
        st.subheader("Verification model")

        st.radio("Verification model",
                 ('ClimateBERT fine-tuned on Climate-FEVER', 'MultiVerS'),
                 key='model',
                 index=0)
        verify_button = st.button('Verify')

        st.write("---")
        # Verify with Multivers
        if verify_button and st.session_state.model == 'MultiVerS':
            if 'input_sentences' not in st.session_state:
                st.warning('Enter some text to continue')
            else:
                apply_filters(filter_claims_checkbox, filter_climate_checkbox)
                clear_keys('verified_with_multivers')
                if 'filtered_input_sentences' in st.session_state:
                    with st.spinner(text='Retrieving relevant evidences'):
                        convert_evidences_from_abstracts_to_multivers_format()
                    with st.spinner(text='Verifying'):
                        predict_with_multivers()
                    with st.spinner(text='Preparing output'):
                        verified_claims = get_verified_claims()
                        if not verified_claims:
                            st.warning("According to MultiVerS model, there's \
                    not enough information to verify any claim from the article")
                        else:
                            if isinstance(verified_claims, str):
                                st.warning(verified_claims)
                            else:
                                st.session_state.verified_with_multivers = verified_claims

        # Classify text and show result
        if verify_button and st.session_state.model != 'MultiVerS':
            if 'input_sentences' not in st.session_state:
                st.warning('Enter some text to continue')
            else:
                apply_filters(filter_claims_checkbox, filter_climate_checkbox)
                clear_keys('verified_with_climatebert', 'slider_value_changed')
                if 'filtered_input_sentences' in st.session_state:
                    with st.spinner(text='Verifying the statements'):
                        res = get_verified_against_phrases()
                        if not res:
                            st.warning("According to the model, there's \
                                        not enough information to verify any claim from the article")
                        else:
                            st.session_state.verified_with_climatebert = res

        multivers_container = st.container()
        if 'verified_with_multivers' in st.session_state:
            multivers_container.header("Multivers predictions")
            show_sentences_to_run_inference_on(multivers_container)
            output_multivers_predictions(multivers_container)
        else:
            multivers_container.empty()

        climatebert_container = st.container()
        if 'verified_with_climatebert' in st.session_state:
            climatebert_container.header("ClimateBERT fine-tuned on Climate-FEVER predictions")
            show_sentences_to_run_inference_on(climatebert_container)
            climatebert_container.columns(4)[0].slider(
                label='**Choose probability threshold** that represents the confidence level of predictions',
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key='prob_slider',
                on_change=output_climatebert_prediction_slider,
                args=(climatebert_container,))
            if 'slider_value_changed' not in st.session_state:
                output_climatebert_prediction(climatebert_container)
        else:
            climatebert_container.empty()

    with tab_how_to:
        fill_tab('https://raw.githubusercontent.com/aaalexlit/cc-omdena-streamlit/main/README.md')

    with tab_faq:
        url = 'https://raw.githubusercontent.com/aaalexlit/cc-evidences-api/main/doc/db.md'
        fill_tab(url)


def mermaid(code: str) -> None:
    components.html(
        f"""
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        
        <pre class="mermaid">
            {code}
        </pre>

        """,
        height=700,
        scrolling=True
    )


def fill_tab(url):
    page = requests.get(url)
    pattern = re.compile(r'```mermaid.*?```', flags=re.DOTALL)
    mermaids = pattern.findall(page.text)
    texts = pattern.split(page.text)
    for t, m in zip_longest(texts, mermaids, fillvalue=''):
        st.write(t)
        if m != '':
            m = m.replace('```mermaid', ' ')
            m = m.replace('```', ' ')
            mermaid(m)


def add_advanced_options():
    st.checkbox('Re-rank evidences',
                key='re_rank',
                value=True)
    st.checkbox('Co-reference resolution of the input text',
                key='coref_source',
                value=False)
    cols = st.columns(2)
    with cols[0]:
        st.number_input('Top evidence number to retrieve',
                        min_value=10,
                        max_value=30,
                        value=10,
                        step=5,
                        key='top_k')
        st.number_input('Evidence similarity threshold',
                        min_value=0.5,
                        max_value=0.9,
                        value=0.7,
                        step=0.1,
                        key='similarity_threshold')
    st.write('---')
    st.write('ClimateBERT-based model options')
    st.checkbox('Include title during verification',
                key='include_title',
                value=True)
    st.checkbox('Retrieve evidence abstracts',
                key='retrieve_abstracts',
                value=False)
    st.checkbox('Show NEI',
                key='show_nei',
                value=False)


def apply_filters(filter_claims_checkbox, filter_climate_checkbox):
    if 'coref_source' in st.session_state:
        if st.session_state.coref_source:
            with st.spinner("Resolving co-references"):
                coref_resolved = coref_resolve(st.session_state.input_text)
                st.session_state.input_sentences = split_into_sentences(coref_resolved)
        else:
            st.session_state.input_sentences = split_into_sentences(st.session_state.input_text)
        st.session_state.filtered_input_sentences = st.session_state.input_sentences

    if 'refilter' in st.session_state and st.session_state.refilter:
        if filter_climate_checkbox:
            with st.spinner("Filtering climate related sentences"):
                st.session_state.filtered_input_sentences = filter_climate_related(
                    st.session_state.filtered_input_sentences)
            if not st.session_state.filtered_input_sentences:
                st.error("None of the extracted sentences are climate related.")
                st.stop()
        if filter_claims_checkbox:
            with st.spinner("Detecting claims"):
                st.session_state.filtered_input_sentences = filter_claims(st.session_state.filtered_input_sentences)
            if not st.session_state.filtered_input_sentences:
                st.error("No check worthy claims were found in the input ")
                st.stop()
        st.session_state.refilter = False


def on_filter_state_change(*keys_to_remove):
    clear_keys(*keys_to_remove)
    st.session_state.refilter = True
    if 'input_sentences' in st.session_state:
        st.session_state.filtered_input_sentences = st.session_state.input_sentences


def on_input_text_change():
    clear_keys('filtered_input_sentences', 'verified_with_climatebert', 'verified_with_multivers')
    st.session_state.refilter = True
    st.session_state.input_text = get_text_from_input(st.session_state.original_input_text)
    # Check if the text is about climate and then continue
    with st.spinner("Check if the input is Climate related text"):
        st.session_state.not_climate_related_text = len(st.session_state.input_text.strip()) and \
                                                    not filter_climate_related([st.session_state.input_text])
    st.session_state.input_sentences = split_into_sentences(st.session_state.input_text)
    st.session_state.filtered_input_sentences = st.session_state.input_sentences


def coref_resolve(text: str) -> str:
    coref_resolver = get_spacy_coref()
    doc = coref_resolver(text, component_cfg={"fastcoref": {'resolve_text': True}})
    return doc._.resolved_text


def display_text_to_analyze():
    if 'not_climate_related_text' in st.session_state and st.session_state.not_climate_related_text:
        st.warning("Looks like the text you entered doesn't concern the topic of Climate")
    if 'original_input_text' in st.session_state and st.session_state.original_input_text.startswith('http'):
        with st.expander('Text that was extracted from the link and will be analyzed'):
            st.write(st.session_state.input_text)


def clear_keys(*keys_to_remove):
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]


def set_header_and_tabs():
    st.header("Media article scientific verification")
    tab_bias_detection, tab_how_to, tab_faq = \
        st.tabs(["Scientific verification", "Main Readme", "Evidences DB"])
    return tab_bias_detection, tab_faq, tab_how_to


def add_sidebar():
    # Add a sidebar with links
    st.sidebar.title("Omdena, Local Chapter, 🇩🇪 Cologne")
    project_link = '[Project Description](https://omdena.com/chapter-challenges/detecting-bias-in-climate-reporting' \
                   '-in-english-and-german-language-news-media/)'
    st.sidebar.markdown(project_link, unsafe_allow_html=True)
    st.sidebar.markdown('[This App Github Repo](https://github.com/aaalexlit/cc-omdena-streamlit/)',
                        unsafe_allow_html=True)
    st.sidebar.markdown('[Evidence Retrieval API Github Repo](https://github.com/aaalexlit/cc-evidences-api/)',
                        unsafe_allow_html=True)
    github_link = '[Omdena Github Repo](https://github.com/OmdenaAI/cologne-germany-reporting-bias/)'
    st.sidebar.markdown(github_link, unsafe_allow_html=True)


def pre_download_used_models():
    download_multivers_models()
    get_climatebert_tokenizer()
    get_climate_sentence_detection_model()
    get_spacy_nlp()
    get_spacy_coref()
    get_claimbuster_model()
    get_claimbuster_tokenizer()


def show_sentences_to_run_inference_on(container):
    with container:
        with st.expander("Sentences that will be analyzed (the ones that are left after applying the filters)"):
            # CSS to inject contained in a string
            hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                        """

            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state.filtered_input_sentences, columns=['sentence']))


def output_climatebert_prediction_slider(container):
    st.session_state.slider_value_changed = True
    output_climatebert_prediction(container)


def output_climatebert_prediction(container):
    if 'verified_with_climatebert' in st.session_state:
        with container:
            for claim, evidences in zip(st.session_state.filtered_input_sentences,
                                        st.session_state.verified_with_climatebert):
                show_claim = st.session_state.show_nei or has_support_or_refute_preds_above_threshold(evidences)
                if show_claim:
                    st.markdown(f"### **Claim  :orange[{claim}]**")
                    evidences.sort(key=lambda ev: ev['probability'], reverse=True)
                    for evidence in evidences:
                        if (((st.session_state.show_nei and evidence['label'] == 'NOT_ENOUGH_INFO') or
                            evidence['label'] != 'NOT_ENOUGH_INFO')) and \
                                evidence['probability'] > st.session_state.prob_slider:
                            label = evidence['label']
                            annotated_text((label, f"{evidence['probability']:.2f}",
                                            label_highlight_color[label],
                                            'black'))
                            st.markdown(f"""**Article title (Year)**: 
                            [{evidence['title']}](https://www.doi.org/{evidence['doi']}) ({evidence['year']})  
                                      **Rationale**: {evidence['text']}  
                                      **Citation count (Influential)**: {evidence['citation_count']} 
                                      ({evidence['influential_citation_count']}) 
                                      """)
                    st.markdown("""---""")


def has_support_or_refute_preds_above_threshold(evidences):
    show_claim = False
    for evidence in evidences:
        if evidence['label'] != 'NOT_ENOUGH_INFO' and evidence['probability'] > st.session_state.prob_slider:
            show_claim = True
    return show_claim


def output_multivers_predictions(container):
    with container:
        verified_claims = st.session_state.verified_with_multivers
        for claim_id, claim in verified_claims.items():
            st.markdown(f"### **Claim  :orange[{claim['claim_text']}]**")
            for evidence in claim['evidences']:
                label = evidence['label']
                annotated_text((label, "",
                                label_highlight_color[label],
                                'black'))
                st.markdown(f"""**Article title (Year)**: 
                            [{evidence['evidence_title']}](https://www.doi.org/{evidence['doi']}) ({evidence['year']})""")
                if evidence['sentences']:
                    for sent in evidence['sentences_text']:
                        st.markdown(f"**Rationale**: {sent}")
                else:
                    st.markdown(f"**Abstract**: {' '.join(evidence['evidence_text'])}")
                st.markdown(f"""**Citation count (Influential)**: {evidence['citation_count']} 
                                      ({evidence['influential_citation_count']}) 
                                      """)
            st.markdown("""---""")


if __name__ == "__main__":
    main()
