from newspaper import Article
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, pipeline, RobertaForSequenceClassification, AutoModelForSequenceClassification
import streamlit as st
from codetiming import Timer
import requests
import subprocess
import pandas as pd

import jsonlines
from annotated_text import annotated_text
import nltk
import torch
import gc
import os
import time
from pathlib import Path

if 'rand_num' not in st.session_state:
    st.session_state.rand_num = int(time.time() * 1000)

CLAIMS_FILE = f"claims_{st.session_state.rand_num}.jsonl"
CORPUS_FILE = f"corpus_{st.session_state.rand_num}.jsonl"
PREDS_FILE = f"preds_{st.session_state.rand_num}.jsonl"
GW_STANCE_TEST_PREFIX = f"test_{st.session_state.rand_num}"
GW_STANCE_PRED_PREFIX = f"pred_{st.session_state.rand_num}"

label_highlight_color = {'CONTRADICT': '#faa',
                         'REFUTES': '#faa',
                         'SUPPORT': '#afa',
                         'SUPPORTS': '#afa'}

label_mapping = {0: 'disagree', 1: 'neutral', 2: 'agree'}


@st.cache_resource
def download_models():
    nltk.download('punkt')
    # make multivers download needed model
    predict_with_multivers()


@st.cache_resource
def get_climate_sentence_detection_model():
    return RobertaForSequenceClassification.from_pretrained('kruthof/climateattention-10k-upscaled',
                                                            num_labels=2)


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


def predict_gw_stance(input_sentences):
    sentences = filter_climate_related(input_sentences)

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
def get_abstracts_matching_claims(top_k=10, threshold=0.7):
    claims = st.session_state.filtered_input_sentences
    api_url = f"http://{os.getenv('EVIDENCE_API_IP')}/api/abstract/evidence"

    responses = []

    for claim in claims:
        request_body = {
            "claim": claim,
            "threshold": threshold,
            "top_k": top_k
        }
        response = requests.post(api_url, json=request_body)
        responses.append(response.json())
    return responses


@Timer(text="get_verified_against_phrases elapsed time: {seconds:.0f} s")
def get_verified_against_phrases(top_k=10, threshold=0.7):
    claims = st.session_state.filtered_input_sentences
    api_url = f"http://{os.getenv('EVIDENCE_API_IP')}/api/phrase/verify"

    responses = []

    for claim in claims:
        request_body = {
            "claim": claim,
            "threshold": threshold,
            "top_k": top_k
        }
        response = requests.post(api_url, json=request_body)
        responses.append(response.json())
    return responses


def convert_evidences_from_abstracts_to_multivers_format():
    claims = st.session_state.filtered_input_sentences
    evidence_abstracts = get_abstracts_matching_claims()
    with jsonlines.open(CLAIMS_FILE, 'w') as claims_writer, \
            jsonlines.open(CORPUS_FILE, 'w') as corpus_writer:
        doc_id = 0

        for claim_id, cur_evidences in enumerate(evidence_abstracts):
            doc_ids = []
            for i, cur_evidence in enumerate(cur_evidences):
                evidence_abstract = {
                    'doc_id': doc_id,
                    'title': cur_evidence["title"],
                    'abstract': sent_tokenize(cur_evidence["text"]),
                    'doi': cur_evidence["doi"],
                    'year': cur_evidence["year"]
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
                            sentences_text = []
                            for sent_num in evidence['sentences']:
                                sentences_text.append(corpus_line['abstract'][sent_num])
                            evidence['sentences_text'] = sentences_text
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
    tab_sci_veri, tab_gw_stance, tab_faq, tab_how_to = set_header_and_tabs()

    with tab_sci_veri:
        st.text_area("Enter Text or URL of a media article about Climate",
                     placeholder="CO2 is not the cause of our current warming trend",
                     on_change=on_input_text_change,
                     key='original_input_text')

        if 'input_sentences' not in st.session_state:
            st.stop()
        display_text_to_analyze()
        filter_climate_checkbox = st.checkbox('Filter climate related sentences', on_change=on_filter_state_change,
                                              args=('verified_with_multivers', 'verified_with_climatebert'))

        filter_claims_checkbox = st.checkbox('Extract check worthy claims', on_change=on_filter_state_change,
                                             args=('verified_with_multivers', 'verified_with_climatebert'))

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

        # Verify with Multivers
        if st.button("Verify article text with Multivers"):
            clear_keys('verified_with_multivers', 'verified_with_climatebert')
            with st.spinner(text='Retrieving relevant evidences'):
                convert_evidences_from_abstracts_to_multivers_format()
            with st.spinner(text='Verifying'):
                predict_with_multivers()
            with st.spinner(text='Preparing output'):
                verified_claims = get_verified_claims()
                if not verified_claims:
                    st.warning("According to Multivers model, there's \
            not enough information to verify any claim from the article")
                else:
                    st.session_state.verified_with_multivers = verified_claims

        # Classify text and show result
        if st.button("Verify article text with ClimateBERT fine-tuned on Climate-FEVER"):
            clear_keys('verified_with_multivers', 'verified_with_climatebert', 'slider_value_changed')
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
            climatebert_container.slider(label='**Choose probability threshold**',
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

    with tab_gw_stance:
        st.write("""Enter a Text below and click the Classify Button 
        to extract Climate Change related  sentences from text and classify them
        as agreeing with Global warming, disagreeing with Global Warming or neutral""")

        text_input = st.text_area("Enter Text")
        input_sentences = sent_tokenize(text_input)

        # Classify text and show result
        if st.button("Detect Global Warming stance in climate related sentences"):
            with st.spinner(text='Performing stance detection'):
                res = predict_gw_stance(input_sentences)
                if res is not None:
                    if isinstance(res, str):
                        st.error(res)
                    else:
                        # CSS to inject contained in a string
                        hide_table_row_index = """
                                    <style>
                                    thead tr th:first-child {display:none}
                                    tbody th {display:none}
                                    </style>
                                    """

                        # Inject CSS with Markdown
                        st.markdown(hide_table_row_index, unsafe_allow_html=True)
                        # st.dataframe(res.style, use_container_width=True)
                        st.table(res.style.applymap(style_agree, props='color:white;background-color:green')
                                 .applymap(style_disagree, props='color:white;background-color:red'))
                else:
                    st.warning("None of the extracted sentences are climate related.")

    with tab_how_to:
        st.write("tbd")

    with tab_faq:
        st.write("tbd")


def style_agree(v, props=''):
    return props if v in ['agree'] else None


def style_disagree(v, props=''):
    return props if v in ['disagree'] else None


def on_filter_state_change(*keys_to_remove):
    clear_keys(*keys_to_remove)
    st.session_state.refilter = True
    st.session_state.filtered_input_sentences = st.session_state.input_sentences


def on_input_text_change():
    clear_keys('filtered_input_sentences', 'verified_with_climatebert', 'verified_with_multivers')
    st.session_state.refilter = True
    st.session_state.input_text = get_text_from_input(st.session_state.original_input_text)
    # Check if the text is about climate and then continue
    with st.spinner("Check if the input is Climate related text"):
        st.session_state.not_climate_related_text = len(st.session_state.input_text.strip()) and \
                                                    not filter_climate_related([st.session_state.input_text])
    st.session_state.input_sentences = sent_tokenize(st.session_state.input_text)
    st.session_state.filtered_input_sentences = st.session_state.input_sentences


def display_text_to_analyze():
    if not len(st.session_state.input_text.strip()):
        st.stop()
    if st.session_state.not_climate_related_text:
        st.warning("Looks like the text you entered doesn't concern the topic of Climate")
    if 'original_input_text' in st.session_state and st.session_state.original_input_text.startswith('http'):
        with st.expander('Text that was extracted from the link and will be analyzed'):
            st.write(st.session_state.input_text)


def clear_keys(*keys_to_remove):
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]


def set_header_and_tabs():
    st.header("Media article scientific verification and Stance Detection")
    tab_bias_detection, tab_gw_stance, tab_how_to, tab_faq = \
        st.tabs(["Scientific verification", "Global Warming Stance Detection", "How-To", "FAQ"])
    return tab_bias_detection, tab_gw_stance, tab_faq, tab_how_to


def add_sidebar():
    # Add a sidebar with links
    st.sidebar.title("Omdena, Local Chapter, 🇩🇪 Cologne")
    project_link = '[Project Description](https://omdena.com/chapter-challenges/detecting-bias-in-climate-reporting' \
                   '-in-english-and-german-language-news-media/)'
    st.sidebar.markdown(project_link, unsafe_allow_html=True)
    github_link = '[Github Repo](https://github.com/OmdenaAI/cologne-germany-reporting-bias/)'
    st.sidebar.markdown(github_link, unsafe_allow_html=True)


def pre_download_used_models():
    download_models()
    get_climatebert_tokenizer()
    get_climate_sentence_detection_model()


def show_sentences_to_run_inference_on(container):
    with container:
        with st.expander("Climate related sentences that we'll attempt to verify"):
            for claim in st.session_state.filtered_input_sentences:
                st.write(claim)


def output_climatebert_prediction_slider(container):
    st.session_state.slider_value_changed = True
    output_climatebert_prediction(container)


def output_climatebert_prediction(container):
    if 'verified_with_climatebert' in st.session_state:
        with container:
            for claim, evidences in zip(st.session_state.filtered_input_sentences,
                                        st.session_state.verified_with_climatebert):
                show_claim = has_support_or_refute_preds_above_threshold(evidences)
                if show_claim:
                    st.markdown(f"### **Claim  :orange[{claim}]**")
                    for evidence in evidences:
                        if evidence['label'] != 'NOT_ENOUGH_INFO' and \
                                evidence['probability'] > st.session_state.prob_slider:
                            label = evidence['label']
                            st.write("**Label**: ")
                            annotated_text((label, "",
                                            label_highlight_color[label],
                                            'black'))
                            st.markdown(f"""**Article title**: {evidence['title']}  
                                      **Year**: {evidence['year']}  
                                      **Article link**: {evidence['doi']}  
                                      **Phrase**: {evidence['text']}  
                                      **Probability**: {evidence['probability']:.2f}""")
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
                st.write("**Label**: ")
                annotated_text((label, "",
                                label_highlight_color[label],
                                'black'))
                st.markdown(f"""**Article title**: {evidence['evidence_title']}  
                    **Year**: {evidence['year']}  
                    **Article link**: {evidence['doi']}""")
                if evidence['sentences']:
                    for sent in evidence['sentences_text']:
                        st.markdown(f"**Phrase**: {sent}")
                else:
                    st.markdown(f"**Abstract**: {' '.join(evidence['evidence_text'])}")
            st.markdown("""---""")


if __name__ == "__main__":
    main()
