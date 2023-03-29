from newspaper import Article
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, pipeline, RobertaForSequenceClassification
import streamlit as st
from codetiming import Timer
import requests
import subprocess

import jsonlines
from annotated_text import annotated_text
import nltk
import torch
import gc
import os

CLAIMS_FILE = "claims.jsonl"
CORPUS_FILE = "corpus.jsonl"
PREDS_FILE = "preds.jsonl"

label_highlight_color = {'CONTRADICT': '#faa',
                         'REFUTES': '#faa',
                         'SUPPORT': '#afa',
                         'SUPPORTS': '#afa'}


@st.cache_resource
def download_models():
    nltk.download('punkt')
    model = RobertaForSequenceClassification.from_pretrained('kruthof/climateattention-10k-upscaled', num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-f")
    return model, tokenizer


def is_about_climate(texts, model, tokenizer):
    if torch.cuda.is_available():
        device = 0
        batch_size = 128
    else:
        device = -1
        batch_size = 1
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


def filter_climate_related(sentences, model, tokenizer):
    labels, _ = is_about_climate(sentences, model, tokenizer)
    return [doc for label, doc in zip(labels, sentences) if label == 'Yes']


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


def predict_with_multivers():
    gc.collect()
    cmd = 'source mult/bin/activate; python multivers/multivers/predict.py \
        --checkpoint_path=multivers/checkpoints/fever_sci.ckpt \
        --input_file="claims.jsonl" \
        --batch_size=5 \
        --corpus_file="corpus.jsonl" \
        --output_file="preds.jsonl"'
    subprocess.call(cmd, shell=True, executable='/bin/bash')


@Timer(text="get_abstracts_matching_claims elapsed time: {seconds:.0f} s")
def get_abstracts_matching_claims(claims, top_k=10, threshold=0.7):
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
def get_verified_against_phrases(claims, top_k=10, threshold=0.7):
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


def convert_evidences_from_abstracts_to_multivers_format(claims):
    evidence_abstracts = get_abstracts_matching_claims(claims)
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
    return claim_id_evidence_ids


# Create the Streamlit app
def main():
    st.set_page_config(page_title="Verify news article with Multivers",
                       page_icon=":earth_americas:",
                       layout='wide')
    model, tokenizer = download_models()

    # Add a sidebar with links
    st.sidebar.title("Omdena, Local Chapter, 🇩🇪 Cologne")
    project_link = '[Project Description](https://omdena.com/chapter-challenges/detecting-bias-in-climate-reporting-in-english-and-german-language-news-media/)'
    st.sidebar.markdown(project_link, unsafe_allow_html=True)
    github_link = '[Github Repo](https://github.com/OmdenaAI/cologne-germany-reporting-bias/)'
    st.sidebar.markdown(github_link, unsafe_allow_html=True)

    st.header("Media article scientific verificaton using Multivers")

    tab_bias_detection, tab_how_to, tab_faq = st.tabs(["Scientific verification with Multivers", "How-To", "FAQ"])

    with tab_bias_detection:

        st.write("""Enter a Text or URL below""")

        text_input = st.text_area("Enter Text")
        is_link = text_input.startswith('http')
        text_input = get_text_from_input(text_input)
        if is_link:
            with st.expander('Text that was extracted from the link and will be analyzed'):
                st.write(text_input)
        input_sentences = sent_tokenize(text_input)

        if st.checkbox('Filter climate related sentences'):
            input_sentences = filter_climate_related(input_sentences, model, tokenizer)
            if not input_sentences:
                st.warning("None of the extracted sentences are climate related.")
                st.stop()

        # Verify with Multivers
        if st.button("Verify article text with Multivers"):
            with st.expander("Climate related sentences that we'll attempt to verify"):
                for claim in input_sentences:
                    st.write(claim)

            with st.spinner(text='Retrieving relevant evidences'):
                convert_evidences_from_abstracts_to_multivers_format(input_sentences)
            with st.spinner(text='Verifying'):
                predict_with_multivers()
            with st.spinner(text='Preparing output'):
                verified_claims = get_verified_claims()
                if not verified_claims:
                    st.warning("According to Multivers model, there's \
            not enough information to verify any claim from the article")
                else:
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
                                st.markdown(f"**Abstract**: {evidence['evidence_text']}")
                        st.markdown("""---""")

        if st.button("Verify article text with ClimateBERT fine-tuned on Climate-FEVER"):
            with st.expander("Climate related sentences that we'll attempt to verify"):
                for claim in input_sentences:
                    st.write(claim)
            with st.spinner(text='Verifying the statements'):
                res = get_verified_against_phrases(input_sentences)
                if not res:
                    st.warning("According to the model, there's \
                                not enough information to verify any claim from the article")
                for claim, evidences in zip(input_sentences, res):
                    show_claim = False
                    for evidence in evidences:
                        if evidence['label'] != 'NOT_ENOUGH_INFO':
                            show_claim = True
                    if show_claim:
                        st.markdown(f"### **Claim  :orange[{claim}]**")
                        for evidence in evidences:
                            if evidence['label'] != 'NOT_ENOUGH_INFO':
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

                        # Classify text and show result
        # if st.button("Detect Global Warming stance in climate related sentences"):
        #   with st.spinner(text='Performing stance detection'):
        #     res = predict_gw_stance(input_sentences, model, tokenizer)
        #     if res is not None:
        #       st.dataframe(res, use_container_width=True)
        #     else:
        #       st.warning("None of the extracted sentences are climate related.")

        # if st.button("Highlight detected claims"):
        #   with st.spinner(text='Detecting claims'):
        #     res = predict_climate_relatedness(input_sentences, model, tokenizer)
        #     st.dataframe(res, use_container_width=True)

    with tab_how_to:
        st.write("tbd")

    with tab_faq:
        st.write("tbd")


if __name__ == "__main__":
    main()
