import streamlit as st
from nltk.tokenize import sent_tokenize
import sys

sys.path.append("..")
import Scientific_Verification as main_page


def main():
    main_page.add_sidebar()
    main_tab, tab_faq, tab_how_to = set_header_and_tabs()

    with main_tab:
        if 'input_text' in st.session_state:
            init_value = st.session_state.input_text
        else:
            init_value = ''

        text_input = st.text_area(label="""Enter a Text below and click the Classify Button 
        to extract Climate Change related  sentences from text and classify them
        as agreeing with Global warming, disagreeing with Global Warming or neutral""",
                                  value=init_value)

        input_sentences = sent_tokenize(text_input)

        # Classify text and show result
        if st.button("Detect Global Warming stance in climate related sentences"):
            with st.spinner(text='Performing stance detection'):
                res = main_page.predict_gw_stance(input_sentences)
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
                        st.table(res.style.applymap(style_agree, props='color:white;background-color:green')
                                 .applymap(style_disagree, props='color:white;background-color:red'))
                else:
                    st.warning("None of the extracted sentences are climate related.")


def style_agree(v, props=''):
    return props if v in ['agree'] else None


def style_disagree(v, props=''):
    return props if v in ['disagree'] else None


def set_header_and_tabs():
    st.header("Media article Global Warming Stance detection")
    tab_bias_detection, tab_how_to, tab_faq = \
        st.tabs(["Global Warming Stance detection", "How-To", "FAQ"])
    return tab_bias_detection, tab_faq, tab_how_to


if __name__ == "__main__":
    main()
