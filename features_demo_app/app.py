import streamlit as st
import openai
import re
import nltk
nltk.download('punkt')
from nltk import tokenize
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

# nltk.download('omw-1.4')

api_key = st.text_input(label='Enter your API Key',)


# # def clean_text(string):
# #     new_string = re.sub('[^a-zA-Z0-9 \n\.]','',string)
# #     return new_string
# def clean_text(doc):
#     # p = re.compile(r"^Speaker$", re.IGNORECASE)
#     # cleaned_doc = p.sub(' ', doc)
#     cleaned_doc = re.sub(r'\d+', '', doc)
#     cleaned_doc = re.sub('[^A-Za-z0-9]+', ' ', doc)
#     return cleaned_doc

def summarize(doc):
    # Object of automatic summarization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer.
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    # Set delimiter for making a list of sentence.
    auto_abstractor.delimiter_list = [".", "\n"]
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    result_dict = auto_abstractor.summarize(doc, abstractable_doc)
    summary = result_dict["summarize_result"]
    summary = ' '.join(summary)
    return summary

# def generate(text,userPrompt="Extract key insights:"):
#     openai.api_key = api_key
#     response = openai.Completion.create(
#     engine="text-davinci-003",
#     prompt=userPrompt + "\n\n" + text,
#     temperature=0.7,
#     max_tokens=250,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     for r in response['choices']:
#         print(r['text'])
#     return response

def ideas(text,userPrompt="Extract the pain points and pleasure points without numbering and bullet points:"):
    openai.api_key = api_key
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.6,
    max_tokens=300,
    top_p=1.0,
    frequency_penalty=1,
    presence_penalty=1
    )
    for r in response['choices']:
        print(r['text'])
    return response


def tldr(text,userPrompt="\n\nTl;dr"):
    openai.api_key = api_key
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt= text + userPrompt,
      temperature=0.7,
      max_tokens=2000,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response

    # topics covered as single words

def topics(text,userPrompt="Highlight the most important single-word topics:"):
    openai.api_key = api_key
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.6,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=1,
    presence_penalty=1
    )
    for r in response['choices']:
        print(r['text'])
    return response

def sentiment(text,userPrompt="Predict the overall sentiment of extracted insights:"):
    openai.api_key = api_key
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.6,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=1,
    presence_penalty=1
    )
    for r in response['choices']:
        print(r['text'])
    return response


def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    Code Credit: https://github.com/soft-nougat/dqw-ivves
    function to display major headers at user interface
    :param main_txt: the major text to be displayed
    :param sub_txt: the minor text to be displayed
    :param is_sidebar: check if its side panel or major panel
    :return:
    """
    html_temp = f"""
    <h2 style = "text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    else:
        st.markdown(html_temp, unsafe_allow_html=True)

def divider():
    """
    Sub-routine to create a divider for webpage contents
    """
    st.markdown("""---""")


@st.cache
def tldr_summary(doc):
    return tldr(doc)


@st.cache
def subjects(doc):
    return topics(doc)

@st.cache
def summary(doc):
    return summarize(doc)

@st.cache
def sent(doc):
    return sentiment(doc)


@st.cache
def generate_insights(text,userPrompt="Extract the pain points and pleasure points without numbering and bullet points:"):
    return ideas(text,userPrompt="Extract the pain points and pleasure points without numbering and bullet points:")

def main():
    st.write("""
    # GPT-3 Text Processing Demo
    """)
    input_help_text = """
    Enter Text
    """
    final_message = """
    The data was successfully analyzed
    """
    text = st.text_area(label='INPUT TEXT',placeholder="Enter Sample Text")
    text = summary(text)
    # text1 = st.text_area(label='INPUT TEXT1',placeholder="Enter Sample Text")
    # text2 = st.text_area(label='INPUT TEXT2',placeholder="Enter Sample Text")
    # text3 = st.text_area(label='INPUT TEXT3',placeholder="Enter Sample Text")
    # text4 = st.text_area(label='INPUT TEXT4',placeholder="Enter Sample Text")
    # text = clean_txt(text)
    # text1 = summary(text1)
    # text2 = summary(text2)
    # text3 = summary(text3)
    # text4 = summary(text4)

    # input = []
    # input.append(text1)
    # input.append(text2)
    # input.append(text3)
    # input.append(text4)
    # st.write(text)
    
    with st.sidebar:
        st.markdown("**Processing**")
        insights = st.button(
            label="Extract Insights",
            help=""
        )
        tldr = st.button(
            label="TL;DR",
            help=""
        )
        subject = st.button(
            label="Topics",
            help=""
        )
        disposition = st.button(
            label="Sentiment",
            help=""
        )

    if insights:
        st.markdown("#### Key Insights")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = generate_insights(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence)
            # else:
            output = generate_insights(text).get("choices")[0]['text']
            # st.write(output)
            text_sentences = tokenize.sent_tokenize(output)
            for sentence in text_sentences:
                st.write(sentence) 

    if tldr:
        st.markdown("#### TLDR")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = tldr_summary(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence)
            # else:
            output = tldr_summary(text).get("choices")[0]['text']
            # st.write(output)
            # text_sentences = tokenize.sent_tokenize(output)
            # for sentence in text_sentences:
            st.write(output) 

    if subject:
        st.markdown("#### Topics")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = subjects(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence)
            # else:
            output = subjects(text).get("choices")[0]['text']
            # st.write(output)
            text_sentences = tokenize.sent_tokenize(output)
            for sentence in text_sentences:
                st.write(sentence)
        
    if disposition:
        st.markdown("#### Sentiment")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = subjects(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence)
            # else:
            output = sent(text).get("choices")[0]['text']
            # st.write(output)
            # text_sentences = tokenize.sent_tokenize(output)
            # for sentence in text_sentences:
            st.write(output)

if __name__ == '__main__':
    main()

# gcloud builds submit --tag gcr.io/text-tagging-api/insights
# gcloud run deploy --image gcr.io/text-tagging-api/insights --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/featuresdemo
# gcloud run deploy --image gcr.io/insight7-353714/featuresdemo --platform managed

