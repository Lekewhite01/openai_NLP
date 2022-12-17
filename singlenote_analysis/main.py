import openai
import textwrap
import re
import nltk
from nltk import tokenize
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from flask import Flask, request
from flask import jsonify
import os
from flask_cors import CORS, cross_origin
nltk.download('punkt')
api_key = ''

app = Flask(__name__)
cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/suggest', methods=['POST'])
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        data = str(request.data)
        # data = clean_text(data)
        # Object of automatic summarization.
        auto_abstractor = AutoAbstractor()
        # Set tokenizer.
        auto_abstractor.tokenizable_doc = SimpleTokenizer()
        # Set delimiter for making a list of sentence.
        auto_abstractor.delimiter_list = [".", "\n"]
        # Object of abstracting and filtering document.
        abstractable_doc = TopNRankAbstractor()
        # Summarize document.
        result_dict = auto_abstractor.summarize(data, abstractable_doc)
        summary = result_dict["summarize_result"]
        summary = ' '.join(summary)
        output = ideas(summary).get("choices")[0]['text']
        text_sentences = tokenize.sent_tokenize(output)
        results = []
        for sentence in text_sentences:
            results.append(sentence)
        response = jsonify(recommendations=results)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        return 'Content-Type not supported!'

# def clean_text(doc):
#     # p = re.compile(r"^Speaker$", re.IGNORECASE)
#     # cleaned_doc = p.sub(' ', doc)
#     cleaned_doc = re.sub(r'\d+', '', doc)
#     cleaned_doc = re.sub('[^A-Za-z0-9]+', ' ', doc)
#     return cleaned_doc

# def text_wrap(doc):
#     wrapper = textwrap.TextWrapper(width=2000)
#     word_list = wrapper.wrap(text=doc)
#     return word_list

#---PROMPTS
# Exctract the pain points from this text:
# Classify the insights in this text based on sentiment:
# Answer this question:
# Summarize this text and extract key insights:
# chunk this text into 10 paragraphs:
# predict overall sentiment:

def ideas(text,userPrompt="Extract key insights without numbering and bullet points:"):
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



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/text-tagging-api/suggest
# gcloud run deploy --image gcr.io/text-tagging-api/suggest --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/gpt3processor
# gcloud run deploy --image gcr.io/insight7-353714/gpt3processor --platform managed
