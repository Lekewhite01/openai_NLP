from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request
from flask import jsonify
import os
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/keywords', methods=['POST'])
# @cross_origin(supports_credentials=True)
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        data = str(request.data)
        output = process(data)
        # response.headers.add('Access-Control-Allow-Origin', '*')
        # return jsonify(Keywords=response)
        return jsonify(Keywords=output)
    else:
        return 'Content-Type not supported!'

def process(doc):
    n_gram_range = (1, 1)
    stop_words = "english"
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names_out()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    top_n = 50
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    return keywords

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/insight7-353714/wordcloud_keywords
# gcloud run deploy --image gcr.io/insight7-353714/wordcloud_keywords --platform managed
