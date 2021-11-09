from api.v1.helper_functions import *
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def home():
    return "======== Welcome to the World of Popularity Prediction ========"


@app.route('/predict_popularity', methods=['GET', 'POST'])
def predict():
    try:
        json_ = request.json
        json_ = {_: [json_[_]] for _ in json_ if _ not in ['down_votes', 'category']}
        return live_prediction(json_=json_, min_word_length=4, embedding_dimension=300)
    except Exception as e:
        print(e)
