from scripts.feature_creation import *
from scripts.text_analysis import *

from flask import jsonify

import pandas as pd

print('Loading models will take some time..')
print('Please wait for a while')
print('....')
lda_model = load_from_pickle(filename='model_objects/lda_model.pkl')
print('Topic-Model Loaded')
xgb_model = load_from_pickle(filename='model_objects/xgb_model_v0.pkl')
print('XGB Model Loaded')
print('Word2Vec Model Getting Loaded..')
w2v_model = load_word2vec_model()
print('All Models Loaded')


def live_prediction(json_=None, min_word_length=None, embedding_dimension=None):
    data_ = pd.DataFrame(json_)
    data_ = create_derived_features(data_)
    data_ = get_word_embeddings(data_, 'title', min_word_length=min_word_length, temp_column='word_vector',
                                w2v_model=w2v_model)
    if not len(data_):
        return jsonify({'response': 'Word Embeddings not obtained', 'status': 404})
    data_ = split_embedding_vector(data_, 'word_vector', dimension=embedding_dimension)
    corpus, id2word = get_bow_corpus(data_, 'title', min_word_length=min_word_length)
    data_ = predict_topic_scores(data_, model_lda=lda_model['model'], corpus=corpus, temp_score_col='topic_scores',
                                 topic_dim=lda_model['no_of_topics'])
    features_ = xgb_model['features']
    missing_cols = list(set(features_).difference(set(data_.columns)))
    for _ in missing_cols:
        data_[_] = 0
        data_[_] = data_[_].astype('int8')
    data_['is_popular_probability'] = list(xgb_model['model'].predict_proba(data_[features_])[:, 1])
    data_['is_popular_probability'] = data_['is_popular_probability'].apply(lambda _: round(_, 4))
    data_['is_popular'] = xgb_model['model'].predict(data_[features_])
    output_ = data_[['title', 'author', 'is_popular', 'is_popular_probability']].to_dict(
        orient='records')

    return jsonify({'response': 'Prediction of Popularity is Successful',
                    'data': output_,
                    'status': 200})
