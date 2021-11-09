from scripts.helper_functions import *

import os
import pandas as pd
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import cross_val_score
import gensim
import gensim.downloader as api


def load_word2vec_model():
    return api.load('word2vec-google-news-300')


def lda_model(corpus=None, num_topics=None, id2word=None, passes=None, workers=None):
    return gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=id2word, passes=passes, workers=workers)


def get_coherence(model=None, corpus=None, coherence_method=None):
    return gensim.models.coherencemodel.CoherenceModel(model=model, corpus=corpus,
                                                       coherence=coherence_method).get_coherence()


def get_coherence_for_lda_models(corpus=None, topics_list=None, id2word=None, coherence_method=None, passes=None,
                                 workers=None):
    coherence_list = list()
    for topic_ in topics_list:
        model_ = lda_model(corpus=corpus, num_topics=topic_, id2word=id2word, passes=passes, workers=workers)
        coh_ = get_coherence(model=model_, corpus=corpus, coherence_method=coherence_method)
        coherence_list.append(coh_)

    return dict(zip(topics_list, coherence_list))


def predict_topic_scores(data_, model_lda=None, corpus=None, temp_score_col=None, topic_dim=None):
    data_[temp_score_col] = [[round(__[-1], 3) for __ in model_lda[_]] for _ in corpus]

    return pd.concat([data_.drop(columns=temp_score_col, axis=1), (pd.DataFrame(
        data_[temp_score_col].tolist(), index=data_.index,
        columns=['topic_{}_score'.format(_) for _ in range(1, topic_dim + 1)])).applymap(lambda x: round(x, 3))],
                     axis=1, ignore_index=False)


def get_best_params(train_data, features, y, path_to_log_file):
    space = {
        'max_depth': hp.choice('max_depth', [6, 10, 15]),
        'n_estimators': hp.choice('n_estimators', [200, 300]),
        'reg_alpha': hp.choice('reg_alpha', [0.01, 0.1]),
        'reg_lambda': hp.choice('reg_lambda', [0.1, 0.5]),
        'min_child_weight': hp.choice('min_child_weight', [3, 10]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.5, 1]),
        'scale_pos_weight': hp.choice('scale_pos_weight', [6, 10, 15])}

    def objective_xgboost(params):
        params = {
            'max_depth': int(params['max_depth']),
            'min_child_weight': int(params['min_child_weight']),
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'colsample_bytree': params['colsample_bytree'],
            'scale_pos_weight': int(params['scale_pos_weight']),
            'n_estimators': int(params['n_estimators'])
        }

        clf = XGBClassifier(
            learning_rate=0.08,
            n_jobs=os.cpu_count() - 1,
            **params
        )

        score = cross_val_score(clf, train_data[features], y, scoring='f1', cv=4).mean()

        with open(path_to_log_file, 'a') as myfile:
            myfile.write('Params : {}\n'.format(params))
            myfile.write('F1_Score : {}\n'.format(score))
            myfile.write('===========================================\n')

        return {'loss': 1 - score, 'status': STATUS_OK}

    return fmin(fn=objective_xgboost, space=space, algo=tpe.suggest, max_evals=288)


def get_best_model(train_data=None, features=None, y=None, path_to_xgb_model=None, path_to_log_file=None):
    print('------------ Started Tuning XGB Model ------------')
    params_ = get_best_params(train_data, features, y, path_to_log_file)
    model_ = XGBClassifier(**params_)
    model_.fit(train_data[features], y)
    dump_as_pickle(object={'model': model_, 'features': features}, filename=path_to_xgb_model)
