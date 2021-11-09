from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import numpy as np
import pickle

analyser = SentimentIntensityAnalyzer()


def dump_as_pickle(object=None, filename=None):
    with open(filename, 'wb') as file_:
        pickle.dump(object, file_)


def load_from_pickle(filename=None):
    with open(filename, 'rb') as file_:
        return pickle.load(file_)


def detect_outlier(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    range_ = q3 - q1

    return {'min': q1 - 1.5 * range_, 'max': q3 + 1.5 * range_}


def get_polarity(text):
    return round(analyser.polarity_scores(text)['compound'], 3)


def is_video_attached(data, column):
    column_series = data[column].apply(lambda _: _.lower())

    return (column_series.str.contains('(video', regex=False) | column_series.str.contains('[video', regex=False) |
            column_series.str.contains('( video', regex=False) | column_series.str.contains('[ video', regex=False) |
            column_series.str.contains('-video', regex=False) | column_series.str.contains('- video', regex=False)).map(
        {True: 1, False: 0}).astype('int8')


def get_optimal_topic(dict_):
    keys_ = list(dict_.keys())
    values_ = list(dict_.values())
    return keys_[np.argmax(values_)]
