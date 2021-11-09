from scripts.helper_functions import *

import numpy as np
import pandas as pd


def read_data(data_path=None, chunksize=None):
    for data_ in pd.read_csv(data_path, chunksize=chunksize):
        data_.dropna(subset=['title'], axis=0, inplace=True)
        data_.drop_duplicates(inplace=True)
        return data_


def preprocess(data_=None, columns_to_drop=None, column_to_detect_outlier=None):
    data_.drop(columns=columns_to_drop, axis=1, inplace=True)
    range_dict = detect_outlier(data_, column_to_detect_outlier)
    data_['is_popular'] = np.where((data_.up_votes >= range_dict['min']) & (data_.up_votes <= range_dict['max']), 0, 1
                                   ).astype('int8')
    del range_dict

    return data_


def clean_titles(text):
    return text.strip().replace('(video', '').replace('[video', '').replace('( video', '').replace(
        '[ video', '').replace('-video', '').replace('- video', '').replace('(Video', '').replace('[Video', '').replace(
        '( Video', '').replace('[ Video', '').replace('-Video', '').replace('- Video', '').replace(
        '(VIDEO', '').replace('[VIDEO', '').replace('( VIDEO', '').replace('[ VIDEO', '').replace('-VIDEO', '').replace(
        '- VIDEO', '')


def clear_video_tags(data_, column):
    return data_[column].map(clean_titles)


