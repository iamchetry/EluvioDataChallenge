from scripts.preprocessing import *
from datetime import datetime


def convert_categories_to_dummies(data_, columns_list=None):
    for _ in columns_list:
        data_ = pd.concat([data_.drop(columns=_, axis=1), pd.get_dummies(data_[_], prefix=_).astype('int8')], axis=1,
                          ignore_index=False)
    return data_


def create_derived_features(data_):
    data_.time_created = data_.time_created.astype('int')

    data_['time_created'] = data_.time_created.astype('int').apply(lambda _: datetime.utcfromtimestamp(_))
    data_['hour'] = data_.time_created.apply(lambda _: _.hour).astype('int8')
    data_['day_of_week'] = data_.time_created.apply(lambda _: _.weekday()).astype('int8')
    data_['is_weekend'] = data_.day_of_week.apply(lambda _: 1 if _ in [5, 6] else 0).astype('int8')
    data_['month'] = data_.time_created.apply(lambda _: _.month).astype('int8')
    data_['age_of_news'] = data_.time_created.apply(lambda _: datetime.now().year - _.year).astype('int8')
    data_.drop(columns='time_created', axis=1, inplace=True)
    data_['over_18'] = data_.over_18.astype('bool').map({True: 1, False: 0}).astype('int8')
    data_['is_video_attached'] = is_video_attached(data_, 'title')
    data_['title'] = clear_video_tags(data_, 'title')
    data_['polarity'] = data_.title.apply(get_polarity)
    data_ = convert_categories_to_dummies(data_, columns_list=['hour', 'day_of_week', 'month'])

    return data_


def get_author_columns(data_, column):
    return pd.concat([data_.drop(columns=column, axis=1), pd.get_dummies(data_[column], prefix=column).astype('int8')],
                     axis=1, ignore_index=False)


def split_embedding_vector(data_, column, dimension=None):
    return pd.concat([data_.drop(columns=column, axis=1), (pd.DataFrame(
        data_[column].tolist(), index=data_.index,
        columns=['word_vector_{}'.format(_) for _ in range(1, dimension+1)])).applymap(lambda x: round(x, 3))], axis=1,
                     ignore_index=False)
