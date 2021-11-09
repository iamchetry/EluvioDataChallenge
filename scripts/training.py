from scripts.text_analysis import *
from scripts.feature_creation import *
from scripts.preprocessing import *
from scripts.constants import *

from sklearn.model_selection import train_test_split
import os


def split_train_test(data_, feature_vector=None, output_column=None):
    X_train, X_test, y_train, y_test = train_test_split(data_[feature_vector], data_[output_column], test_size=0.3,
                                                        stratify=data_[output_column], random_state=150)
    return X_train, X_test, y_train, y_test


def train(data_path=None, chunksize=None, desired_rows=None, columns_to_drop=None, column_to_detect_outlier=None,
          min_word_length=None, topics_list=None, coherence_method=None, passes_lda=None, path_to_lda_model_obj=None,
          path_to_xgb_model_obj=None, path_to_log_file=None):

    data_ = read_data(data_path=data_path, chunksize=chunksize)

    length_ = len(data_)
    if length_ > desired_rows:
        data_ = data_.sample(n=desired_rows, random_state=150)

    data_ = preprocess(data_=data_, columns_to_drop=columns_to_drop, column_to_detect_outlier=column_to_detect_outlier)

    data_ = create_derived_features(data_)

    data_ = get_word_embeddings(data_, 'title', min_word_length=min_word_length, temp_column='word_vector')

    data_ = split_embedding_vector(data_, 'word_vector', dimension=embedding_dimension)

    X_train, X_test, y_train, y_test = split_train_test(data_, feature_vector=features, output_column=output_feature)

    train_corpus, id2word = get_bow_corpus(X_train, 'title', min_word_length=min_word_length)

    coh_dict = get_coherence_for_lda_models(corpus=train_corpus, topics_list=topics_list, id2word=id2word,
                                            coherence_method=coherence_method, passes=passes_lda,
                                            workers=os.cpu_count()-1)

    topic_number = get_optimal_topic(coh_dict)

    model_lda = lda_model(corpus=train_corpus, num_topics=topic_number, id2word=id2word, passes=passes_lda,
                          workers=os.cpu_count()-1)
    dump_as_pickle(object={'model': model_lda, 'no_of_topics': topic_number}, filename=path_to_lda_model_obj)

    model_lda = load_from_pickle(path_to_lda_model_obj)
    topic_number = model_lda['no_of_topics']
    model_lda = model_lda['model']

    X_train = predict_topic_scores(X_train, model_lda=model_lda, corpus=train_corpus, temp_score_col='topic_scores',
                                   topic_dim=topic_number)

    features_ = ['is_weekend', 'age_of_news', 'over_18', 'is_video_attached', 'polarity'] + word_vectors + \
                hour_vectors + day_vectors + month_vectors + ['topic_{}_score'.format(_) for _ in range(
                 1, topic_number + 1)]

    X_train.drop(columns='title', axis=1, inplace=True)

    xgb_model = load_from_pickle(path_to_xgb_model_obj)
    X_train['is_popular_probability'] = list(xgb_model['model'].predict_proba(X_train[features_])[:, 1])
    X_train['is_popular_probability'] = X_train['is_popular_probability'].apply(lambda _: round(_, 4))
    X_train['is_popular_predicted'] = xgb_model['model'].predict(X_train[features_])

    pd.concat([X_train[['is_popular_predicted', 'is_popular_probability']], y_train], axis=1, ignore_index=True).to_csv(
        'data/train_predictions.csv', index=False)

    get_best_model(train_data=X_train, features=features_, y=y_train, path_to_xgb_model=path_to_xgb_model_obj,
                   path_to_log_file=path_to_log_file)

    return X_test, y_test, model_lda
