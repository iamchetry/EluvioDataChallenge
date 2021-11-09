from scripts.training import *
from scripts.testing import *

X_test, y_test, model_lda = train(data_path='data/Eluvio_DS_Challenge.csv', chunksize=600000,
                                  desired_rows=300000,
                                  columns_to_drop=['down_votes', 'category'],
                                  column_to_detect_outlier='up_votes', min_word_length=4,
                                  topics_list=[3, 5, 7, 9, 11, 13, 15], coherence_method='u_mass',
                                  passes_lda=1,
                                  path_to_lda_model_obj='model_objects/lda_model.pkl',
                                  path_to_log_file='logs/log.txt',
                                  path_to_xgb_model_obj='model_objects/xgb_model.pkl')

xgb_model = load_from_pickle(path_to_xgb_model_obj)

test_ = test(X_test, y_test, xgb_model=xgb_model, lda_model=model_lda, min_word_length=4)

