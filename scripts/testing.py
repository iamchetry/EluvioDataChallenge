from scripts.text_analysis import *


def test(test_x, test_y, xgb_model=None, lda_model=None, min_word_length=None):
    corpus, id2word = get_bow_corpus(test_x, 'title', min_word_length=min_word_length)

    test_x = predict_topic_scores(test_x, model_lda=lda_model['model'], corpus=corpus, temp_score_col='topic_scores',
                                  topic_dim=lda_model['no_of_topics'])
    features_ = xgb_model['features']
    missing_cols = list(set(features_).difference(set(test_x.columns)))

    for _ in missing_cols:
        print(_)
        test_x[_] = 0
        test_x[_] = test_x[_].astype('int8')

    test_x['is_popular_probability'] = list(xgb_model['model'].predict_proba(test_x[features_])[:, 1])
    test_x['is_popular_probability'] = test_x['is_popular_probability'].apply(lambda _: round(_, 4))
    test_x['is_popular_predicted'] = xgb_model['model'].predict(test_x[features_])

    pd.concat([test_x[['is_popular_predicted', 'is_popular_probability']], test_y], axis=1, ignore_index=False).to_csv(
        'data/test_predictions.csv', index=False)

    return pd.concat([test_x, test_y], axis=1, ignore_index=False)
