embedding_dimension = 300
word_vectors = ['word_vector_{}'.format(_) for _ in range(1, embedding_dimension+1)]
hour_vectors = ['hour_{}'.format(_) for _ in range(0, 24)]
day_vectors = ['day_of_week_{}'.format(_) for _ in range(0, 7)]
month_vectors = ['month_{}'.format(_) for _ in range(1, 13)]

features = ['title', 'is_weekend', 'age_of_news', 'over_18', 'is_video_attached', 'polarity', 'author'] + \
           word_vectors + hour_vectors + day_vectors + month_vectors
output_feature = 'is_popular'
path_to_xgb_model_obj = 'model_objects/xgb_model_v0.pkl'
