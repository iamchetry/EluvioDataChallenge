from scripts.modelling import *

import numpy as np
import string
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import nltk
import ssl

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
lemmatizer = nltk.wordnet.WordNetLemmatizer()


def get_embedding_vector(word, w2v_model):
    try:
        vec_ = w2v_model[word]
    except KeyError:
        vec_ = None
        pass

    return vec_


def get_lemmatized_text(text):
    text_v = lemmatizer.lemmatize(text, pos='v')
    if text_v != text:
        return text_v
    else:
        text_ad = lemmatizer.lemmatize(text, pos='a')
        if text_ad != text:
            return text_ad
        else:
            text_n = lemmatizer.lemmatize(text, pos='n')
            if text_n != text:
                return text_n
            else:
                return text


def tokenize_text(text, min_word_length=None, is_word2vec=False):
    if not is_word2vec:
        return [stemmer.stem(get_lemmatized_text(_)) for _ in simple_preprocess(text) if _ not in STOPWORDS
                and len(_) >= min_word_length]
    else:
        return [_ for _ in text.translate(str.maketrans('', '', string.punctuation)).strip().split()
                if _.lower() not in STOPWORDS and len(_) >= min_word_length]


def get_bow_corpus(data_, column, min_word_length=None):
    corpus = data_[column].apply(lambda _: tokenize_text(_, min_word_length=min_word_length))
    id2word = gensim.corpora.Dictionary(corpus)
    return [id2word.doc2bow(_) for _ in corpus], id2word


def get_embeddings_for_single_doc(list_of_words, w2v_model):
    word_vecs_ = [_ for _ in [get_embedding_vector(_, w2v_model) for _ in list_of_words] if _ is not None]
    if word_vecs_:
        return list(np.around(np.average(np.array(word_vecs_), axis=0), 4))
    else:
        return None


def get_word_embeddings(data_, column, min_word_length=None, temp_column=None, w2v_model=None):
    data_[temp_column] = data_[column].apply(lambda _: tokenize_text(_, min_word_length=min_word_length,
                                                                     is_word2vec=True))
    if not w2v_model:
        w2v_model = load_word2vec_model()
    data_[temp_column] = data_[temp_column].apply(lambda _: get_embeddings_for_single_doc(_, w2v_model))
    del w2v_model
    data_.dropna(subset=[temp_column], axis=0, inplace=True)

    return data_
