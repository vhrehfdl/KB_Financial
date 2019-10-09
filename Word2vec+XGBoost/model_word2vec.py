import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from konlpy.tag import *
import pandas as pd
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec
import numpy as np
import xgboost
from sklearn.metrics import classification_report
from gensim.models import Word2Vec


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train_x = train["title"].to_list()
    train_y = train["category"].tolist()

    test_x = test["title"].tolist()
    test_y = test["category"].tolist()

    return train_x, train_y, test_x, test_y


def data_preprocessing(text_list):
    okt = Okt()
    temp_list = []

    for text in text_list:
        sentence = text.strip()
        morphs = okt.nouns(sentence)
        temp_list.append(morphs)

    return temp_list


def make_feature_vector(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    word_num = 0
    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            word_num = word_num + 1
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, word_num)

    return featureVec


def text_to_vector(text_list, model, num_features):
    counter = 0
    word_embedding_vector = np.zeros((len(text_list), num_features), dtype="float32")

    for text in text_list:
        word_embedding_vector[counter] = make_feature_vector(text, model, num_features)
        counter = counter + 1

    return word_embedding_vector


def build_model(train_x, train_y):
    # clf = MultinomialNB()
    # clf.fit(X, Y)

    # clf = SVC(gamma='auto')
    # clf.fit(X, Y)

    # clf = RandomForestClassifier()
    # clf.fit(train_x, train_y)

    model = xgboost.XGBClassifier()
    model.fit(train_x, train_y, eval_metric='merror')

    return model


def evaluate(test_x, test_y, model):
    predictions = model.predict(test_x)
    print(classification_report(test_y, predictions))


def main():
    train_dir, test_dir = "./data/train_data.csv", "./data/test_data.csv"
    train_x, train_y, test_x, test_y = load_data(train_dir, test_dir)

    train_x = data_preprocessing(train_x)
    test_x = data_preprocessing(test_x)

    word2vec_model = Word2Vec(train_x, workers=4, size=300, min_count=10, iter=5)

    avg_feature_vectors_train = text_to_vector(train_x, word2vec_model, 300)
    avg_feature_vectors_test = text_to_vector(test_x, word2vec_model, 300)

    model = build_model(avg_feature_vectors_train, train_y)
    evaluate(avg_feature_vectors_test, test_y, model)


if __name__ == '__main__':
    main()

