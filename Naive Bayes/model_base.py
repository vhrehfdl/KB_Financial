import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from konlpy.tag import *
import xgboost


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train_x = train["text"].tolist()
    train_y = train["senti"].tolist()

    test_x = test["text"].tolist()
    test_y = test["senti"].tolist()

    return train_x, train_y, test_x, test_y


# def data_preprocessing(text):
#     okt = Okt()
#     sentence_list = []
#
#     for i in text:
#         sentence = i.strip()
#         morphs = okt.nouns(sentence[:100])
#         temp_sentence = ""
#
#         for temp_word in morphs:
#             temp_sentence = temp_sentence + temp_word + " "
#
#         sentence_list.append(temp_sentence)
#
#     return sentence_list


def text_to_vector(train_x, test_x):
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)

    return train_x, test_x


def build_model(train_x, train_y):
    # clf = MultinomialNB()
    # clf.fit(X, Y)

    # clf = SVC(gamma='auto')
    # clf.fit(X, Y)

    # clf = RandomForestClassifier()
    # clf.fit(train_x, train_y)

    clf = xgboost.XGBClassifier(n_estimators=2000, scale_pos_weight=1)
    clf.fit(train_x, train_y, eval_metric='merror')

    return clf


def evaluate(test_x, test_y, model):
    pred_list = []

    for i in test_x:
        prediction = model.predict(i)
        pred_list.append(prediction[0])

    print(classification_report(test_y, pred_list))


def main():
    train_dir, test_dir = "./data/train.csv", "./data/test.csv"
    train_x, train_y, test_x, test_y = load_data(train_dir, test_dir)
    # train_x = data_preprocessing(train_x)
    # test_x = data_preprocessing(test_x)
    train_x, test_x = text_to_vector(train_x, test_x)
    model = build_model(train_x, train_y)
    evaluate(test_x, test_y, model)


if __name__ == '__main__':
    main()

