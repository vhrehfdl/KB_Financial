import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train_x = train["cleaned_tweets"].astype(str).to_list()
    train_y = train["sentiment"].tolist()

    test_x = test["cleaned_tweets"].tolist()
    test_y = test["sentiment"].tolist()

    return train_x, train_y, test_x, test_y


def text_to_vector(train_x, test_x):
    vectorizer = CountVectorizer()
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)

    return train_x, test_x


def build_model(train_x, train_y):
    # clf = MultinomialNB()
    # clf.fit(X, Y)

    # clf = SVC(gamma='auto')
    # clf.fit(X, Y)

    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)

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
    train_x, test_x = text_to_vector(train_x, test_x)
    print(train_x)
    model = build_model(train_x, train_y)
    evaluate(test_x, test_y, model)


if __name__ == '__main__':
    main()

