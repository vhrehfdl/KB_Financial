from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import numpy as np


def label_distribution_table(train, test):
    le = preprocessing.LabelEncoder()
    genre_num = le.fit_transform(train["sentiment"])
    print(genre_num)

    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(genre_num)
    plt.show()


def text_length_analysis(data):
    train_length = data["cleaned_tweets"].astype(str).apply(len)

    print('텍스트 길이 최대값: {}'.format(np.max(train_length)))
    print('텍스트 길이 평균값: {}'.format(np.mean(train_length)))
    print('텍스트 길이 표준편차: {}'.format(np.std(train_length)))
    print('텍스트 길이 중간값: {}'.format(np.median(train_length)))
    print('텍스트 길이 제1사분위: {}'.format(np.percentile(train_length, 25)))
    print('텍스트 길이 제3사분위: {}'.format(np.percentile(train_length, 75)))


def wordcloud(data):
    cloud = WordCloud(width=800, height=600).generate(" ".join(data["cleaned_tweets"].astype(str)))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()


def main():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    # # 라벨링 분포표를 막대그래프로 보여준다.
    # label_distribution_table(train, test)

    # # 텍스트의 최대값, 평균관, 표준편차 등을 보여준다.
    # text_length_analysis(train)

    # # 텍스트의 Word Cloud를 만들어 보여준다.
    wordcloud(train)


if __name__ == '__main__':
    main()