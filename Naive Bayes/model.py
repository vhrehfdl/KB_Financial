import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
import math


def load_data(train_dir, test_dir):
    train_data = pd.read_csv(train_dir)
    test_data = pd.read_csv(test_dir)

    train = train_data.to_dict(orient='records')
    test = test_data.to_dict(orient='records')

    return train, test


# P(C) 확률 구하기
def make_prior_prob(train_data):
    pos_num, neg_num = 0, 0
    for i in range(0, len(train_data)):
        if train_data[i]["senti"] == "neg":
            pos_num += 1
        else:
            neg_num += 1

    prob_pos = pos_num / (pos_num + neg_num)
    prob_neg = neg_num / (pos_num + neg_num)

    return prob_pos, prob_neg


# pos 리뷰, neg 리뷰 중복 단어 카운트 하고 pos_dic, neg_dic에 넣기
# pos 리뷰, neg 리뷰 단어수 카운트하기
def make_likelihood(train):
    pos_dic, neg_dic = {}, {}
    pos_all_count, neg_all_count = 0, 0

    for i in range(0, len(train)):
        token_list = word_tokenize(train[i]["text"].strip())
        if train[i]["senti"] == "pos":
            for w in token_list:
                pos_all_count += 1  # pos 리뷰에 사용된 전체 단어수 카운팅하기

                if w in pos_dic:
                    pos_dic[w] += 1
                else:
                    pos_dic[w] = 1

        else:
            for w in token_list:
                neg_all_count += 1  # neg 리뷰에 사용된 전체 단어수 카운팅하기

                if w in neg_dic:
                    neg_dic[w] += 1
                else:
                    neg_dic[w] = 1

    return pos_dic, neg_dic, pos_all_count, neg_all_count


def predict(test, pos_dic, neg_dic, pos_all_count, neg_all_count, prob_pos, prob_neg):
    pred_list = []

    for i in range(0, len(test)):
        temp_sentence = test[i]["text"]
        temp_token = word_tokenize(temp_sentence)

        # 해당 리뷰가 pos 리뷰일 확률 구하기
        pos_cal = math.log(prob_pos)  # Log(P(pos))
        for j in range(0, len(temp_token)):
            if temp_token[j] in pos_dic:
                pos_cal += math.log(((pos_dic[temp_token[j]] + 1) / pos_all_count))  # Log(P(pos)) + Log(P(Word1|pos) + Log(P(Word2|pos) ...
            else:
                pos_cal += math.log((1 / pos_all_count))  # train 데이터에 존재하지 않는 단어 예외 처리

        # 해당 리뷰가 neg 메일일 확률 구하기
        neg_cal = math.log(prob_neg)  # Log(P(neg))
        for k in range(0, len(temp_token)):
            if temp_token[k] in neg_dic:
                neg_cal += math.log(((neg_dic[temp_token[k]]+1) / neg_all_count))  # Log(P(neg)) + Log(P(Word1|neg) + Log(P(Word2|neg) ...
            else:
                neg_cal += math.log((1 / neg_all_count))  # train 데이터에 존재하지 않는 단어 예외 처리

        # 확률 비교해서 해당 리뷰가 pos or neg으로 결정하기
        if pos_cal > neg_cal:
            pred_list.append("pos")
        else:
            pred_list.append("neg")

    return pred_list


def main():
    train_dir = "./data/train.csv"
    test_dir = "./data/test.csv"

    train, test = load_data(train_dir, test_dir)
    prob_pos, prob_neg = make_prior_prob(train)    # P(pos), P(neg) 확률 구하기
    pos_dic, neg_dic, pos_all_count, neg_all_count = make_likelihood(train)   # 중복 단어 카운팅하고 pos 리뷰와 neg 리뷰에 사용된 단어 수 구하기

    pred_list = predict(test, pos_dic, neg_dic, pos_all_count, neg_all_count, prob_pos, prob_neg)  # test 데이터에 메일 분류하기
    print(pred_list)

    test_label_list = []
    for i in range(0, len(test)):
        test_label_list.append(test[i]["senti"])
    print(test_label_list)

    print(classification_report(test_label_list, pred_list))


if __name__ == '__main__':
    main()