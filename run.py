from summa.summarizer import summarize
from data import Summaries
import logging
from rouge import Rouge
from nltk.tokenize import sent_tokenize
import spacy
import re
import pandas as pd
import pytextrank
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
import pandas as pd
import csv
import json

# nlp = spacy.load('en_core_web_sm')
logging.basicConfig(level=logging.INFO)

def print_scores(scores):
    toprint = ""
    for key in scores.keys():
        toprint += " & " + str(round(scores[key]['f']*100,2))

    print(toprint)

def unique_ngrams():
    duc = [0.12156329, 0.45697504, 0.64084184, 0.73831058]
    legal = [0.41459283, 0.78600083, 0.88397464, 0.92316188]
    ind = np.arange(4)
    width = .34
    fig1, ax1 = plt.subplots()
    ax1.set_title('Unique n-grams')
    plt.ylabel('% that are unique')
    plt.bar(ind, duc, width, label="DUC2002")
    plt.bar(ind + width + .02, legal, width, label="Ours")
    plt.xticks(ind + width/2 + 0.01, ('1-grams', '2-grams', '3-grams', '4-grams'))
    plt.legend(loc='best')
    fig1.set_size_inches(5, 3.5)

    fig1.show()

def save_legaltldr():
    df = pd.read_csv("data/tldrlegal_v1.csv")
    df.set_index('uid', inplace=True)
    export = df.to_json(r'tldrlegal_v1.json', orient='index')

    pass

def save_tosdr():
    reader = csv.DictReader(open("data/tosdr_annotated.csv"))
    data = {}
    for row in reader:
        if row['tldr_code'].startswith("1") \
             or row['title_code'].startswith("1") \
             or row['case_code'].startswith("1"):

            data[row['uid']] = {}
            data[row['uid']]['note'] = row['note']
            data[row['uid']]['name'] = row['quoteDoc']
            data[row['uid']]['text'] = row['text']
            data[row['uid']]['urls'] = row['urls']

            summ = False
            onethrow = False
            twothrow = False

            if row['tldr_code'].startswith("1") \
                    or row['tldr_code'].startswith("2") \
                    or row['tldr_code'].startswith("3"):
                data[row['uid']]['tldr_text'] = row['tldr']
                data[row['uid']]['tldr_code'] = row['tldr_code']

                if row['tldr_code'].startswith("1"):
                    if len(row['tldr']) > len(row['text']):
                        onethrow = True
                    else:
                        data[row['uid']]['summary'] = row['tldr']
                        summ = True

            if row['case_code'].startswith("1") \
                    or row['case_code'].startswith("2") \
                    or row['case_code'].startswith("3"):
                data[row['uid']]['case_text'] = row['case']
                data[row['uid']]['case_code'] = row['case_code']
                if not summ:

                    if row['case_code'].startswith("1"):
                        if len(row['case']) > len(row['text']):
                            onethrow = True
                        else:
                            data[row['uid']]['summary'] = row['case']
                            summ = True
                    # if onethrow:
                    #     if row['case_code'].startswith("2"):
                    #         if len(row['case']) > len(row['text']):
                    #             twothrow = True
                    #         else:
                    #             data[row['uid']]['summary'] = row['case']
                    #             summ = True

            if row['title_code'].startswith("1") \
                    or row['title_code'].startswith("2") \
                    or row['title_code'].startswith("3"):

                data[row['uid']]['title_text'] = row['title']
                data[row['uid']]['title_code'] = row['title_code']

                if row['title_code'].startswith("1"):
                    if len(row['title']) > len(row['text']):
                        onethrow = True
                    else:
                        data[row['uid']]['summary'] = row['title']
                        summ = True
                # elif onethrow:
                #     if row['title_code'].startswith("2"):
                #         if len(row['title']) > len(row['text']):
                #             twothrow = True
                #         else:
                #             data[row['uid']]['summary'] = row['title']
                #             summ = True
                # elif twothrow:
                #     if row['title_code'].startswith("3"):
                #         if len(row['title']) > len(row['text']):
                #             data[row['uid']]['summary'] = row['title']
                #             summ = True


            # if not summ:
            #     print(f"throw {row['uid']}")
            #     data.pop(row['uid'])

    json.dump(data, open("tosdr_v1.json", "w"))

    pass




if __name__ == '__main__':
    data2 = Summaries(['tosdr_annotated'])


    save_tosdr()

    # unique_ngrams()

    #
    # data1 = Summaries(['tldrlegal'])
    # print("\n TLDRLegal")

    # print("TextRank")
    # print_scores(data.textrank())
    #
    # print("KLSumm")
    # print_scores(data.greedy_kl())
    #
    # print("LEAD1")
    # print_scores(data.firstsent())
    #
    # print("Leadk")
    # print_scores(data.firstk())
    #
    # print("RandomK")
    # print_scores(data.randomk())

    # print("Readability")
    # data.readibility_score()

    # # data.log_odds_ratio()


    # print("Metrics")
    # data.get_metrics()
    #
    # data2 = Summaries(['tosdr_annotated'])
    # print("\n TOSDR Annotated")

    # print("TextRank")
    # print_scores(data.textrank())
    #
    # print("KLSumm")
    # print_scores(data.greedy_kl())
    #
    # print("LEAD1")
    # print_scores(data.firstsent())
    #
    # print("Leadk")
    # print_scores(data.firstk())
    #
    # print("RandomK")
    # print_scores(data.randomk())

    # print("Readability")
    # data.readibility_score()

    # data.log_odds_ratio()

    # print("Metrics")
    # data.get_metrics()


    # data = Summaries(['tldrlegal','tosdr_annotated'])
    # print("\n TLDRLegal and TOSDR")

    # print("TextRank")
    # print_scores(data.textrank())
    #
    # print("KLSumm")
    # print_scores(data.greedy_kl())
    #
    # print("LEAD1")
    # print_scores(data.firstsent())
    #
    # print("Leadk")
    # print_scores(data.firstk())

    # print("RandomK")
    # print_scores(data.randomk())

    # print("Readability")
    # data.readibility_score()

    # print("Metrics")
    # data.get_metrics()

    # data.log_odds_ratio()




    # data = Summaries(['DUC2002'])
    # print("\n DUC2002")

    # print("TextRank")
    # print_scores(data.textrank())
    #
    # print("KLSumm")
    # print_scores(data.greedy_kl())
    #
    # print("LEAD1")
    # print_scores(data.firstsent())
    #
    # print("Leadk")
    # print_scores(data.firstk())

    # print("RandomK")
    # print_scores(data.randomk())

    # print("Readability")
    # data.readibility_score()
    #
    # print("Metrics")
    # data.get_metrics()

