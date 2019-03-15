from summa.summarizer import summarize
from data import Summaries
import logging
from rouge import Rouge
from nltk.tokenize import sent_tokenize
import spacy
import re
import pandas as pd
import pytextrank


# nlp = spacy.load('en_core_web_sm')
logging.basicConfig(level=logging.INFO)


def print_scores(scores):
    toprint = ""
    for key in scores.keys():
        toprint += " & " + str(round(scores[key]['f']*100,2))

    print(toprint)

if __name__ == '__main__':


    data = Summaries(['tldrlegal'])
    print("\n TLDRLegal")

    print("TextRank")
    print_scores(data.textrank())

    print("LEAD1")
    print_scores(data.firstsent())

    print("KLSumm")
    print_scores(data.greedy_kl())


    print("Readability")
    data.readibility_score()


    # data.log_odds_ratio()

    data = Summaries(['tosdr_annotated'])
    print("\n TOSDR Annotated")

    print("TextRank")
    print_scores(data.textrank())

    print("LEAD1")
    print_scores(data.firstsent())

    print("KLSumm")
    print_scores(data.greedy_kl())

    print("Readability")
    data.readibility_score()

    # data.log_odds_ratio()

    data = Summaries(['tldrlegal','tosdr_annotated'])
    print("\n TLDRLegal and TOSDR")

    print("TextRank")
    print_scores(data.textrank())

    print("LEAD1")
    print_scores(data.firstsent())

    print("KLSumm")
    print_scores(data.greedy_kl())

    print("Readability")
    data.readibility_score()

    # data.log_odds_ratio()