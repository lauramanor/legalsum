from summa.summarizer import summarize
from data import Summaries
import logging
from rouge import Rouge
from nltk.tokenize import sent_tokenize
import spacy
import re
import pandas as pd
import pytextrank


nlp = spacy.load('en_core_web_sm')
logging.basicConfig(level=logging.DEBUG)


def print_scores(scores):
    toprint = ""
    for key in scores.keys():
        toprint += str(round(scores[key]['f']*100,2)) + " & "

    print(toprint)

if __name__ == '__main__':
    data = Summaries(['tldrlegal','tosdr_annotated'])
    print_scores(data.textrank())
    print_scores(data.textrank(use_spacy=True))


    data = Summaries(['tosdr_annotated'])
    print_scores(data.textrank())
    print_scores(data.textrank(use_spacy=True))


    data = Summaries(['tldrlegal'])
    print_scores(data.textrank())
    print_scores(data.textrank(use_spacy=True))
