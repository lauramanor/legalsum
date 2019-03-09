import os.path
from os import walk
import json
import logging

import pandas as pd
import csv

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
import spacy
import re
from summa.summarizer import summarize
from rouge import Rouge

logging.basicConfig(level=logging.DEBUG)
nlp = spacy.load('en_core_web_sm')
rouge = Rouge()


class Summaries():
    # tosdr_annotated code guide
    annotations = ["1", "2", "3", "s", "j", "d", "o", "q"]

    # handeled = {'tosdr_annotated':True, 'tldrlegal':True}

    def __init__(self, loads=[]):
        self.items = []
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.files = {}

        for load in loads:
            self.load(load)

    def load(self, file):
        self.check_dir()
        reader = csv.DictReader(open(self.files[file]))
        for row in reader:
            self.items.append(Summary(row, file=file))

        logging.info(f'Loaded {file}')

    def check_dir(self, check=None):
        if not check:
            check = self.dir_path

        for (dirpath, dirnames, filenames) in walk(check):
            for file in filenames:
                if file.endswith(".csv"):
                    self.files[file[:-4]] = os.path.join(dirpath, file)
        logging.debug(f'Checking {check}')
        logging.debug(', '.join(self.files))

    def textrank(self, use_spacy=False):
        hypes = []
        refs = []
        refs_dirty = []
        for item in self.items:
            if item.ref:

                if item.quote.nsents(use_spacy) <= 1:
                    hypes.append(item.quote.clean)
                    refs.append(item.ref.clean)
                    refs_dirty.append(item.ref.original)
                    # print(item.quote.clean)
                else:
                    for x in range(2, 11):
                        ratio = x * .1
                        hyp = summarize("\n ".join(item.quote.sents(use_spacy)), ratio)
                        if len(hyp) > 0:
                            hypes.append(hyp)
                            refs.append(item.ref.clean)
                            refs_dirty.append(item.ref.original)

                            break

        # print(len(hypes), use_spacy)
        #
        # print(rouge.get_scores(hypes, refs_dirty, avg=True))

        return rouge.get_scores(hypes, refs, avg=True)

class Summary:

    def __init__(self, row, file):
        self.row = row
        self.uid = row['uid']
        self.quote = Document(row['text'])
        if file == "tldrlegal":
            self.ref = Document(row['summary'])
            self.id = row['id']
            self.title = row['title']
            self.name = row['name']
        elif file == "tosdr_annotated":
            self.note = row["note"]
            self.urls = row["urls"]
            self.quote_doc = row["quoteDoc"]
            if self.identify_code("1"):
                self.ref = Document(self.identify_code("1"))
            else:
                self.ref = False

    def identify_code(self, x):
        for key in self.row.keys():
            if key.endswith('_code'):
                if x in self.row[key]:
                    # print(key[:-5])
                    return self.row[key[:-5]]

class Document:

    def __str__(self):
        return self.clean

    def __init__(self, text, code=None):
        self.original = text
        self.clean = self.clean_text()

        self.sents_nltk = self.get_sentences()
        self.nsents_nltk = len(self.sents_nltk)

        self.sents_spacy = self.get_sentences(use_spacy=True)
        self.nsents_spacy = len(self.sents_spacy)

        self.ngrams = self.get_ngrams()
        self.nwords = word_tokenize(self.clean)

        self.code = code

    def clean_text(self):
        text = self.original
        # print(text)
        for match in re.finditer("\.”", text):
            index = match.start()
            text = text[:index] + "”." + text[index + 2:]

        for match in re.finditer("[a-z]\.[A-Z]", text):
            index = match.start()
            text = text[:index + 2] + " " + text[index + 2:]

        return text.replace('&nbsp', ' ').replace('\n', ' ').lower()

    def get_sentences(self, text=None, use_spacy=False):

        if not text:
            text = self.clean

        if use_spacy:
            tokenized = list(nlp(text).sents)
            return [sent.text for sent in tokenized]
        else:
            return sent_tokenize(text)

    def get_ngrams(self, text=None, num=4):
        """
        :return: dict of list of ngrams (1-num)
        """
        if not text:
            text = self.clean
        grams = {}
        for n in range(1, num + 1):
            grams[n] = ngrams(text, n)

        return grams

    def nsents(self, use_spacy=False):
        if use_spacy:
            return self.nsents_spacy
        else:
            return self.nsents_nltk

    def sents(self, use_spacy=False):
        if use_spacy:
            return self.sents_spacy
        else:
            return self.sents_nltk


if __name__ == '__main__':
    logging.debug(f'Invoking __init__.py for {__name__}')
