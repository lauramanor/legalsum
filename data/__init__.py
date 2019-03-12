import os.path
from os import walk
import json
import logging

import pandas as pd
import numpy as np
import math
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
import spacy
import re
from summa.summarizer import summarize
from rouge import Rouge
from collections import defaultdict
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
import statistics
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# from ./readability-score-master import readability-score as rs

plt.style.use('seaborn-colorblind')
# plt.rcParams.update(IPython_default)

logging.basicConfig(level=logging.DEBUG)
nlp = spacy.load('en_core_web_sm')
rouge = Rouge()


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def safe_div_array(a, b):
    if 0 in b:
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    else:
        return np.divide(a, b)


def safe_div(a, b):
    if b == 0:
        return 1
    else:
        return a / b


def num_unique(ref, quote):
    unique_count = 0
    for x in ref:
        if x not in quote:
            unique_count += 1
    return unique_count


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

        return rouge.get_scores(hypes, refs, avg=True)

    def firstsent(self, use_spacy=False):
        hypes = []
        refs = []
        refs_dirty = []
        for item in self.items:
            if item.ref:
                hypes.append(item.quote.sents(use_spacy)[0])
                refs.append(item.ref.clean)

        return rouge.get_scores(hypes, refs, avg=True)

    def get_metrics(self):

        # sentence and word counts

        quote_sentence_counts_nltk = []
        quote_sentence_counts_spacy = []
        quote_word_counts = []
        quote_ngrams = []
        # quote_1grams = []
        # quote_2grams = []
        # quote_3grams = []
        # quote_4grams = []

        ref_sentence_counts_nltk = []
        ref_sentence_counts_spacy = []
        ref_word_counts = []
        ref_ngrams = []

        unique_ngram_counts = np.empty((0, 4))
        ref_ngram_counts = np.empty((0, 4))

        unique_sentences_counts_nltk = []
        unique_sentences_counts_spacy = []

        # ref_1grams = []
        # ref_2grams = []
        # ref_3grams = []
        # ref_4grams = []
        for item in self.items:
            if item.ref:
                quote_sentence_counts_nltk.append(item.quote.nsents())
                quote_sentence_counts_spacy.append(item.quote.nsents(use_spacy=True))
                quote_word_counts.append(item.quote.ntokens)
                # quote_ngrams.append(item.quote.ngrams)

                ref_sentence_counts_nltk.append(item.ref.nsents())
                ref_sentence_counts_spacy.append(item.ref.nsents(use_spacy=True))
                ref_word_counts.append(item.ref.ntokens)
                # ref_ngrams.append(item.ref.ngrams)

                temp_unique = []
                temp_ref = []
                for key in item.quote.ngrams.keys():
                    temp_unique.append(num_unique(ref=item.ref.ngrams[key], quote=item.quote.ngrams[key]))
                    temp_ref.append(len(item.ref.ngrams[key]))

                unique_ngram_counts = np.vstack((unique_ngram_counts, temp_unique))
                ref_ngram_counts = np.vstack((ref_ngram_counts, temp_ref))

                unique_sentences_counts_nltk.append(num_unique(ref=item.ref.sents(), quote=item.quote.sents()))
                unique_sentences_counts_spacy.append(
                    num_unique(ref=item.ref.sents(use_spacy=True), quote=item.quote.sents(use_spacy=True)))

        # TODO: print histogram average, etc

        # micro, macro ratios

        sents_ratio_micro_nltk = sum(ref_sentence_counts_nltk) / sum(quote_sentence_counts_nltk)
        sents_ratio_micro_spacy = sum(ref_sentence_counts_spacy) / sum(quote_sentence_counts_nltk)
        words_ratio_micro = sum(ref_word_counts) / sum(quote_word_counts)
        unique_ngrams_ratios_micro = sum(unique_ngram_counts) / sum(ref_ngram_counts)
        unique_ratio_array_test = safe_div_array(unique_ngram_counts, ref_ngram_counts)
        unique_ngrams_ratios_macro = sum(safe_div_array(unique_ngram_counts, ref_ngram_counts)) / \
                                     np.shape(unique_ngram_counts)[0]

        unique_sentence_ratio_micro_nltk = sum(unique_sentences_counts_nltk) / sum(quote_sentence_counts_nltk)
        unique_sentence_ratio_micro_spacy = sum(unique_sentences_counts_spacy) / sum(quote_sentence_counts_spacy)

        words_ratio = safe_div_array(ref_word_counts, quote_word_counts)
        sents_ratio_nltk = safe_div_array(ref_sentence_counts_nltk, quote_sentence_counts_nltk)
        sents_ratio_spacy = safe_div_array(ref_sentence_counts_spacy, quote_sentence_counts_spacy)
        unique_sents_ratio_nltk = safe_div_array(unique_sentences_counts_nltk, ref_sentence_counts_nltk)
        unique_sents_ratio_spacy = safe_div_array(unique_sentences_counts_spacy, ref_sentence_counts_spacy)

        words_ratio_macro = sum(words_ratio) / len(words_ratio)
        sents_ratio_macro_nltk = sum(sents_ratio_nltk) / len(sents_ratio_nltk)
        sents_ratio_macro_spacy = sum(sents_ratio_spacy) / len(sents_ratio_spacy)
        unique_sents_ratio_macro_nltk = sum(unique_sents_ratio_nltk) / len(unique_sents_ratio_nltk)
        unique_sents_ratio_fig1macro_spacy = sum(unique_sents_ratio_spacy) / len(unique_sents_ratio_spacy)

        sents_ratio_macro_spacy = sum(sents_ratio_spacy)

        fig1, ax1 = plt.subplots()
        ax1.set_title('unique n-grams')
        ax1.bar(['1gram', '2gram', '3gram', '4gram'], unique_ngrams_ratios_micro)
        fig1.show()

        fig2, ax2 = plt.subplots()

        ax2.boxplot([quote_sentence_counts_nltk, ref_sentence_counts_nltk], showfliers=False)
        ax2.set_title('Sentence Counts')
        ax2.set_xticklabels(['Original Text', 'Reference'])
        fig2.show()
        # fig2, ax2 = plt.subplots()
        # ax2.set_title('Sentence Counts')
        # ax2.boxplot()
        # ax2.xticks(2, ('Original Text','Reference'))
        # fig2.show()

        # plt.boxplot([quote_word_counts, ref_word_counts], showfliers=False)
        # plt.title('Word Counts')
        # plt.xticks(('Original Text', 'Reference'))
        # plt.show()
        fig3, ax3 = plt.subplots()
        ax3.set_title('Word Counts')
        ax3.boxplot([quote_word_counts, ref_word_counts], showfliers=False)
        ax3.set_xticklabels(['Original Text', 'Reference'])

        fig3.show()

        pass

    def log_odds_ratio(self):
        """

        :return: dict of
        """
        ref_counts = Counter()
        quote_counts = Counter()

        lemmatize = WordNetLemmatizer()


        for item in self.items:
            if item.ref: #skip if no reference summary
                for word, pos in item.ref.tokens_pos:
                    if word.isalpha():
                        if get_wordnet_pos(pos):
                            lemma = lemmatize.lemmatize(word, get_wordnet_pos(pos))
                        else:
                            lemma = lemmatize.lemmatize(word)
                        ref_counts[lemma] += 1
                for word, pos in item.quote.tokens_pos:
                    if word.isalpha():
                        if get_wordnet_pos(pos):
                            lemma = lemmatize.lemmatize(word, get_wordnet_pos(pos))
                        else:
                            lemma = lemmatize.lemmatize(word)
                        if lemma == 'cooky':
                            lemma == 'cookie'
                        if lemma == 'u':
                            lemma = 'us'
                        if lemma == 'b':
                            print(f'{word}')
                        quote_counts[lemma] += 1

        ref_total = sum(ref_counts.values())
        quote_total = sum(quote_counts.values())

        all_counts = ref_counts + quote_counts
        ave_count = statistics.mean(all_counts.values())
        print(f'average count: {ave_count}')

        log_probs = defaultdict(int)

        log_probs_list = []
        # stops = list(stopwords.words('english')).append('us')

        for word in all_counts:
            # if word not in stopwords.words('english'):
            #     if word != 'us':
            ref_count = ref_counts[word]
            quote_count = quote_counts[word]

            if ref_count + quote_count >= ave_count:
                ref_prob = ref_count / ref_total
                quote_prob = quote_count / quote_total

                if quote_prob == 0:
                    quote_prob = .00000000000000000000000000000001
                    print(f'not in quote {word}')
                if ref_prob == 0:
                    ref_prob = .00000000000000000000000000000001
                    print(f'not in ref {word}')

                ratio = ref_prob / quote_prob
                # print(word, ref_prob, quote_prob, ratio)

                log_probs[word] = math.log(ratio)

                log_probs_list.append((word, log_probs[word]))

        log_probs_list = sorted(log_probs_list, key=lambda x: x[1], reverse=True)
        top_ref = log_probs_list[:50]

        print(f'Top 25 reference {top_ref[:][0]}')


        log_probs_list = sorted(log_probs_list, key=lambda x: x[1])
        top_quote = log_probs_list[:50]

        print(f'Top 25 quote {top_quote}')

        pass


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

        self.tokens = word_tokenize(self.clean)
        self.tokens_pos = nltk.pos_tag(self.tokens)
        self.ntokens = len(self.tokens)
        self.ngrams = self.get_ngrams()

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

    def get_ngrams(self, tokens=None, num=4):
        """
        :return: dict of list of ngrams (1-num)
        """
        if not tokens:
            tokens = self.tokens
        grams = {}
        for n in range(1, num + 1):
            grams[n - 1] = [gram for gram in ngrams(tokens, n)]

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
