import os.path
from os import walk
import json
import logging
from scipy.stats import entropy as kl
import pandas as pd
import numpy as np
import math
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
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
from readability_score.calculators.fleschkincaid import *
from readability_score.calculators.colemanliau import *
from readability_score.calculators.dalechall import *
from readability_score.calculators.smog import *
from readability_score.calculators.ari import *
from readability_score.calculators.linsearwrite import *
from readability_score.calculators.flesch import *
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

# from ./readability-score-master import readability-score as rs

plt.style.use('seaborn-colorblind')
# plt.rcParams.update(IPython_default)

logging.basicConfig(level=logging.DEBUG)
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


def readability_scores(text, sents=False):
    """
    https://github.com/wimmuskee/readability-score

     - Flesch-Kincaid fleschkincaid.FleschKincaid()
     - Coleman-Liau colemanliau.ColemanLiau()
     - Dale-Chall dalechall.DaleChall()
     - SMOG smog.SMOG()
     - Automated Readability Index ari.ARI()
     - LinsearWrite linsearwrite.LinsearWrite()

    :return:
    """
    scores = {}
    scores["fk"] = FleschKincaid(text, locale='en_US').min_age
    # scores["dc"] = DaleChall(text, locale='en_US').min_age
    scores["smog"] = SMOG(text, locale='en_US').min_age
    scores["ari"] = ARI(text, locale='en_US').min_age
    # scores["lw"] = LinsearWrite(text, locale='en_US').min_age
    # scores["fl"] = Flesch(text, locale='en_US').min_age # no min age!
    if sents:
        scores["cl"] = ColemanLiau(text, locale='en_US').min_age
        scores["ave"] = int(sum(scores.values()) / 4)
    else:
        scores["ave"] = int(sum(scores.values()) / 3)

    return scores

class Summaries():
    # tosdr_annotated code guide
    annotations = ["1", "2", "3", "s", "j", "d", "o", "q"]

    # handeled = {'tosdr_annotated':True, 'tldrlegal':True}

    def __init__(self, loads=[]):
        self.mean_ntokens = []

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

        self._update_mean_words()

    def _update_mean_words(self):
        ntokens = []
        for item in self.items:
            if item.ref:
                ntokens.append(item.ref.ntokens)

        self.mean_ntokens = statistics.mean(ntokens)

    def check_dir(self, check=None):
        if not check:
            check = self.dir_path

        for (dirpath, dirnames, filenames) in walk(check):
            for file in filenames:
                if file.endswith(".csv"):
                    self.files[file[:-4]] = os.path.join(dirpath, file)
        logging.debug(f'Checking {check}')
        logging.debug(', '.join(self.files))

    def textrank(self):
        hypes = []
        refs = []
        refs_dirty = []
        for item in self.items:
            if item.ref:

                if item.quote.nsents <= 1:
                    hypes.append(item.quote.clean)
                    refs.append(item.ref.clean)
                    refs_dirty.append(item.ref.original)
                    # print(item.quote.clean)
                else:
                    for x in range(2, 11):
                        ratio = x * .1
                        hyp = summarize("\n ".join(item.quote.sents), ratio)
                        if len(hyp) > 0:
                            hypes.append(hyp)
                            refs.append(item.ref.clean)
                            refs_dirty.append(item.ref.original)

                            break

        return rouge.get_scores(hypes, refs, avg=True)

    def firstsent(self):
        hypes = []
        refs = []
        refs_dirty = []
        for item in self.items:
            if item.ref:
                hypes.append(item.quote.sents[0])
                refs.append(item.ref.clean)

        return rouge.get_scores(hypes, refs, avg=True)

    def get_metrics(self):

        # sentence and word counts

        quote_sentence_counts = []
        quote_word_counts = []

        ref_sentence_counts = []
        ref_word_counts = []

        unique_ngram_counts = np.empty((0, 4))
        ref_ngram_counts = np.empty((0, 4))

        unique_sentences_counts = []

        for item in self.items:
            if item.ref:
                quote_sentence_counts.append(item.quote.nsents())
                quote_word_counts.append(item.quote.ntokens)

                ref_sentence_counts.append(item.ref.nsents())
                ref_word_counts.append(item.ref.ntokens)

                temp_unique = []
                temp_ref = []
                for key in item.quote.ngrams.keys():
                    temp_unique.append(num_unique(ref=item.ref.ngrams[key], quote=item.quote.ngrams[key]))
                    temp_ref.append(len(item.ref.ngrams[key]))

                unique_ngram_counts = np.vstack((unique_ngram_counts, temp_unique))
                ref_ngram_counts = np.vstack((ref_ngram_counts, temp_ref))

                unique_sentences_counts.append(num_unique(ref=item.ref.sents(), quote=item.quote.sents()))

                if item.ref.ntokens/item.quote.ntokens > 2:
                    print(f'\t {item.ref.ntokens} : {item.ref} \n {item.quote.ntokens} {item.quote} \n ')

        # micro, macro ratios

        print(pd.DataFrame(ref_word_counts).describe())

        sents_ratio_micro = sum(ref_sentence_counts) / sum(quote_sentence_counts)
        words_ratio_micro = sum(ref_word_counts) / sum(quote_word_counts)
        unique_ngrams_ratios_micro = sum(unique_ngram_counts) / sum(ref_ngram_counts)
        unique_ratio_array_test = safe_div_array(unique_ngram_counts, ref_ngram_counts)
        unique_ngrams_ratios_macro = sum(safe_div_array(unique_ngram_counts, ref_ngram_counts)) / \
                                     np.shape(unique_ngram_counts)[0]

        unique_sentence_ratio_micro = sum(unique_sentences_counts) / sum(quote_sentence_counts)

        words_ratio = np.divide(ref_word_counts, quote_word_counts)
        sents_ratio = safe_div_array(ref_sentence_counts, quote_sentence_counts)
        unique_sents_ratio = safe_div_array(unique_sentences_counts, ref_sentence_counts)

        words_ratio_macro = sum(words_ratio) / len(words_ratio)
        sents_ratio_macro = sum(sents_ratio) / len(sents_ratio)
        unique_sents_ratio_macro = sum(unique_sents_ratio) / len(unique_sents_ratio)

        #
        # fig1, ax1 = plt.subplots()
        # ax1.set_title('unique n-grams')
        # ax1.bar(['1gram', '2gram', '3gram', '4gram'], unique_ngrams_ratios_micro)
        # fig1.show()
        #
        # fig2, ax2 = plt.subplots()
        #
        # ax2.boxplot([quote_sentence_counts, ref_sentence_counts], showfliers=False)
        # ax2.set_title('Sentence Counts')
        # ax2.set_xticklabels(['Original Text', 'Reference'])
        # fig2.show()
        # # fig2, ax2 = plt.subplots()
        # # ax2.set_title('Sentence Counts')
        # # ax2.boxplot()
        # # ax2.xticks(2, ('Original Text','Reference'))
        # # fig2.show()
        #
        # # plt.boxplot([quote_word_counts, ref_word_counts], showfliers=False)
        # # plt.title('Word Counts')
        # # plt.xticks(('Original Text', 'Reference'))
        # # plt.show()
        # fig3, ax3 = plt.subplots()
        # ax3.set_title('Word Counts')
        # ax3.boxplot([quote_word_counts, ref_word_counts], showfliers=False)
        # ax3.set_xticklabels(['Original Text', 'Reference'])
        #
        # fig3.show()


        words_ratio[words_ratio > 1.5] = 1.5
        # [ x for x in words_ratio if (x < 2) else 2 ]
        # fig4, ax4 = plt.subplots()
        # ax4.set_title('Word Ratios')
        # ax4.hist(words_ratio_alt)
        # fig4.show()

        n, bins, patches = plt.hist(x=words_ratio, bins='auto',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Ratio')
        plt.ylabel('Frequency')
        plt.title('Word Ratios')
        # plt.text(23, 45, r'$\mu=15, b=3$')
        # maxfreq = n.max()
        plt.show()
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
            if word not in stopwords.words('english'):
                if word != 'us':
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

        print(f'Top 50 reference {top_ref}')
        score = []
        for word, probs in top_ref:
            score.append(readability_scores(word))

        scores = pd.DataFrame(score)
        print(scores.describe())

        log_probs_list = sorted(log_probs_list, key=lambda x: x[1])
        top_quote = log_probs_list[:50]

        print(f'Top 50 quote {top_quote}')

        score = []
        for word, probs in top_quote:
            score.append(readability_scores(word))

        scores = pd.DataFrame(score)
        print(scores.describe())


        pass

    def readibility_score(self):
        ref_scores = []
        quote_scores = []
        for item in self.items:
            if item.ref: #ski
                ref_scores.append(readability_scores(item.ref.clean, sents=True))
                quote_scores.append(readability_scores(item.quote.clean, sents=True))
        print("\n Reference Readability Scores")
        print(pd.DataFrame(ref_scores).mean())
        print("\n Quote Readability Scores")
        print(pd.DataFrame(quote_scores).mean())

    def greedy_kl(self):
        """ Runs greedy KL on the documents"""

        #scipy.stats.entropy(pk, qk)
        hypes = []
        refs = []
        refs_dirty = []
        for item in self.items:
            if item.ref:
                refs.append(item.ref.clean.lower())
                #average len of words is 17.3 --
                if item.quote.ntokens <= self.mean_ntokens:
                    hypes.append(item.quote.clean.lower())
                else:
                    hypes.append(" ".join(item.quote.sents[x].lower() for x in self.get_kl_indexes(item=item)))
                    pass

        return rouge.get_scores(hypes, refs, avg=True)

    def get_kl_indexes(self, item):
        sentences = item.quote.get_lemmas()
        ntokens = item.quote._ntokens_by_sentence
        full_text = sum(sentences, Counter())
        full_text_nwords = sum(full_text.values())
        full_text_dist = [(full_text[key] + 1) / (full_text_nwords + len(full_text.keys())) for key in full_text.keys()]

        summary_idx = []
        summary_ntokens = 0
        summary_counts = Counter()

        while summary_ntokens < self.mean_ntokens:
            current_scores = np.ones((len(sentences)))
            for idx, lemmas in enumerate(sentences):
                if idx not in summary_idx:
                    if summary_ntokens > 0:
                        lemmas = summary_counts + lemmas
                    dist = [(lemmas[key] + 1) / (sum(lemmas.values()) + len(full_text.keys())) for key in
                            full_text.keys()]
                    current_scores[idx] = kl(full_text_dist, dist)

            idx_max = np.argmin(current_scores)
            idx_ntokens = ntokens[idx_max]
            if summary_ntokens > 0 and \
                    abs(self.mean_ntokens - summary_ntokens - idx_ntokens) > abs(self.mean_ntokens - summary_ntokens):
                return summary_idx
            summary_idx.append(idx_max)
            summary_counts += sentences[idx_max]
            summary_ntokens += idx_ntokens
            # print(summary_counts)

        return summary_idx



class Summary:

    def __init__(self, row, file):
        self.row = row
        self.uid = row['uid']
        self.quote = Document(parent=self, text=row['text'])
        if file == "tldrlegal":
            self.ref = Document(parent=self, text=row['summary'])
            self.id = row['id']
            self.title = row['title']
            self.name = row['name']
        elif file == "tosdr_annotated":
            self.note = row["note"]
            self.urls = row["urls"]
            self.quote_doc = row["quoteDoc"]
            if self.identify_code("1"):
                key = self.identify_code("1")
                self.ref = Document(parent=self, text=self.row[key[:-5]], code=self.row[key])
            else:
                self.ref = False

    def identify_code(self, x):
        for key in self.row.keys():
            if key.endswith('_code'):
                if x in self.row[key]:
                    # print(key[:-5])
                    return key #self.row[key[:-5]]

class Document:

    def __str__(self):
        return self.clean

    def __init__(self, parent, text, code=None):
        # self.parent = parent
        self.original = text
        self.clean = self.clean_text()

        self.sents = self.get_sentences()
        self.nsents = len(self.sents)

        self.tokens = word_tokenize(self.clean)
        self.tokens_pos = nltk.pos_tag(self.tokens)
        self.ntokens = len(self.tokens)
        self.ngrams = self.get_ngrams()

        self._lemmas_by_sentence = None
        self._ntokens_by_sentence = None

        self.code = code

        # self.readability = readability_scores(self.clean)

    def clean_text(self):
        """
        :return: remove spaces, remove n,
        edit some periods, and spacing around the periods
        """
        text = self.original.replace('&nbsp', ' ')

        https = "https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        indexes = []
        text = re.sub(https, "", text)

        # print(text)
        for match in re.finditer("\.”", text):
            index = match.start()
            text = text[:index] + "”." + text[index + 2:]

        indexes = []
        for match in re.finditer("[a-z][\.][A-Z]", text):
            indexes.append(match.start())
        for index in reversed(indexes):
            text = text[:index + 2] + " " + text[index + 2:]
            pass

        indexes = []
        for match in re.finditer("\S[,:;][a-z]", text):
            indexes.append(match.start())
        for index in reversed(indexes):
            text = text[:index + 2] + " " + text[index + 2:]
            pass

        indexes = []
        for match in re.finditer("-\n[a-z]", text):
            indexes.append(match.start())
        for index in reversed(indexes):
            text = text[:index] + text[index + 2:]
            #TODO: does this work?
            pass

        indexes = []
        for match in re.finditer("[a-z]\n[A-Z]", text):
            indexes.append(match.start())
        for index in reversed(indexes):
            text = text[:index+1] + ". " + text[index + 2:]
            #TODO: does this work?
            pass

        return text.replace('\n', ' ')



    def get_lemmas(self):
        """
        :return: list of list of lemmas
        """
        if not self._lemmas_by_sentence:
            lemmatize = WordNetLemmatizer()
            all_lemmas = []
            nwords_by_sents = []
            for sentence in self.sents:
                lemmas = Counter()
                tokens = word_tokenize(sentence)
                nwords_by_sents.append(len(tokens))
                for word, pos in nltk.pos_tag(tokens):
                    if word.isalpha():
                        if get_wordnet_pos(pos):
                            lemma = lemmatize.lemmatize(word.lower(), get_wordnet_pos(pos))
                        else:
                            lemma = lemmatize.lemmatize(word.lower())
                        if lemma not in stopwords.words('english'):
                            if lemma is "cooky" and word is "cookies":
                                    lemmas["cookies"] += 1
                            elif lemma is not "u":
                                lemmas[lemma] += 1
                all_lemmas.append(lemmas)
            self._ntokens_by_sentence = nwords_by_sents
            self._lemmas_by_sentence = all_lemmas

        return self._lemmas_by_sentence

    def get_sentences(self, text=None):

        if not text:
            text = self.clean

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

    def nsents(self):

        return self.nsents

    def sents(self):

        return self.sents




if __name__ == '__main__':
    logging.debug(f'Invoking __main__.py for {__name__}')
