import os.path
from os import walk
import json
import string
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
from gensim.summarization.summarizer import summarize as gensim_summarize
import random


# from ./readability-score-master import readability-score as rs
from matplotlib.pyplot import figure
figure(figsize=(5, 3.5))
plt.style.use('seaborn-colorblind')
# plt.rcParams.update(IPython_default)

logging.basicConfig(level=logging.INFO)
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
    text = text.lower()
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

    handeled = {'tosdr_annotated': True, 'tldrlegal': True, 'DUC2002': True}

    def __init__(self, loads=[]):
        self.mean_ntokens = []
        self.max_ntokens = int()
        self.mean_nsentences = int()

        self.items = []
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.files = {}

        for load in loads:
            if Summaries.handeled[load]:
                self.load(load)

        # unique_templates = []
        # for item in self.items:
        #     if item.ref.code:
        #         if item.ref.code == "1,s":
        #             unique_templates.append(item.ref.clean)
        #
        # print(f"templates: {len(set(unique_templates))}")


    def load(self, file):

        if file is "DUC2002":
            abstract_path = "data/DUC2002/data/test/summaries/extracts_abstracts"
            doc_sent_path = "data/DUC2002/data/test/docs.with.sentence.breaks"
            cleanr = re.compile('<.*?>')
            doc_sent_paths = []
            for (dirpath, dirnames, filenames) in walk(doc_sent_path):
                for f in filenames:
                    doc_sent_paths.append((os.path.relpath(os.path.join(dirpath, f), "."),f[:-1]))

            doc_sents = {}
            for doc_path, doc_name in doc_sent_paths:
                with open(doc_path, 'r') as f:
                    sents = []
                    read = False
                    for line in f:
                        # print(f"{doc_name} \t | {line}")
                        if not read and line.startswith( "<TEXT>"):
                            read = True
                        elif read:
                            line = re.sub(cleanr, '', line[:-1])
                            if len(line) > 1:
                                sents.append(line)
                doc_sents[doc_name[:-1]] = sents

            for (dirpath_a, dirnames_a, filenames_a) in walk(abstract_path):
                for docset in dirnames_a:
                    if docset is not "withdrawn":
                        perdoc_path = os.path.join(os.path.join(dirpath_a, docset), "perdocs")

                        try:
                            with open(perdoc_path, 'r') as f:
                                sents = []
                                read = False
                                for line in f:
                                    # print(f"{doc_name} \t | {line}")
                                    if line.startswith("DOCREF"):
                                        uid = docset + "_" + line[8:-2].strip()
                                        try:
                                            doc = doc_sents[line[8:-2].strip()]
                                        except KeyError as key_error:
                                            # print(perdoc_path, key_error.args)
                                            pass

                                        ref = []
                                    elif line.startswith("SUMMARIZER"):
                                        read = True
                                    elif line.endswith("</SUM>\n"):
                                        ref.append(line[:-7])
                                        read = False
                                        ref = " ".join(ref)
                                        self.items.append(SummarySet(file=file, uid=uid, doc_sents=doc, ref=ref))
                                    elif read and len(line) > 1:
                                        ref.append(line)
                        except FileNotFoundError as not_found:
                            # print(not_found.filename)
                            pass
        else:
            self.check_dir()
            reader = csv.DictReader(open(self.files[file]))
            for row in reader:
                set = SummarySet(row=row, file=file)
                if set.ref:
                    self.items.append(set)

        logging.info(f'Loaded {file}')

        self._update_mean_words()

    def _update_mean_words(self):
        ntokens = []
        nsents = []
        for item in self.items:
            if item.ref:
                ntokens.append(item.ref.ntokens)
                # nsents.append(item.ref.nsents)

        self.max_ntokens = max(ntokens)
        self.mean_ntokens = statistics.mean(ntokens)
        # self.mean_nsentences = statistics.mean(nsents)

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

                if item.quote.ntokens <= self.mean_ntokens or \
                        item.quote.nsents == 1:
                    hypes.append(item.quote.cleaned_text)
                    refs.append(item.ref.cleaned_text)

                    # print(item.quote.cleaned_text)
                else:
                    hyp = summarize("\n ".join(item.quote.clean_sents), words=self.mean_ntokens)
                    if len(hyp) > 0:
                        hypes.append(hyp)
                        refs.append(item.ref.cleaned_text)
                        score = rouge.get_scores(hyp, item.ref.cleaned_text)
                        print(f"{item.uid}\n \t {score}\n \t Hyp: {hyp} \n \t Ref: {item.ref}")

                    elif item.quote.ntokens/item.quote.nsents > self.mean_ntokens:
                        hyp = summarize("\n ".join(item.quote.clean_sents), words=self.max_ntokens)
                        if len(hyp) > 0:
                            hypes.append(hyp)
                            ref = item.ref.cleaned_text
                            refs.append(ref)
                            score = rouge.get_scores(hyp, item.ref.cleaned_text)
                            print(f"{item.uid}\n \t {score}\n \t Hyp: {hyp} \n \t Ref: {item.ref}")

                    else:
                        print(f"\n No summary for {item.uid} | {int(item.quote.ntokens/item.quote.nsents)}\t| {item.quote.cleaned_text}")

                            # break


        return rouge.get_scores(hypes, refs, avg=True)


    def gensim_textrank(self):
        hypes = []
        refs = []
        refs_dirty = []
        for item in self.items:
            if item.ref:

                if item.quote.ntokens <= self.mean_ntokens or \
                        item.quote.nsents == 1:
                    hypes.append(item.quote.cleaned_text)
                    refs.append(item.ref.cleaned_text)
                    refs_dirty.append(item.ref.original)
                    # print(item.quote.cleaned_text)
                else:
                    # for x in range(2, 11):
                    #     ratio = x * .1
                    hyp = gensim_summarize("\n ".join(item.quote.clean_sents), word_count=self.mean_ntokens)
                    if len(hyp) > 0:
                        hypes.append(hyp)
                        refs.append(item.ref.cleaned_text)
                        score = rouge.get_scores(hyp, item.ref.cleaned_text)
                        # print(f"{item.uid}\n \t {score}\n \t Hyp: {hyp} \n \t Ref: {item.ref}")

                    elif item.quote.ntokens/item.quote.nsents > self.mean_ntokens:
                        hyp = gensim_summarize("\n ".join(item.quote.clean_sents), word_count=self.max_ntokens)
                        if len(hyp) > 0:
                            hypes.append(hyp)
                            refs.append(item.ref.cleaned_text)
                    else:
                        print(f"No summary for {item.uid} | {int(item.quote.ntokens/item.quote.nsents)}\t| {item.quote.cleaned_text}")

                            # break

        hypes = [x.lower() for x in hypes]
        refs = [x.lower() for x in refs]

        return rouge.get_scores(hypes, refs, avg=True)


    def firstsent(self):
        hypes = []
        refs = []
        refs_dirty = []
        for item in self.items:
            if item.ref:
                hypes.append(item.quote.clean_sents[0])
                refs.append(item.ref.cleaned_text)
                # print(f"{item.uid} \n \t Hyp: {item.quote.clean_sents[0]} \n \t Ref: {item.ref.cleaned_text}")

        hypes = [x.lower() for x in hypes]
        refs = [x.lower() for x in refs]

        return rouge.get_scores(hypes, refs, avg=True)

    def firstk(self):
        hypes = []
        refs = []

        for item in self.items:
            if item.ref:
                refs.append(item.ref.cleaned_text)

                if item.quote.ntokens <= self.mean_ntokens:
                    hyp = item.quote.cleaned_text
                    hypes.append(hyp)
                    score = rouge.get_scores(hyp, item.ref.cleaned_text)
                    print(f"{item.uid}\n \t {score}\n \t Hyp: {hyp} \n \t Ref: {item.ref}")

                else:
                    hyp = self._firstk_item(item)
                    hypes.append(hyp)
                    score = rouge.get_scores(hyp, item.ref.cleaned_text)
                    print(f"{item.uid}\n \t {score}\n \t Hyp: {hyp} \n \t Ref: {item.ref}")

                    pass

        return rouge.get_scores(hypes, refs, avg=True)

    def _firstk_item(self, item):
        ntokens = item.quote.ntokens_by_sentence
        summary_ntokens = 0
        idx = 0
        while summary_ntokens < self.mean_ntokens:
            idx_ntokens = ntokens[idx]

            if summary_ntokens > 0 and \
                    abs(self.mean_ntokens - summary_ntokens - idx_ntokens) > abs(
                self.mean_ntokens - summary_ntokens):
                return " ".join(item.quote.clean_sents[:idx])

            summary_ntokens += idx_ntokens
            idx += 1

        return " ".join(item.quote.clean_sents[:idx])

    def randomk(self, n=10):
        hypes = []
        refs = []
        for _ in range(n):
            for item in self.items:
                if item.ref:
                    refs.append(item.ref.cleaned_text)

                    if item.quote.ntokens <= self.mean_ntokens:
                        hyp = item.quote.cleaned_text
                        hypes.append(hyp)
                        # score = rouge.get_scores(hyp, item.ref.cleaned_text)
                        # print(f"{item.uid}\n \t {score}\n \t Hyp: {hyp} \n \t Ref: {item.ref.cleaned_text}")

                    else:
                        hyp = self._randomk_item(item)
                        hypes.append(hyp)
                        # print(f"{item.uid}\n \t {score}\n \t Hyp: {hyp} \n \t Ref: {item.ref.clean}")

                        pass

        return rouge.get_scores(hypes, refs, avg=True)

    def _randomk_item(self, item):

        ntokens = item.quote.ntokens_by_sentence
        summary_ntokens = 0
        idxs = [x for x in range(item.quote.nsents)]
        hyp_idx = []

        while len(idxs) > 0 and summary_ntokens < self.mean_ntokens:
            idx = random.choice(idxs)
            idxs.remove(idx)
            idx_ntokens = ntokens[idx]

            if summary_ntokens > 0 and \
                    abs(self.mean_ntokens - summary_ntokens - idx_ntokens) > abs(
                self.mean_ntokens - summary_ntokens):
                return " ".join(item.quote.clean_sents[x] for x in hyp_idx)
            hyp_idx.append(idx)
            summary_ntokens += idx_ntokens


        return " ".join(item.quote.clean_sents[x] for x in hyp_idx)


    def get_metrics(self):

        # sentence and word counts

        quote_sentence_counts = []
        quote_word_counts = []

        ref_sentence_counts = []
        ref_word_counts = []

        unique_ngram_counts = np.empty((0, 4))
        ref_ngram_counts = np.empty((0, 4))

        unique_sentences_counts = []
        codes = Counter()

        for item in self.items:
            if item.ref:
                quote_sentence_counts.append(item.quote.nsents)
                quote_word_counts.append(item.quote.ntokens)

                ref_sentence_counts.append(item.ref.nsents)
                ref_word_counts.append(item.ref.ntokens)

                temp_unique = []
                temp_ref = []
                for key in item.quote.ngrams.keys():
                    temp_unique.append(num_unique(ref=item.ref.ngrams[key], quote=item.quote.ngrams[key]))
                    temp_ref.append(len(item.ref.ngrams[key]))
                if item.ref.code:
                    codes[item.ref.code] += 1

                unique_ngram_counts = np.vstack((unique_ngram_counts, temp_unique))
                ref_ngram_counts = np.vstack((ref_ngram_counts, temp_ref))

                unique_sentences_counts.append(num_unique(ref=item.ref.clean_sents, quote=item.quote.clean_sents))

                if item.ref.ntokens/item.quote.ntokens > 2:
                    print(f'\t {item.ref.ntokens} : {item.ref} \n {item.quote.ntokens} {item.quote} \n ')

        # micro, macro ratios

        print(f"Number of documents: {len(quote_sentence_counts)}")

        print(f"Reference nsentence information")
        print(pd.DataFrame(ref_sentence_counts).describe())

        print(f"original nsentence information")
        print(pd.DataFrame(quote_sentence_counts).describe())


        print(f"reference word count information")
        print(pd.DataFrame(ref_word_counts).describe())

        print(f"original text word count information")
        print(pd.DataFrame(quote_word_counts).describe())


        print(f"Codes information")
        print(codes)

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


        print(f"Words Ratio infomation")
        print(pd.DataFrame(words_ratio).describe())


        print("unique_ngrams_ratios_micro: ", unique_ngrams_ratios_micro)
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
        # fig2, ax2 = plt.subplots()
        # ax2.set_title('Sentence Counts')
        # ax2.boxplot()
        # ax2.xticks(2, ('Original Text','Reference'))
        # fig2.show()

        # plt.boxplot([quote_word_counts, ref_word_counts], showfliers=False)
        # plt.title('Word Counts')
        # plt.xticks(('Original Text', 'Reference'))
        # plt.show()
        # fig3, ax3 = plt.subplots()
        # ax3.set_title('Word Counts')
        # ax3.boxplot([quote_word_counts, ref_word_counts], showfliers=False)
        # ax3.set_xticklabels(['Original Text', 'Reference'])
        # fig3.set_size_inches(5, 3.5)
        #
        # fig3.show()

        print(f"original text word count information")
        print(pd.DataFrame(words_ratio).describe())


        words_ratio[words_ratio > 1.5] = 1.5
        # [ x for x in words_ratio if (x < 2) else 2 ]
        # fig4, ax4 = plt.subplots()
        # ax4.set_title('Word Ratios')
        # ax4.hist(words_ratio_alt)
        # fig4.show()



        n, bins, patches = plt.hist(x=words_ratio, bins=25,
                                     rwidth=0.9)
        plt.grid(axis='y')
        plt.xlabel('# words in reference summary / # words in original text')
        plt.ylabel('Frequency')
        plt.title('Compression Rates')
        plt.figure(figsize=(5, 3.5))

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

                ref_counts = sum(item.ref.get_lemmas(), ref_counts)
                quote_counts = sum(item.quote.get_lemmas(), quote_counts)


        remove = ['us', 'discogs', 'niantic', 'npm', 'airbnb', 'http', 'tpci']
        for w in remove:
            ref_counts[w] = 0
            quote_counts[w] = 0

        ref_total = sum(ref_counts.values())
        quote_total = sum(quote_counts.values())

        all_counts = ref_counts + quote_counts
        ave_count = statistics.mean(all_counts.values())
        print(f'average count: {ave_count}')

        log_probs = defaultdict(int)

        log_probs_list = []

        for word in all_counts:
            ref_count = ref_counts[word]
            quote_count = quote_counts[word]

            if ref_count + quote_count >= ave_count:
                ref_prob = ref_count / ref_total
                quote_prob = quote_count / quote_total

                if quote_prob == 0:
                    quote_prob = .00000000000000000000000000000001
                    # print(f'not in quote {word}')
                if ref_prob == 0:
                    ref_prob = .00000000000000000000000000000001
                    # print(f'not in ref {word}')

                ratio = ref_prob / quote_prob
                # print(word, ref_prob, quote_prob, ratio)

                log_probs[word] = math.log(ratio)

                log_probs_list.append((word, log_probs[word]))

        log_probs_list = sorted(log_probs_list, key=lambda x: x[1], reverse=True)
        top_ref = log_probs_list[:50]
        top_ref_str = ", ".join([x[0] for x in top_ref])

        print(f'Top 50 reference {top_ref_str}')
        score = []
        for word, ratio in top_ref:
            # print(word)
            score.append(readability_scores(word))

        scores = pd.DataFrame(score)
        print(scores.describe())

        log_probs_list = sorted(log_probs_list, key=lambda x: x[1])
        top_quote = log_probs_list[:50]
        top_quote_str = ", ".join([x[0] for x in top_quote])

        print(f'Top 50 quote {top_quote_str}')

        score = []
        for word, ratio in top_quote:
            # print(word)
            score.append(readability_scores(word))

        scores = pd.DataFrame(score)
        print(scores.describe())


        pass

    def readibility_score(self):
        ref_scores = []
        quote_scores = []
        for item in self.items:
            if item.ref: #ski
                ref_scores.append(readability_scores(item.ref.cleaned_text, sents=True))
                quote_scores.append(readability_scores(item.quote.cleaned_text, sents=True))
        print("\n Reference Readability Scores")
        print(pd.DataFrame(ref_scores).mean())
        print("\n Quote Readability Scores")
        print(pd.DataFrame(quote_scores).mean())

    def greedy_kl(self):
        """ Runs greedy KL on the documents"""

        #scipy.stats.entropy(pk, qk)
        hypes = []
        refs = []

        for item in self.items:
            if item.ref:
                refs.append(item.ref.cleaned_text)

                if item.quote.ntokens <= self.mean_ntokens:
                    hyp = item.quote.cleaned_text
                    hypes.append(hyp)
                    # if item.ref.ntokens / item.quote.ntokens < .05:
                    score = rouge.get_scores(hyp, item.ref.cleaned_text)
                    print(f"{item.uid}\n \t {score}\n \t quote: {item.quote} \n \t Hyp: {hyp} \n \t Ref: {item.ref}")

                else:
                    hyp = " ".join(item.quote.clean_sents[x] for x in self.get_kl_indexes(item=item))
                    hypes.append(hyp)
                    # if item.ref.ntokens / item.quote.ntokens < .05:

                    score = rouge.get_scores(hyp, item.ref.cleaned_text)

                    print(f"{item.uid}\n \t {score}\n \t quote: {item.quote} \n \t Hyp: {hyp} \n \t Ref: {item.ref}")

                    pass

        return rouge.get_scores(hypes, refs, avg=True)

    def get_kl_indexes(self, item):
        sentences = item.quote.get_lemmas()
        ntokens = item.quote._nlemmas_by_sentence
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

        if summary_ntokens is 0:
            print(f"No summary for {item.uid} | {int(item.quote.ntokens / item.quote.nsents)}\t| {item.quote.cleaned_text}")

        return summary_idx

    def to_json(self, filename=none):
        data = {}



class SummarySet:

    def __init__(self,  file, row=None,
                 uid=None, doc_sents = None, ref=None):
        if file == "DUC2002":
            self.uid = uid
            self.quote = Document(sents=doc_sents)
            self.ref = Document(text=ref)
        else:
            self.row = row
            self.uid = row['uid']
            self.quote = Document( text=row['text'])
            if file == "tldrlegal":
                self.ref = Document(text=row['summary'])
                self.id = row['id']
                self.title = row['title']
                self.name = row['name']
            elif file == "tosdr_annotated":
                self.note = row["note"]
                self.urls = row["urls"]
                self.quote_doc = row["quoteDoc"]

                #todo: make ref == summary, but also have the others in h ere
                if self.identify_code("1"):
                    key = self.identify_code("1")
                    self.ref = Document(text=self.row[key[:-5]], code=self.row[key])
                else:
                    self.ref = False
        if self.ref and self.ref.ntokens > self.quote.ntokens:
            self.ref = False
        # if self.ref and self.quote.ntokens < self.ref.ntokens:
        #     print(f"{self.uid} \n \t Quote: {self.quote} \n \t Ref: {self.ref}")
    def identify_code(self, x):
        for key in self.row.keys():
            if key.endswith('_code'):
                if x in self.row[key]:
                    # print(key[:-5])
                    return key #self.row[key[:-5]]

class Document:

    def __str__(self):
        return self.cleaned_text

    def __init__(self, text=None, sents=None, code=None):
        # self.parent = parent
        if text:
            self.original = text
            self.sents = sent_tokenize(self._clean_text())
            self.clean_sents = self._clean_sents()

        else:
            self.sents = sents
            self.clean_sents = self._clean_sents()
            self.original = " ".join(sents)

        self.cleaned_text = " ".join(self.clean_sents)
        self.nsents = len(self.clean_sents)

        self.ntokens_by_sentence = []
        self.tokens = []
        for sent in self.clean_sents:
            tokens = word_tokenize(sent)
            self.ntokens_by_sentence.append(len(tokens))
            self.tokens += tokens

        self.tokens_pos = nltk.pos_tag(self.tokens)
        self.ntokens = len(self.tokens)
        self.ngrams = self.get_ngrams()

        self._lemmas_by_sentence = None
        self._nlemmas_by_sentence = None

        self.code = code

        # self.readability = readability_scores(self.clean)

    def _clean_text(self):
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
        for match in re.finditer("[a-z]\n[A-Z]", text):
            indexes.append(match.start())
        for index in reversed(indexes):
            text = text[:index+1] + ". " + text[index + 2:]
            pass

        return text.replace('\n', ' ')

    def _clean_sents(self):
        stripped = []
        stop_words = stopwords.words('english')
        for sent in self.sents:
            sent = re.sub(r"[\W]+\s*", " ", sent)
            sent = sent.lower().strip()
            # sent = " ".join([x for x in sent if x not in stop_words])
            if len(sent) > 1:
                stripped.append(sent+".")
        return stripped

    def get_lemmas(self):
        """
        :return: list of list of lemmas
        """
        if not self._lemmas_by_sentence:
            lemmatize = WordNetLemmatizer()
            all_lemmas = []
            nwords_by_sents = []
            for sentence in self.clean_sents:
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
            self._nlemmas_by_sentence = nwords_by_sents
            self._lemmas_by_sentence = all_lemmas

        return self._lemmas_by_sentence


    def get_ngrams(self, tokens=None, num=4):
        """
        :return: dict of list of ngrams (1-num)
        """
        if not tokens:
            tokens = self.tokens
        tokens = [tok.lower() for tok in tokens]
        grams = {}
        for n in range(1, num + 1):
            grams[n - 1] = [gram for gram in ngrams(tokens, n)]

        return grams





if __name__ == '__main__':
    logging.debug(f'Invoking __main__.py for {__name__}')
