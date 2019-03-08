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


if __name__ == '__main__':
    # data = Summaries(['tldrlegal','tosdr_annotated'])
    data = Summaries(['tldrlegal'])

    # hypes = []
    # refs = []
    # count = 0
    # 
    # for thing in data.items:
    #     count += 1
    #     stuff = data.items[thing]
    # 
    #     text = stuff['text']
    # 
    # 
    # 
    #     ratio = .2
    #     uid = stuff['uid']
    # 
    #     spacy = True
    # 
    #     #nltk
    #     tokenized = sent_tokenize(text)
    #     num_sents = len(tokenized)
    #     # if num_sents == 1:
    #     if spacy:
    #         spacy = False
    #         tokenized = list(nlp(text).sents)
    #         num_sents = len(tokenized)
    #         if num_sents != 1:
    #             input = "\n".join(sent.text for sent in tokenized)
    #             print("SPACY")
    #         else:
    #             hypes.append(hyp)
    #             refs.append(stuff['ref'])
    #             print(f'number: {count, len(hypes)} | {uid} | #sent: {num_sents} | ratio: {ratio} | BREAK')
    # 
    #             ratio = 1.1
    #     else:
    #         input = "\n ".join(tokenized)
    # 
    # 
    # 
    #     while ratio <= 1.0:
    # 
    #         hyp = summarize(input, ratio)
    #         if len(hyp) > 0:
    #             hypes.append(hyp)
    #             refs.append(stuff['ref'])
    #             print(f'number: {count, len(hypes)} | Success {uid} | #sent: {num_sents} | ratio: {ratio} | spacy: {spacy}')
    #             break
    #         else:
    # 
    #             ratio = round(ratio+.1,1)
    # 
    # 
    # print(len(hypes))
    # rouge = Rouge()
    # scores = rouge.get_scores(hypes, refs, avg=True)
    print(data.textrank())
    print(data.textrank(spacy=True))