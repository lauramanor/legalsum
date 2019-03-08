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

    hypes = []
    refs = []
    count = 0

    for thing in data.items:
        count += 1
        stuff = data.items[thing]

        text = stuff['text']

        for match in re.finditer("\.”", text):
            index = match.start()
            # print(match)

            # print(text)
            text = text[:index] + "”." + text[index+2:]
            # print(text)


        for match in re.finditer("[a-z]\.[A-Z]", text):
            index = match.start()
            # print(match)

            # print(text)
            text = text[:index+2] + " " + text[index+2:]
            # print(text)

        text = text.replace('&nbsp', ' ').replace('\n', ' ').lower()

        ratio = .2
        uid = stuff['uid']

        spacy = True

        #nltk
        tokenized = sent_tokenize(text)
        num_sents = len(tokenized)
        # if num_sents == 1:
        if spacy:
            spacy = False
            tokenized = list(nlp(text).sents)
            num_sents = len(tokenized)
            if num_sents != 1:
                input = "\n".join(sent.text for sent in tokenized)
                print("SPACY")
            else:
                hypes.append(hyp)
                refs.append(stuff['ref'])
                print(f'number: {count, len(hypes)} | {uid} | #sent: {num_sents} | ratio: {ratio} | BREAK')

                ratio = 1.1
        else:
            input = "\n ".join(tokenized)



        while ratio <= 1.0:

            hyp = summarize(input, ratio)
            if len(hyp) > 0:
                hypes.append(hyp)
                refs.append(stuff['ref'])
                print(f'number: {count, len(hypes)} | Success {uid} | #sent: {num_sents} | ratio: {ratio} | spacy: {spacy}')
                break
            else:

                ratio = round(ratio+.1,1)
                # if ratio > .8 and spacy is True:
                #     # spacy
                #     tokenized = list(nlp(text).sents)
                #     num_sents = len(tokenized)
                #     input = "\n".join(sent.text for sent in tokenized)
                #     ratio = .2
                #
                #     spacy = False
                # if int(ratio) >= 1.05:
                #     hypes.append(input)
                #     refs.append(stuff['ref'])
                #     print(f'number: {count, len(hypes)} | Fail {uid} | #sent: {num_sents} | ratio: {ratio} | ')


    print(len(hypes))
    rouge = Rouge()
    scores = rouge.get_scores(hypes, refs, avg=True)
    print(scores)

