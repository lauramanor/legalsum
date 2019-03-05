from summa.summarizer import summarize
from data import Summaries
import logging
from rouge import Rouge
from nltk.tokenize import sent_tokenize
import spacy
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
        text = stuff['text'].replace('\n', '')
        ratio = .2


        #nltk
        tokenized = sent_tokenize(text)
        input = "\n".join(tokenized)

        spacy = True
        while ratio < 1:
            hyp = summarize(input)
            if len(hyp) > 0:
                hypes.append(hyp)
                refs.append(stuff['ref'])
                print(count)
                break
            else:

                ratio += .05
                if ratio > .8 and spacy:
                    # spacy
                    tokenized = list(nlp(text).sents)
                    input = "\n".join(sent.text for sent in tokenized)

                    spacy = False
                # print(count)

    rouge = Rouge()
    scores = rouge.get_scores(hypes, refs)
    print(len(scores))

    pass