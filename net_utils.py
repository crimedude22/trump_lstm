from __future__ import print_function
import nltk
import codecs
import re
import os
from time import strftime,gmtime
import pandas
import ggplot

import numpy as np
import random
import sys

def prep_strings(speech_path,maxlen,step,contextual=False,futurelen=None,remove_ones=False):
    speech_file = codecs.open(speech_path,'r','ascii',errors='ignore')

    # Get as lines first to shuffle
    speech_raw = [line for line in speech_file]
    random.shuffle(speech_raw)

    # Join lines, tokenize & join, then eliminate whitespace before punctuation
    speech_raw = ' '.join(speech_raw)
    speech_raw = nltk.word_tokenize(speech_raw.lower())

    # Remove squirrelly 'number words'
    speech_raw = [x for x in speech_raw if not any(c.isdigit() for c in x)]
    speech_raw = [x for x in speech_raw if not any(c == "'" for c in x)]
    speech_raw = [x for x in speech_raw if not x=="."]
    speech_raw = [x for x in speech_raw if not x==","]

    # Remove words that only appear once in the shittiest way imaginable
    if remove_ones:
        print("removing rare tokens")

        fdist = nltk.FreqDist(speech_raw)

        words = []
        vals = []
        for w,v in fdist.iteritems():
            words.append(w)
            vals.append(v)

        d = {"words":words,"vals":vals}

        fdist_df = pandas.DataFrame.from_dict(d)
        #fdist_df = fdist_df.sort_values(by="vals",ascending=False)
        singletons = fdist_df["words"][fdist_df["vals"] == 1]
        singletons = singletons.get_values().tolist()
        speech_raw = [x for x in speech_raw if not x in singletons]



    #following https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    # Make dicts of words with pretrained weights
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    words = sorted(list(set(speech_raw)))
    char_indices = dict((c, i) for i, c in enumerate(words))
    indices_char = dict((i, c) for i, c in char_indices.items())
    weights = np.zeros((len(words)+1,100),dtype=np.float)

    print("loading vectors")
    word_pop = words
    glove_f = open(os.path.realpath(os.path.join(
                    os.getcwd(), 'glove/glove.6B.100d.txt')))
    for line in glove_f:
        print("on line: {}, words left:{}".format(glove_f.tell(),len(word_pop)),end="\r")
        values = line.split()
        if values[0] in word_pop:
            weights[char_indices[values[0]],:] = values[1:]
            word_pop = [x for x in word_pop if not x == values[0]]
        if len(word_pop) == 0:
            break




    # cut the text in semi-redundant sequences of maxlen characters
    if contextual:
        past_words = []
        present_words = []
        future_words = []
        for i in range(0, len(speech_raw) - (maxlen + futurelen + 1), step):
            past_words.append(speech_raw[i: i + maxlen])
            present_words.append(speech_raw[i + maxlen])
            future_words.append(speech_raw[(i + maxlen + 1): (i + maxlen + 1+futurelen)])
        print('nb sequences:', len(past_words))

        # Vectorize sentences
        nsent = len(past_words)
        past    = np.zeros((nsent, maxlen),     dtype=np.int32)
        present = np.zeros((nsent, len(words)), dtype=np.bool)
        future  = np.zeros((nsent, futurelen),     dtype=np.int32)
        for i, sentence in enumerate(past_words):
            for t, char in enumerate(sentence):
                past[i, t] = char_indices[char]
            present[i, char_indices[present_words[i]]] = 1
        for i, revsent in enumerate(future_words):
            for t, char in enumerate(revsent.reverse()):
                future[i, t] = char_indices[char]

        return past,present,future,speech_raw,words,char_indices,indices_char,nsent,weights

    else:
        sentences = []
        next_words = []
        for i in range(0, len(speech_raw) - maxlen, step):
            sentences.append(speech_raw[i: i + maxlen])
            next_words.append(speech_raw[i + maxlen])
        print('nb sequences:', len(sentences))

        # Vectorize sentences
        nsent = len(sentences)
        X = np.zeros((nsent, maxlen), dtype=np.int32)
        y = np.zeros((nsent, len(words)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t] = char_indices[char]
            y[i, char_indices[next_words[i]]] = 1

        return X,y,speech_raw,words,char_indices,indices_char,nsent,weights

# def plot_word_dist(word_list):
#     try:
#         import nltk
#         import ggplot
#         import pandas
#     except:
#         Exception("Need nltk, pandas and ggplot")

# fdist = nltk.FreqDist(word_list)

# words = []
# vals = []
# for w,v in fdist.iteritems():
#     words.append(w)
#     vals.append(v)

# d = {"words":words,"vals":vals}

# fdist_df = pandas.DataFrame.from_dict(d)
# fdist_df = fdist_df.sort_values(by="vals",ascending=False)
# fdist_df = fdist_df.set_index(np.arange(0,len(words)))
# fdist_df['idx'] = pandas.Series(np.arange(0,len(words)))

# g = ggplot(aes(x="idx",y="vals"),data=fdist_df[0:100]) + geom_point()+\
#     sca
# g = g + geom_bar()
# g

# text = nltk.Text(speech_raw)
# colo = text.collocations()

