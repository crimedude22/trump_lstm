from __future__ import print_function
import nltk
import codecs
import re
import os
from time import strftime,gmtime

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, advanced_activations
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.models import load_model

import numpy as np
import random
import sys

maxlen = 30
netsize = 256
step = 2
embed_dim = 256
speech_path = "/path/trump_lstm/trump_dox/all_combined_speeches.txt"




# Directly tokenize
# speech_tokens = nltk.word_tokenize(speech_file.read())
# len(set(speech_tokens))

def prep_strings(speech_path,maxlen,step):
    speech_file = codecs.open(speech_path,'r','utf-8-sig')

    # Get as lines first to shuffle
    speech_raw = [line for line in speech_file]
    random.shuffle(speech_raw)

    # Join lines, tokenize & join, then eliminate whitespace before punctuation
    speech_raw = ' '.join(speech_raw)
    speech_raw = nltk.word_tokenize(speech_raw.lower())

    #following https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    # Make dicts of words
    words = sorted(list(set(speech_raw)))
    char_indices = dict((c, i) for i, c in enumerate(words))
    indices_char = dict((i, c) for i, c in enumerate(words))

    # cut the text in semi-redundant sequences of maxlen characters
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

    return X,y,speech_raw,words,char_indices,indices_char,nsent

#X, y, speech_raw, words, char_indices, indices_char, nsent = prep_strings(speech_path, maxlen, step)


# Build keras model
#model = Sequential()
# Word-level LSTM w/ dropout
#model.add(Embedding(len(words)+1, embed_dim, input_length=maxlen))
#model.add(LSTM(netsize,
#          dropout_W=0.2,dropout_U=0.3,
#          return_sequences=True
#          ))
#model.add(advanced_activations.ELU())
#model.add(LSTM(netsize, dropout_W=0.3,dropout_U=0.4))
#model.add(advanced_activations.ELU())
#model.add(Dense(len(words),activation='softmax'))

#optimizer = Adam(lr=0.005) # RMSProp w/ momentum
#model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Or load it
model = load_model('/path/trump_lstm/trump_word_01271053_L30_S256')
#print('Loaded Model!')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
model_str = '/path/trump_lstm/trump_word_{}_L{}_S{}'.format(strftime("%m%d%H%M", gmtime()),maxlen,netsize)


for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    print('Generating Strings')
    X,y,speech_raw,words,char_indices,indices_char,nsent = prep_strings(speech_path,maxlen,step)

    if iteration == 1:
        start_index = random.randint(0, len(speech_raw) - maxlen - 1)

        # Generate test strings
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = speech_raw[start_index:(start_index + maxlen)]
            generated = sentence

            print('----- Generating with seed: "' + ' '.join(sentence) + '"')
            sys.stdout.write(' '.join(sentence))

            for i in range(150):
                x = np.zeros((1,maxlen))
                for t, char in enumerate(sentence):
                    if t < maxlen:
                        x[0,t] = char_indices[char]
                    else:
                        pass

                preds = model.predict(x, batch_size=1, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated.append(next_char)
                sentence = sentence[1:] + [next_char]

                sys.stdout.write(" " + next_char)
                sys.stdout.flush()
            print()

    model.fit(X, y, batch_size=256, nb_epoch=1)
    model.save(model_str)
    start_index = random.randint(0, len(speech_raw) - maxlen - 1)

    # Generate test strings
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = speech_raw[start_index:(start_index + maxlen)]
        generated = sentence

        print('----- Generating with seed: "' + ' '.join(sentence) + '"')
        sys.stdout.write(' '.join(sentence))

        for i in range(150):
            x = np.zeros((1,maxlen))
            for t, char in enumerate(sentence):
                if t < maxlen:
                    x[0,t] = char_indices[char]
                else:
                    pass

            preds = model.predict(x, batch_size=1, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated.append(next_char)
            sentence = sentence[1:] + [next_char]

            sys.stdout.write(" " + next_char)
            sys.stdout.flush()
        print()

