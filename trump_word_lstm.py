from __future__ import print_function
import nltk
import codecs
import re
import os
from time import strftime,gmtime

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, advanced_activations, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2,activity_l2
from keras.models import load_model

import numpy as np
import random
import sys

from net_utils import *

#################################################################
# Parameters

maxlen = 75
netsize = 128
step = 1
embed_dim = 100
L2_WEIGHT = 0.0001

speech_path = os.path.realpath(os.path.join(
                    os.getcwd(),
                    'trump_dox/all_combined_speeches.txt'))

# Directly tokenize
# speech_tokens = nltk.word_tokenize(speech_file.read())
# len(set(speech_tokens))
# X, y, speech_raw, words, char_indices, indices_char, nsent = prep_strings(speech_path, maxlen, step, remove_ones=True)


# # Build keras model
# model = Sequential()
# # # Word-level LSTM w/ dropout
# model.add(Embedding(len(words)+1, embed_dim, input_length=maxlen))

# model.add(LSTM(netsize,
#                dropout_W=0.2,dropout_U=0.2,
#                init="glorot_uniform",
#                W_regularizer=l2(L2_WEIGHT),
#                return_sequences=True,
#                activation="tanh"
#                ))
# #model.add(advanced_activations.ELU())

# model.add(LSTM(netsize,
#                dropout_W=0.2,dropout_U=0.2,
#                init="glorot_uniform",
#                W_regularizer=l2(L2_WEIGHT),
#                activation="tanh"
#                ))
# #model.add(advanced_activations.ELU())


# model.add(Dense(len(words),init="glorot_uniform",activation='softmax'))
# model.add(Dropout(0.2))

# optimizer = Adam(lr=0.002) # RMSProp w/ momentum
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Or load it
model = load_model('/Users/Jonny/PycharmProjects/Scratch/trump_lstm/trump_word_simpler_02151823_L75_S128')
print('Loaded Model!')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
# model_str = '/Users/Jonny/PycharmProjects/Scratch/trump_lstm/trump_word_simpler_{}_L{}_S{}'.format(strftime("%m%d%H%M", gmtime()),maxlen,netsize)


for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    print('Generating Strings')
    X,y,speech_raw,words,char_indices,indices_char,nsent = prep_strings(speech_path,maxlen,step,remove_ones=True)
    try:
        model.fit(X, y, batch_size=128, nb_epoch=1)
    except:
        # Low rent fuckin way of doing this
        # Sometimes tokenization leaves us w/ different numbers of words. can't have that.
        X, y, speech_raw, words, char_indices, indices_char, nsent = prep_strings(speech_path, maxlen, step)
        model.fit(X, y, batch_size=128, nb_epoch=1)

    model_str = '/Users/Jonny/PycharmProjects/Scratch/trump_lstm/trump_word_simpler_{}_L{}_S{}'.format(strftime("%m%d%H%M", gmtime()),maxlen,netsize)
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
        sys.stdout.write(' ')

        for i in range(100):
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

