from __future__ import print_function
import nltk
import codecs
import re
import os
from time import strftime,gmtime

from keras.models import Model
from keras.layers import Input, Dense, Activation, LSTM, advanced_activations, merge, RepeatVector
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

pastlen = 50
futurelen = 15
step = 1

past_netsize = 128
future_netsize = 64
present_dense_netsize = 128
future_dense_netsize = 128

embed_dim = 100
L2_WEIGHT = 0.001
LEARNING_RATE = 0.001

speech_path = os.path.realpath(os.path.join(
                    os.getcwd(),
                    'trump_dox/all_combined_speeches.txt'))


past, present, future, speech_raw, words, char_indices, indices_char, nsent, weights = prep_strings(speech_path, pastlen, step, contextual=True, futurelen = futurelen)


# # Build keras model (ordered input-> output)
# INPUT
# TRY SKIPGRAMS: https://keras.io/preprocessing/sequence/

l_past_in   = Input(shape=(pastlen,))
l_future_in = Input(shape=(futurelen,))

# EMBEDDING
l_embed        = Embedding(len(words) + 1, 100,
                           weights=weights,
                           trainable=False)
l_embed_past   = l_embed(l_past_in)
l_embed_future = l_embed(l_future_in)

# LSTMs
d_lstm_past = LSTM(past_netsize,
                   dropout_W=0.2,dropout_u=0.3,
                   init="he_normal",
                   W_regularizer=l2(L2_WEIGHT),
                   inner_activation="tanh",
                   activation="tanh")
d_lstm_future = LSTM(future_netsize,
                   dropout_W=0.5,dropout_u=0.4,
                   init="he_normal",
                   W_regularizer=l2(L2_WEIGHT),
                   inner_activation="tanh",
                   activation="tanh")
l_lstm_past   = d_lstm_past(l_embed_past)
l_lstm_future = d_lstm_future(l_embed_future)

d_present_dense_1 = Dense(present_dense_netsize,activation='relu')
d_present_dense_2 = Dense(len(words)+1,activation='softmax')

d_pastfuture_cat = merge([l_lstm_past,l_lstm_future],
                         mode='concat',concat_axis=-1)
l_present_dense_1 = d_present_dense_1(d_pastfuture_cat)
l_present_dense_2 = d_present_dense_1(l_present_dense_1)


d_future_dense_1 = TimeDistributed(Dense(future_dense_netsize,activation='relu'),
                                 input_shape=(futurelen,
                                             (past_netsize + future_netsize)))
d_future_dense_2 = TimeDistributed(Dense(len(words)+1,activation='softmax'))

l_future_dense_1 = d_future_dense(d_pastfuture_cat)
l_future_dense_2 = d_future_dense_2(l_future_dense_1)

# USE REPEATVECTOR TO DO MULTIPLE FUTURE PREDICTIONS
# ALSO JUST RETURN SEQUENCE FROM FUTURE, JUST WANT TO COMBINE SEQUENCE OUTPUT FROM FUTURE LSTM AND WHATEVER WORD IS CHOSEN



model = Model(input=[l_past_in,l_future_in],
              output=[l_present_dense_2,l_future_dense_2])
model.compile(optimizer=Adam(lr=LEARNING_RATE),
              loss='categorical_crossentropy',
              loss_weights=[1,0.75])

# # # Word-level LSTM w/ dropout
# model.add(Embedding(len(words)+1, embed_dim, input_length=maxlen))

# model.add(LSTM(netsize,
#                dropout_W=0.3,dropout_U=0.5,
#                init="glorot_uniform",
#                W_regularizer=l2(L2_WEIGHT),
#                inner_activation="sigmoid",
#                return_sequences=True
#                ))
# #model.add(advanced_activations.ELU())

# model.add(LSTM(netsize,
#                dropout_W=0.5,dropout_U=0.3,
#                init="glorot_uniform",
#                inner_activation="sigmoid",
#                W_regularizer=l2(L2_WEIGHT)))
# #model.add(advanced_activations.ELU())

# model.add(Dense(len(words),activation='softmax'))

# optimizer = Adam(lr=0.002) # RMSProp w/ momentum
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Or load it
#model = load_model('/Users/Jonny/PycharmProjects/Scratch/trump_lstm/trump_word_02070916_L100_S256')
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
model_str = '/Users/Jonny/PycharmProjects/Scratch/trump_lstm/trump_contextual_word_{}_L{}_S{}'.format(strftime("%m%d%H%M", gmtime()),maxlen,netsize)


for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    print('Generating Strings')
    if i>0:
        past, present, future, speech_raw, words, char_indices, indices_char, nsent, weights = prep_strings(speech_path, pastlen, step, contextual=True, futurelen = futurelen)
    try:
        model.fit([past,future], [present,future], batch_size=64, nb_epoch=1)
    except:
        # Low rent fuckin way of doing this
        # Sometimes tokenization leaves us w/ different numbers of words. can't have that.
        past, present, future, speech_raw, words, char_indices, indices_char, nsent, weights = prep_strings(speech_path, pastlen, step, contextual=True, futurelen = futurelen)
        model.fit([past,future], [present,future], batch_size=64, nb_epoch=1)

    model.save(model_str)
    start_index = random.randint(0, len(speech_raw) - maxlen - 1)

    # Generate test strings
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        futureind = (start_index+maxlen+2+futurelen)
        sentence = speech_raw[start_index:(start_index + maxlen)]
        s_present = speech_raw[start_index + maxlen + 1]
        s_future = speech_raw[(start_index+maxlen+2):futureind].reverse()

        past_generated = sentence
        future_generated = s_future


        print('----- Generating with seed: "' + ' '.join(sentence) + '"')
        sys.stdout.write(' '.join(sentence))

        for i in range(60):
            x1 = np.zeros((1,maxlen))
            x2 = np.zeros((1,futurelen))
            for t, char in enumerate(sentence):
                if t < maxlen:
                    x1[0,t] = char_indices[char]
                else:
                    pass
            for t, char in enumerate(s_future):
                if t < futurelen:
                    x2[0,t] = char_indices[char]

            preds,future_fb = model.predict([x1,x2], batch_size=1, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            futureind = futureind+1
            sentence = sentence[1:] + [next_char]
            s_future = s_future[1:] + speech_raw[futureind]

            sys.stdout.write(" " + next_char)
            sys.stdout.flush()
        print()

