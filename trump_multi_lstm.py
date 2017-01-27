from __future__ import print_function
import nltk
import codecs
import re
import os
from time import strftime,gmtime

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, advanced_activations
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.models import load_model

import numpy as np
import random
import sys

maxlen = 125
netsize = 128
step = "/path/trump_lstm/trump_dox/all_combined_speeches.txt"



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
    speech_raw = ' '.join(nltk.word_tokenize(speech_raw.lower()))
    speech_raw = re.sub("(?<=\w) (?=\.)","",speech_raw) # def
    speech_raw = re.sub("(?<=\w) (?=\?)","",speech_raw) # more
    speech_raw = re.sub("(?<=\w) (?=')","",speech_raw)  # better
    speech_raw = re.sub("(?<=\w) (?=,)","",speech_raw)  # wayz

    #following https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    # Make dicts of chars
    chars = sorted(list(set(speech_raw)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(speech_raw) - maxlen, step):
        sentences.append(speech_raw[i: i + maxlen])
        next_chars.append(speech_raw[i + maxlen])
    print('nb sequences:', len(sentences))

    # Vectorize sentences
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return X,y,speech_raw,chars,char_indices,indices_char

X, y, speech_raw, chars, char_indices, indices_char = prep_strings(speech_path, maxlen, step)


# Build keras model
#model = Sequential()
# Multilevel LSTM w/ dropout
#model.add(LSTM(netsize, input_shape=(maxlen, len(chars)),
#          dropout_W=0.1,dropout_U=0.15,
#          return_sequences=True
#          ))
#model.add(advanced_activations.ELU())
#model.add(LSTM(netsize, dropout_W=0.15,dropout_U=0.2))
#model.add(advanced_activations.ELU())
#model.add(Dense(len(chars),activation='softmax'))

#optimizer = Adam(lr=0.003) # RMSProp w/ momentum
#model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Or load it
model = load_model('/path/trump_lstm/trump_multi_01270044_L125_S128')
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
model_str = '/path/trump_multi_{}_L{}_S{}'.format(strftime("%m%d%H%M", gmtime()),maxlen,netsize)


for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    print('Generating Strings')
    X,y,speech_raw,chars,char_indices,indices_char = prep_strings(speech_path,maxlen,step)

    model.fit(X, y, batch_size=128, nb_epoch=1)
    model.save(model_str)
    start_index = random.randint(0, len(speech_raw) - maxlen - 1)

    # Generate test strings
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = speech_raw[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


