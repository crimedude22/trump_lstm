from keras.models import Sequential
from keras.layers import Input, Dense, Activation, LSTM, advanced_activations
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adagrad, Adam
from keras.regularizers import l2
from keras.models import load_model, Model

#  From Zhang, Lu, and Lapata 2016
#  arXiv:1511.00060v3
#  https://arxiv.org/pdf/1511.00060.pdf

# Network
EMBED_SIZE = 100
LSTM_SIZE = 300

# Training
SEQ_LENGTH = 151
BATCH_SIZE = 64
LEARN_RATE = 0.002
DECAY_RATE = 0

##################################################################

# Load sentences
X, y, speech_raw, words, char_indices, indices_char, nsent = prep_strings(speech_path, maxlen, step)

##################################################################


# adagrad
optimizer = Adagrad(lr=LEARN_RATE, decay=DECAY_RATE)

# Define Layers
l_input = Input(shape=(BATCH_SIZE,SEQ_LENGTH))
l_embed = Embedding(len(words) + 1, EMBED_SIZE,
                    input_length=SEQ_LENGTH,
                    initialization="glorot_normal"
                    activation="linear")(l_input)


