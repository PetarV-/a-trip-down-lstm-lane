from keras.layers import Input, LSTM, Dense
from keras.models import Model
from preprocess import get_batches
from keras.utils import np_utils
from keras.models import load_model

import numpy as np
import sys

def get_lstm(seq_len, vocab_len):
    # Define a simple two-layer LSTM network with 512 features
    inp = Input(shape=(seq_len, vocab_len))
    h_1 = LSTM(512, return_sequences=True)(inp)
    h_2 = LSTM(512, dropout_W=0.5)(h_1)
    out = Dense(vocab_len, activation='softmax')(h_2)

    model = Model(input=inp, output=out)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Sample a probability distribution, with a given temperature parameter
def sample(p, t=1.0):
    p = np.asarray(p).astype('float64')
    p = np.log(p) / t
    ex_p = np.exp(p)
    p = ex_p / np.sum(ex_p)
    p[p < 0] = 0
    p = p / np.sum(p)
    p = np.random.multinomial(1, p)
    return np.argmax(p)

def generate(model, i2c, seed, text_len=500, t=1.0):
    _, seed_len, vocab_len = seed.shape
    ret = ''
    while text_len > 0:
        p = model.predict(seed, verbose=0)[0]
        ind = sample(p, t)
        ch = i2c[ind]
        ret += ch
        next_vec = np_utils.to_categorical(ind, vocab_len)
        seed = np.concatenate((seed[0,1:,:], next_vec)).reshape(1, seed_len, vocab_len)
        text_len -= 1
    return ret

def train(model, X, Y, i2c, batch_size=1024, max_epochs=20, gen=True, text_len=500, t=1.0):
    for it in range(max_epochs):
        print('Epoch:', it)
        model.fit(X, Y, batch_size=batch_size, nb_epoch=5)
        model.save_weights('lstm_wts_d_en_{}.h5'.format(it))
        if gen:
            print('Sampling', text_len, 'characters at temperature', t, '...')
            seed = np.random.randint(0, X.shape[0], 1)
            text = generate(model, i2c, np.copy(X[seed]), text_len, t)
            print(text)

(X, Y), i2c = get_batches('the_trial.txt')
lstm = get_lstm(X.shape[1], X.shape[2])
train(lstm, X, Y, i2c)

