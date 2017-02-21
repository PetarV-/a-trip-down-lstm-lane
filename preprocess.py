from keras.utils import np_utils

import numpy as np

def get_batches(filename, seq_len=100, step=1):
    # Fetch all the text in the provided file
    text = open(filename, 'r', encoding='ascii').read()
    print('Read', len(text), 'characters.')

    # Use only the characters in the text, map them to integers
    vocab = sorted(list(set(text)))
    c2i = dict((ch, i) for i, ch in enumerate(vocab))
    i2c = dict((i, ch) for i, ch in enumerate(vocab))
    print('Vocabulary size is', len(vocab), 'characters.')

    # Split the text into (semi-)overlapping sequences, record ground truths for next character
    seqs = [list(map(lambda ch: c2i[ch], text[i:(i + seq_len)])) for i in range(0, len(text) - seq_len, step)]
    nxts = [c2i[text[i + seq_len]] for i in range(0, len(text) - seq_len, step)]

    # Convert all characters into one-hot encodings
    X = np.array([np_utils.to_categorical(seq, len(vocab)) for seq in seqs])
    Y = np_utils.to_categorical(nxts, len(vocab))

    print('Input set shape:', X.shape)
    print('Output set shape:', Y.shape)

    # Return the data, along with the inverse mapping
    return ((X, Y), i2c)
