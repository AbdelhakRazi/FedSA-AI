import random
import numpy as np
import pickle
from pathlib import Path
# import DataOperations as do


def FABRICATED_DATASET(file_full_name, client_file_index):
    # return get_keras_data(data)
    # return get_data(random.randint(100,200))
    return get_data_from_file(file_full_name)[client_file_index]

def get_data_from_file(full_file_name):
    with open(Path(full_file_name), 'rb') as f:
        x = pickle.load(f)
        f.close

    print(f'Data loaded successfully')

    return x
    # return do.load_pickle_file(full_file_name)

def get_data(seed):
    # create dummy data for training
    random.seed(seed)
    x_values = [i * random.random() for i in range(1, 12)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


def get_keras_data(data):
    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps):
        X, y = [], []
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    # define input sequence
    raw_seq = data
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    return X, y