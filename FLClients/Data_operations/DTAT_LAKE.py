import math
import os
import random
import time
from threading import Thread

from Colors.print_colors import (print_purple,
                                 print_blue,
                                 print_cyne,
                                 print_orange)

import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from lstm_objects import lstmObjts
# import dataOperations as do
from Data_operations.DataOperations import load_pickle_file


def FABRICATED_DATASET(file_full_name, client_file_index):
    # return get_keras_data(data)
    # return get_data(random.randint(100,200))
    return get_data_from_file(file_full_name)[client_file_index]


def get_data_from_file(full_file_name):
    return load_pickle_file(full_file_name)


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


def get_data_batch___(dataset, segment_size, segment_no, rounds=0):
    print(f"\t\t\tdataset:{dataset.shape}, batch_size:{segment_size}, batch_no:{segment_no}")
    print(f"\t\t\tdataset[{int(segment_size* segment_no)}:{int(segment_size * (segment_no + 1))}]")
    segment_size = int(segment_size)

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", dataset.shape)

    new_batch = dataset[:math.ceil(segment_size)]
    rest_of_data = dataset[math.ceil(segment_size):]
    print_purple("\t\t\t batch size= ", new_batch.shape, dataset.shape, segment_size, segment_no)
    
    if segment_no <  rounds-1:
        new_batch = dataset[:math.ceil(segment_size)]
        rest_of_data = dataset[math.ceil(segment_size):]
        return new_batch, rest_of_data
    else:
        print_orange("\t\tno more batches, the last one will be repeated")
        rest_of_data = dataset
        return rest_of_data, rest_of_data


def get_data_batch(dataset, segment_size, segment_no, rounds=0):
    print(f"\t\t\tdataset:{dataset.shape}, batch_size:{segment_size}, batch_no:{segment_no}")
    #segment_size = int(segment_size)

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", dataset.shape)

    print_purple("\t\t\t batch setting:  ",  dataset.shape, segment_size, segment_no)
    if segment_no <  rounds-1:
        new_batch = dataset[math.ceil(segment_no * segment_size):math.ceil(segment_size * (segment_no + 1))]
        print(f"\t\t\tdataset[{math.ceil(segment_size* segment_no)}:{math.ceil(segment_size * (segment_no + 1))}]")
        return new_batch
    else:
        print_orange("\t\tno more batches, will return all the remaining samples")
        print(f"\t\t\tdataset[{math.ceil(segment_size* segment_no)}:]")
        return dataset[math.ceil(segment_no * segment_size):]
    



def get_data_batch__(dataset, segment_size, segment_no, rounds=0):
    print(f"\t\t\tdataset:{dataset.shape}, batch_size:{segment_size}, batch_no:{segment_no}")
    print(f"\t\t\tdataset[{int(segment_size* segment_no)}:{int(segment_size * (segment_no + 1))}]")
    segment_size = int(segment_size)

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", dataset.shape)

    new_batch = dataset[math.ceil(segment_no * segment_size):math.ceil(segment_size * (segment_no + 1))]
    print_purple("\t\t\t batch size= ", new_batch.shape, dataset.shape, segment_size, segment_no)
    if new_batch.shape[0] > 0:
        return new_batch
    else:
        print_orange("\t\tno more batches, the last one will be repeated")
        return dataset[dataset.shape[0] - (segment_size + 1):]



def get_multi_day_ds_split(split_no="", full_path_file=""):
    return load_pickle_file(full_path_file)


#TODO should be removed to a suitable pkg
def get_formatted_elapsed_time(t):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - t))


def load_database_file_split(train_data_dir_path, split_index, no_samples):
    client_data_split_file = train_data_dir_path  # os.path.join(f'{train_data_dir_path}/training_splits', f'{NUMBER_OF_PARTICIPANTS}_dataset_split/dataset_{i}')
    print("training_dataset_file: ", client_data_split_file)

    t = time.time()
    print_purple(f"loading client {split_index} dataset...")
    training_dataset = get_data_from_file(client_data_split_file)

    #TODO: remove the file is commented in using eNodeB training
    # after loading client split remove it from file system

    # Thread(target=os.remove, args=(client_data_split_file,)).start()

    print_purple(f"Dataset loaded successfully in {get_formatted_elapsed_time(t)} ...")
    # as the dataset is loaded as lstmObjct, only get the data


    # test_data = dataset[:int(no_samples_ * test_percent)] #TODO
    test_dataset = training_dataset[-1000:]  # the last 1000 samples

    print_orange("Dataset shape:", training_dataset.shape)
    print_orange("Eval data shape:", test_dataset.shape)

    # dataset = dataset[:int(no_samples * (1-test_percent))] #TODO
    # dataset = dataset[:int(no_samples)]

    # TODO========================================================

    print("Full dataset:", training_dataset.shape)

    # TODO: uncomment when external file is used, each client will have its own data as one file
    # train_data_multi_splits_cpu = get_data_from_file(train_data_splits_cpu_file)
    # all_dataset = train_data_multi_splits_cpu
    # TODO the client splits the dataset and keep it local: not in an external dir.
    # all_dataset = split_dataset(no_clients=NUMBER_OF_PARTICIPANTS, train_data=dataset)

    return training_dataset, test_dataset


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
