import gc
import os
import stat
import time

import numpy as np
import pickle, json
from pathlib import Path
from Colors.print_colors import print_orange, print_blue, print_cyne

from sklearn.preprocessing import MinMaxScaler  # important for unpickling


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=2):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)


# remove sequence from 3D data and returns 2D np array
def flatten(data):
    flat_data = []
    for i in range(len(data)):
        flat_data.append(data[i, 0, :])
    return np.stack(flat_data)


def split_datasets(data, no_of_clients, client_no):
    no_samples = int(len(data) / no_of_clients)
    train_data = []

    if no_of_clients < client_no:
        raise ValueError('Client number exceeds the limit')

    if client_no > 0:
        i = client_no - 1
        train_data = data[i * no_samples:no_samples * (i + 1)]
    elif no_of_clients <= 0 or client_no:
        raise ValueError("Number of client and Client number can't be zero")

    return train_data


def get_clients_dataset_splits(dataset, client_count):
    client_data = []
    for i in range(1, client_count + 1):
        dt = split_datasets(dataset, client_count, i)
        client_data.append(dt)

    return client_data


def split_dataset_to_training_files(dataset_path, data_part, client_count, split_files_path, rsc, instant_label,
                                    no_training_samples):
    # TODO: just for the sake of dockerring
    # TODO: dataset is split and saved in en external folder
    # TODO: UNCOMMENT WHEN NEW DATASET IS PROVIDED
    created_files_paths = []

    splits_dir_name = create_dir(split_files_path, instant_label)

    #os.chmod(splits_dir_name, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


    t = time.time()
    print("loading lstm_obj file ", dataset_path)
    lstm_obj_file = load_pickle_file(dataset_path)
    dataset = lstm_obj_file.X_train_x[:no_training_samples]
    print("splitting dataset and saving splits to file system..")

    #TODO: this is every other sample
    all_dataset = split_data_sample_intervals_per_client(dataset, client_count)
    #print("THIS IS SPLITTING DATA PER LOT")
    #all_dataset = get_clients_dataset_splits(dataset, client_count)

    # path dir structure example: /splits_dir/cpu_1_2  >> {resourceName_day_clientIndex}
    for i in range(client_count):
        dataset_file = os.path.join(splits_dir_name, f'{rsc}_{data_part}_{i}')
        # print(f'dataset saved in file: {dataset_file}')
        save_pickle_to_file(all_dataset[i], dataset_file)
        created_files_paths.append(dataset_file)
        
        os.chmod(dataset_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    os.chmod(splits_dir_name, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    tt = time.time() - t
    msg = f"All data parts are split [{tt} sec.]"
    print("=" * (len(msg) + 1))
    print_orange(msg)
    print("=" * len(msg))
    del all_dataset
    gc.collect()

    return created_files_paths


def split_data_sample_intervals_per_client(data, client_count=0):
    all_client = [[] for i in range(client_count)]
    data_len = len(data)
    data_len_1 = data_len - 1

    for i in range(0, data_len, client_count):
        for client_idx in range(client_count):
            if client_idx + i == data_len_1:
                i = i - 1

            all_client[client_idx].append(data[client_idx + i])

    return np.stack(all_client)


def save_pickle_to_file(data_, file_full_name):
    with open(Path(file_full_name), 'wb') as f:
        pickle.dump(data_, f,protocol=4)
        f.close()
    print(f'Data saved to {file_full_name} successfully')


def load_pickle_file(file_full_name):
    with open(Path(file_full_name), 'rb') as f:
        x = pickle.load(f)
        f.close()

    print(f'Data loaded successfully')

    return x

            
def save_json_to_file(data_, file_full_name):
    with open(file_full_name, 'w') as f:
        json.dump(data_, f, indent=4,default=tuple)
    print(f'Data saved to {file_full_name} successfully')


def load_json_file(file_full_name):
    with open(file_full_name, 'r') as f:
        x = json.load(f)

    print(f'Data loaded successfully')

    return x


def create_dir(parent_dir, directory):
    path = os.path.join(parent_dir, directory)
    try:
        os.makedirs(path, exist_ok=False)
        return path
    except OSError as error:
        print(f"Directory {directory} can not be created\n", error)
        return path
