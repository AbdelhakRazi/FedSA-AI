import  os
import numpy as np
import pickle, json
from pathlib import Path


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
        pickle.dump(data_, f)
        f.close()
    print(f'Data saved to {file_full_name} successfully')


def load_pickle_file(file_full_name):
    with open(Path(file_full_name), 'rb') as f:
        x = pickle.load(f)

    print(f'Data loaded successfully')

    return x

def save_json_to_file(data_, file_full_name):
    with open(Path(file_full_name), 'w') as f:
        json.dump(data_, f)
        f.close()
    print(f'Data saved to {file_full_name} successfully')


def load_json_file(file_full_name):
    with open(Path(file_full_name), 'r') as f:
        x = json.load(f)
        f.close()

    print(f'Data loaded successfully')

    return x


def create_dir(parent_dir, directory):
    path = os.path.join(parent_dir, directory)
    try:
        os.makedirs(path, exist_ok=False)
        return path
    except OSError as error:
        print("Directory '%s' can not be created" % directory)
        return path