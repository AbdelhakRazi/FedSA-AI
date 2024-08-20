#!/usr/bin/env python
# coding: utf-8
import json
import os
import threading as T
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import time
from tensorflow import keras
def arr_to_str(arr, preamb=""):
    return preamb + str(arr).rstrip("]").lstrip("[").replace(",", " & ")



def get_seconds_from_time_string(time_string, OK=False):  # '00:08:38'
    sec = time_string
    if OK:
        microseconds = time_string.split('.')[1]
        x = time.strptime(time_string.split(',')[0], '%H:%M:%S.%f')
        # sec = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
        micros = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec, microseconds= int(microseconds)).total_seconds()
        millisecond = micros*1000
    return  millisecond

def get_thresholds_different_quantiles(json_file_name, client_idx, quantiles=[], layer="", rsc="", fun='avg'):
    print(f"loading threshold json file {json_file_name}  [{layer} {rsc} {client_idx}]")
    jsf = load_json_from_file(json_file_name)
    jsf_client = jsf[layer][rsc][client_idx]
    # return {"avg": jsf_client['avg'] , "min": jsf_client['min'], "max": jsf_client['max']}
    thresholds = {}
    for q in quantiles:
        # thresholds[str(q)] = {fun: jsf_client["statics"][str(q)][fun]}
        thresholds[str(q)] = jsf_client["statics"][str(q)]

    return thresholds
def get_anomalies(mses, threshold):
    return np.sum(mses > threshold)


def get_anomalies_value_and_index(mses, threshold):
    anm_values = mses[mses > threshold]
    ano_indices = tuple(zip(*anm_values))

    return anm_values, ano_indices


def get_keras_model(model_name):
    if model_name == "":
        ValueError("No model provided")
    #
    try:
        model = keras.models.load_model(Path(f'{model_name}'))
    except Exception:
        model = keras.models.load_model(Path(f'{model_name}'), compile=False)

    #     model =  keras.models.load_model(Path(f'{model_name}.h5'), custom_objects={'Custom>Adam':Adam})

    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])
    return model


def get_mse(model, infrence_data):
    predicted = model.predict(infrence_data)
    flat_data = (flatten(infrence_data))
    flat_predicted = (flatten(predicted))
    test_squared_error = np.square(flat_data - flat_predicted)  # power
    # train_squared_error_pd = pd.DataFrame(np.square(train_flat_x - pred_train_flat_x))  # power
    train_MSE_loss = np.mean(test_squared_error, axis=1)
    return train_MSE_loss, flat_data

def get_mse_with_local_treshold(model, infrence_data):
    predicted = model.predict(infrence_data)
    flat_data = (flatten(infrence_data))
    flat_predicted = (flatten(predicted))
    test_squared_error = np.square(flat_data - flat_predicted)  # power
    # train_squared_error_pd = pd.DataFrame(np.square(train_flat_x - pred_train_flat_x))  # power
    train_MSE_loss = np.mean(test_squared_error, axis=1)
    threshold = np.quantile(train_MSE_loss, q=0.999) #ToDO: dont confiuse.. its only for training
    return train_MSE_loss, threshold

def get_anomalies(mses, threshold):
    return np.sum(mses > threshold)


# [Train] MSE and Threshold - Train Data
def get_train_loss(model, train_data, n_quantile = 0.999):
    train_MSE_loss = get_mse(model, train_data)
    threshold = np.quantile(train_MSE_loss, n_quantile)
    print(f'threshold = {threshold}')
    train_anomalous_data = train_MSE_loss > threshold

    return train_anomalous_data,train_MSE_loss, threshold


# [Test ] MSE and Threshold - Test Data
def get_test_loss(model, test_data, threshhold_):
    #     train_anomalous_data , threshold_ = get_train_loss(model, train_x)
    test_MSE_loss = get_mse(model, test_data)

    test_anomalous_data = test_MSE_loss > threshhold_

    return test_anomalous_data,test_MSE_loss, threshhold_

def save_json_to_file(data_, file_full_name, indent=4):
    with open(file_full_name, 'w') as f:
        json.dump(data_, f, indent=indent)

def load_json_from_file(file_full_name):
    return json.load(open(file_full_name))

def load_pickle_file(file_full_name):
    with open(Path(file_full_name), 'rb') as f:
        x = pickle.load(f)

    # print(f'Data loaded successfully')

    return x


def save_pickle_to_file_thread(data_, file_full_name):
    t = T.Thread(target=save_pickle_to_file, args=(data_, file_full_name))
    t.start()

def save_pickle_to_file(data_, file_full_name):
    with open(Path(file_full_name), 'wb') as f:
        pickle.dump(data_, f)
    print(f'Data saved to {file_full_name} successfully')


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been successfully deleted.")
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")


def flatten(data):
    flat_data = []
    for i in range(len(data)):
        flat_data.append(data[i, 0,])
    return np.stack(flat_data)


def get_data_from_file(full_file_name):
    return load_pickle_file(full_file_name)


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

def get_formatted_elapsed_time(t):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - t))


def create_dir(parent_dir, directory):
    path = os.path.join(parent_dir, directory)
    try:
        os.makedirs(path, exist_ok=False)
        return path
    except OSError as error:
        # print("Directory '%s' can not be created" % directory)
        return path
