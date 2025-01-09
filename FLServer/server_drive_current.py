import fileinput
import time

from server_multi_current import FLServer, ModelTypeStruct
from AutoencoderModels import karas_model

from AutoencoderModels.lstmAoencoders import KerasLSTMModel_M1
from AutoencoderModels.lstmAoencoders import KerasLSTMModel_M2
from AutoencoderModels.lstmAoencoders import KerasLSTMModel_M3
from AutoencoderModels.lstmAoencoders import KerasLSTMModel_M4

# DATA
from Data_operations.DTAT_LAKE import get_data_from_file
import Data_operations.dataOperations as do

MAX_WORKERS = 128  # max_workers

import json
import os
import numpy as np

ROOT_DIR = os.getcwd()
print("ROOT_DIR", ROOT_DIR)
ROOT_DIR = ROOT_DIR.split("/")
ROOT_DIR = ROOT_DIR[:len(ROOT_DIR) - 1]
ROOT_DIR = "/".join(ROOT_DIR)
print("ROOT_DIR", ROOT_DIR)

# TODO in containers only: remove in normal
# ROOT_DIR=""


fl = f'{ROOT_DIR}/FLConfig/config.json'

# read from a json file if no envr var ar provided
print(fl)
f = json.load(open(fl))
fed_config = f['fed_config']
optim_config = f['optim_config']

file_paths = f['file_paths']
URL = f['network']['url']
PORT_NO = f['network']['port']

fraction = fed_config["C"]
total_rounds = fed_config["R"]
aggregation_method = fed_config["M"]

data_splits_count = f['data_splits_count']
local_epochs = fed_config["E"]
batch_size = fed_config["B"]
lr = optim_config['lr']

optimizer = optim_config['optim']
client_count = fed_config['K']  # number of participants
rsc_target = f['rsc_target']
layer = f['layer']
client_to_send_globalMODEL_with_ssh = f['clients']


# number of participants #TODO number of participants

# #TODO REMOVE IT============================================
# client_count = 1


NUMBER_OF_PARTICIPANTS = client_count
MAX_WORKERS = client_count + 1  # server threads


def serve():
    # train_data_cpu_file = file_paths['train_data_cpu_file']
    # train_data_cpu_file = f'{ROOT_DIR}/{train_data_cpu_file}'
    # test_data_cpu_file = file_paths['test_data_cpu_file']
    # test_data_cpu_file = f'{ROOT_DIR}/{test_data_cpu_file}'
    # train_data_splits_cpu_file = file_paths['train_data_splits_cpu_file']
    # train_data_splits_cpu_file = f'{ROOT_DIR}/{train_data_splits_cpu_file}'
    #
    # train_data_dir_path = file_paths['train_data_dir_path']
    # train_data_dir_path = f'{ROOT_DIR}/{train_data_dir_path}'
    #
    # train_data_cpu = get_data_from_file(train_data_cpu_file)
    # test_data_cpu = get_data_from_file(test_data_cpu_file)

    trained_model_folder = file_paths["model_dir_path"]
    trained_model_folder = f'{ROOT_DIR}/{trained_model_folder}'

    analysis_dir_path = file_paths['analysis_dir_path']
    analysis_dir_path = f'{ROOT_DIR}/{analysis_dir_path}'

# print(train_data_cpu.shape)

    # split dataset to the number of clients and save each in a file
    # split_dataset_to_training_files(train_data_cpu, NUMBER_OF_PARTICIPANTS, train_data_dir_path)

    # TODO: Remove ref
    train_data_cpu = np.array([])
    test_data_cpu = np.array([])

    fl_server = FLServer(train_data=train_data_cpu, test_data=test_data_cpu, client_count=client_count,
                         trained_model_folder=trained_model_folder, rsc_target=rsc_target,layer=layer)

    model_data_shape = f[f"{layer}_data_shape"]

    feature_size = model_data_shape[rsc_target]
    print("\n---->feature_size=", feature_size)

    # model = KerasLSTMModel_M1(feature_size=60, lookback=2)
    model = KerasLSTMModel_M4(feature_size=feature_size, lookback=2)
    # model = KerasLSTMModel_M4(feature_size=1380, lookback=2)
    # model = KerasLSTMModel_M2(lookback=2)

    fl_server.model_type = ModelTypeStruct.KERAS
    print(f"-----------------{fl_server.model_type}---------------------")
   # print(model.summary())
    fl_server.set_ML_model_class(model)  # LSTM_ML_model.LSTMAutoencoder(10,1))
    fl_server.set_training_function(karas_model.keras_lstm_training)  # LSTM_ML_model.train_tr_model)

    # fl_server.set_ML_model_class(bClass.ModelCode(1,1))
    # fl_server.set_training_function(bClass.train_model)

    fl_server.init_parameters(total_rounds=total_rounds,
                              aggregation_method=aggregation_method,
                              data_splits_count=data_splits_count,
                              local_epochs=local_epochs,
                              batch_size=batch_size,
                              learning_rate=lr,
                              sampling_fraction=fraction,
                              optimizer=optimizer)
    
    fl_server.set_client_to_send_global_model(client_to_send_globalMODEL_with_ssh)
    
    fl_server.run_server(url=URL, port=PORT_NO, max_workers=MAX_WORKERS)


def split_dataset_to_training_files(dataset, client_count, path_):
    # TODO: just for the sake of dockerring
    # TODO: dataset is split and saved in en external folder
    # TODO: UNCOMMENT WHEN NEW DATASET IS PROVIDED
    print("splitting Dataset..")
    data_dir_name = do.create_dir(f'/{path_}/training_splits', f'{client_count}_dataset_split')

    t = time.time()
    all_dataset = do.split_data_sample_intervals_per_client(dataset, client_count)

    for i in range(client_count):
        dataset_file = os.path.join(data_dir_name, f'dataset_{i}')
        print(f'dataset saved in file: {dataset_file}')
        do.save_pickle_to_file(all_dataset[i], dataset_file)

    del all_dataset
    tt = time.time() - t
    print(f"data split and saved successfully in {tt} sec.")
    # TODO: ==================================================================


import socket, sys  # Import socket module

if __name__ == '__main__':

    s = socket.socket()  # Create a socket object
    host = socket.gethostname()  # Get local machine name
    local_ip = socket.gethostbyname(host)
    #
    # __path = f'{ROOT_DIR}Output/code_output_server.fl'
    # print(__path)
    # sys.stdout = open(__path, 'w')  # direct output to a file
    # # sys.stdout = sys.__stdout__ # direct output to console (standard output)

    print("Federated Learning Server ")
    print(f'Connection {local_ip}:{PORT_NO}')
    print(f'Number of participants:{client_count}')
    print(f"Training parameters: {fed_config}")
    print(f"Optimizer parameters: {optim_config}")
    serve()
