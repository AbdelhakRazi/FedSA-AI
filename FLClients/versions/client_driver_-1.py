import math

from client_multi_current import Client
from Colors.print_colors import (print_purple,
                                 print_blue,
                                 print_cyne,
                                 print_orange)

import uuid
import json
import os
import threading as thr
# DATA
from Data_operations.DTAT_LAKE import get_data_from_file

import Data_operations.DataOperations as do

# TODO: SEE IF IT POSSIBEL TO LEAVE IT THERE
# MAX_MESSAGE_LENGTH = 1024 * 1024 * 20
MAX_MESSAGE_LENGTH = 1024 * 1024 * 390

# TODO: MOVE THEM TO A SHARED LOCATION WITH client_multi.py : DONE
CPU = do.CPU
MEMORY = do.MEMORY
NETWORK = do.NETWORK
DISK = do.DISK

# NOTICE: match the key name with that in the container launching bash script
rsc_name_dir = {"cpu": CPU, "memory": MEMORY, "network": NETWORK, "disk": DISK}

ROOT_DIR = os.getcwd()
print(ROOT_DIR)
ROOT_DIR = ROOT_DIR.split("/")
ROOT_DIR = ROOT_DIR[:len(ROOT_DIR) - 1]
ROOT_DIR = "/".join(ROOT_DIR)
print(ROOT_DIR)

# TODO in containers only: remove in normal
# ROOT_DIR="/"

fl = f'{ROOT_DIR}/FLConfig/config.json'
# fl = f'/FLDocker/FLConfig/config__.json'

#
#
#

analysis_data = {}


# URL = '163.173.228.188'
# PORT_NO = 80
# URL = 'localhost' #[::]'
# PORT_NO = 80 # 505543

def train(client_id, client_str, all_dataset, train_data, test_data, dir_path, client_file_index, client_count,
          running_mode, quantile, set_rounds=0, data_lake_param={}, rsc_target="unknown", segement_size=0):
    client = Client(target=f'{URL}:{PORT_NO}',
                    client_id=client_id,
                    client_str=client_str,
                    full_dataset=train_data,
                    analysis_dir_path=dir_path,
                    running_mode=running_mode,
                    set_rounds=set_rounds,
                    options=[
                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
                    ],
                    rsc_target=rsc_target,
                    layer=layer,
                    data_lake_param=data_lake_param,
                    quantile=quantile)

    # What to print on screen
    client.feedback_type()

    # ML model summery: layers, neurons
    client.get_model_summery('CPU')

    # what extra params
    client.get_required_params()

    # send data before training
    client_file_index = 0
    x_train = all_dataset  # FABRICATED_DATASET(data_file_path, client_id) #FABRICATED_DATASET(f'C:/Users/Salah/My notebooks/AD_DATA/{"CPU_Train_Ca.pkl"}', client_file_index)
    y_train = x_train
    # TODO: REMOVE THEM AFTER THE TEST
    # Fix the test data to the first batch
    # if running_mode["mode"].upper() == "Batch".upper():
    #     batch_size = running_mode["batch_size"]
    #     y_train = x_train[:batch_s
    #     print('y_train is assigned the first batch', y_train.shape)

    print(x_train.shape)
    ad = client.send_data(rsc_name_dir[rsc_target], x_train, y_train, test_data, segement_size=segement_size,
                          client_count=client_count)
    # analysis_data[client_str] = ad

    # send if there are more params needed
    client.send_extra_params()
    # start training
    # x=input("strat?")
    print("started")
    client.start_training(rsc_name_dir[rsc_target])


# TODO: NOT USED
def split_dataset(no_clients, train_data):
    # TODO: just for the sake of dockering
    # TODO: dataset is split and saved in en external folder
    # TODO: UNCOMMENT WHEN NEW DATASET IS PROVIDED
    print("splitting Dataset..")
    all_dataset = do.split_data_sample_intervals_per_client(train_data, no_clients)
    # do.save_pickle_to_file(all_dataset, train_data_splits_cpu_file)
    print("data split successfully")
    # TODO: ==================================================================
    return all_dataset


import sys

if __name__ == '__main__':
    config = json.load(open(fl))
    URL = config['network']['url']
    PORT_NO = config['network']['port']
    # url = f['network']['url']  # '172.17.0.2'

    fed_config = config['fed_config']

    file_paths = config['file_paths']
    running_mode = config['running_mode']

    quantile = config['quantile']

    # 1=full data. 0,5= half.  The actual data size should be loaded for training from a dataset file
    no_samples = config["data"][
        "no_samples"]  # used to have limited number of samples from the whole DB. In config.json, set it to a large NO. to have all samples in a DS

    try:
        URL = os.environ["URL"]
        client_count = os.environ['k']
        rsc_target = os.environ['rsc_target']
        layer = os.environ['layer']
        layer = os.environ['layer']
        print(URL)
    except:
        client_count = fed_config['K']  # number of participants #TODO
    print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    print("layer", layer, "\trsc_target = ", rsc_target)
    # train_data_cpu_file = file_paths['train_data_cpu_file']
    # train_data_cpu_file = f'{ROOT_DIR}{train_data_cpu_file}'
    # main_path = file_paths['main_path']

    # test_data_cpu_file = file_paths['test_data_cpu_file']
    # test_data_cpu_file = f'{ROOT_DIR}/{test_data_cpu_file}'

    output_dir_path = file_paths['output_dir_path']
    output_dir_path = f'{ROOT_DIR}/{output_dir_path}'

    current_dataset_dir = "day1"
    train_data_dir_path = file_paths[f'train_data_dir_path_{layer}']
    train_data_dir_path = f'{ROOT_DIR}/{train_data_dir_path}/{layer}/{current_dataset_dir}/{rsc_target}'  # not the file name (cpu,memory, network, disk or radio)

    train_data_splits_file = file_paths['train_data_splits_file']
    train_data_splits_file = f'{ROOT_DIR}/{train_data_splits_file}'

    # train_data_cpu = get_data_from_file(train_data_cpu_file)
    # test_data_cpu = get_data_from_file(test_data_cpu_file)
    test_data = []  # TODO: REMOVE, set it in client_multi.py file
    print_blue(fed_config)
    print_blue(running_mode)
    print_blue(config["data"])

    print(f'Connection: {URL}:{PORT_NO}')

    NUMBER_OF_PARTICIPANTS = int(client_count)

    dir_path = output_dir_path

    # for i in range(NUMBER_OF_PARTICIPANTS):
    #     print(all_dataset[i].shape, all_dataset[i][2, :, 16:18])

    analysis_data.clear()

    print(f'Number of participating members: {NUMBER_OF_PARTICIPANTS}')
    # file_name = f'C:/Users/Salah/My notebooks/AD_DATA/{"CPU_Train_Ca.pkl"}'
    # client_file_index = 0
    running_clients = []
    client_ids = []

    """"""""""""""""""""""""""""""""""""""""""""""""
    # TO RUN MULTIPLE CLIENTS
    """"""""""""""""""""""""""""""""""""""""""""""""
    # for i in range(NUMBER_OF_PARTICIPANTS):
    #     c_id = str(uuid.uuid4())
    #     print(f'client {i} [{c_id}]-> [Data shape: {all_dataset[i].shape}]')
    #     t = thr.Thread(target=train, args=(i, c_id, all_dataset[i], i, NUMBER_OF_PARTICIPANTS))
    #     running_clients.append(t)
    #     client_ids.append(c_id)
    #     t.start()

    """"""""""""""""""""""""""""""""""""""""""""""""
    # TO RUN A SINGLE CLIENT
    """"""""""""""""""""""""""""""""""""""""""""""""
    i = int(os.environ.get('CLIENTID'))  # TODO Replace with a value

    training_dataset_filepath = train_data_dir_path  # os.path.join(f'{train_data_dir_path}/training_splits', f'{NUMBER_OF_PARTICIPANTS}_dataset_split/dataset_{i}')
    print("training_dataset_file: ", training_dataset_filepath)

    dataset = get_data_from_file(training_dataset_filepath)
    # as the dataset is loaded as lstmObjct, only get the data
    # TODO: USED WITH VALUES FROM CONFIG FILE
    # TODO======= BE CAREFUL WITH THE NO. OF SAMPLES =============
    test_percent = config["test_data_percentage"]  # 0.10
    dataset = dataset.X_train_x[:no_samples]
    no_samples_ = dataset.shape[0]
    # test_data = dataset[:int(no_samples_ * test_percent)] #TODO
    test_data = dataset[-1000:]  # the last 1000 samples
    print_orange("Dataset shape:", dataset.shape)
    print_orange("Eval data shape:", test_data.shape)
    # dataset = dataset[:int(no_samples * (1-test_percent))] #TODO
    dataset = dataset[:int(no_samples)]

    # TODO========================================================

    print("Full dataset:", dataset.shape)

    # TODO: uncomment when external file is used, each client will have its own data as one file
    # train_data_multi_splits_cpu = get_data_from_file(train_data_splits_cpu_file)
    # all_dataset = train_data_multi_splits_cpu
    # TODO the client splits the dataset and keep it local: not in an external dir.
    all_dataset = split_dataset(no_clients=NUMBER_OF_PARTICIPANTS, train_data=dataset)

    client_dataset = all_dataset[i]  # Pick the split by clientno TODO, comment if not multi user..

    print_cyne(f"client dataset shape:", client_dataset.shape)

    """""
    Runining Mode 'batch' or not batch
    batch: the file is red segment by segment
    """""
    mode = running_mode['mode']
    # segement_size = running_mode['batch_size'] # TODO CAN IT BE FROM CONFIG

    sp_rounds = int(fed_config['R'])
    segment_size = client_dataset.shape[0] / sp_rounds  # Each batch is one hour

    # TODO: [comment if no setting] ADJUST PARAMETERS dependently
    if mode.lower() == "batch":
        print_blue(config["COMMENTS_"])
        settings = config['setting']
        print("Current running settings: ", end="")
        print_orange(settings)

        # TODO, Redundant
        if settings == "ROUND":  # Adjust Rounds and Fix Batch size
            sp_rounds = math.ceil(client_dataset.shape[0] / segment_size)
            print("Current rounds setting: ", end="")
            print_orange(sp_rounds)
        elif settings == "BATCH":  # Adjust Batch size and Fix Rounds
            segment_size = math.ceil(client_dataset.shape[0] / sp_rounds)
            running_mode['batch_size'] = segment_size
            print("Current batch size setting : ", end="")
            print_orange(segment_size)
        elif settings == "TIME":
            pass
        elif settings == "SAMPLE":
            pass

    # TODO: NOTICE
    # if the Dataset is large.. just get the right no

    # if mode=="batch":
    #     no_rounds_ = int(fed_config['R'])
    #     client_dataset = client_dataset[:int(batch_size) * no_rounds_]
    # else:
    #     client_dataset = client_dataset[: no_samples]

    print_blue(f'Client no: {i}')

    # not important just for checking the data..
    if len(client_dataset.shape) == 3:
        if client_dataset.shape[0] > 100 and client_dataset.shape[2] > 15:
            print(client_dataset.shape, client_dataset[2, :, 10:15])

    train(i, str(uuid.uuid4()), client_dataset,
          train_data=client_dataset,
          test_data=test_data,
          dir_path=dir_path,
          client_file_index=i,
          running_mode=running_mode,
          client_count=NUMBER_OF_PARTICIPANTS,
          set_rounds=sp_rounds,
          rsc_target=rsc_target,
          segement_size=segment_size,
          quantile=quantile)
