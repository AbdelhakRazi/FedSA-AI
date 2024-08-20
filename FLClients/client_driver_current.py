import math
import time
import sys
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
from Data_operations.DTAT_LAKE import get_data_from_file, load_database_file_split

import Data_operations.DataOperations as do



# TODO: SEE IF IT POSSIBEL TO LEAVE IT THERE
# MAX_MESSAGE_LENGTH = 1024 * 1024 * 20
MAX_MESSAGE_LENGTH = 1024 * 1024 * 390

# TODO: MOVE THEM TO A SHARED LOCATION WITH client_multi.py : DONE
CPU = do.CPU
MEMORY = do.MEMORY
NETWORK = do.NETWORK
DISK = do.DISK
ENB = do.ENB

# NOTICE: match the key name with that in the container launching bash script
rsc_name_dir = {"cpu": CPU, "memory": MEMORY, "network": NETWORK, "disk": DISK, "enb": ENB}

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

def train(client_id, client_str, training_data_dir_path, test_data, analysis_dir_path, client_file_index, client_count,
          running_mode, quantile, rounds=0, epochs_before_aggregation=0,data_splits_count=0, data_lake_param={}, rsc_target="unknown",
          number_of_participants=0, no_samples=0, test_data_percent=0.01, lr = 0.001,data_pipeline_parameter={}):
    client = Client(target=f'{URL}:{PORT_NO}',
                    client_id=client_id,
                    client_str=client_str,
                    training_data_dir_path=training_data_dir_path,
                    analysis_dir_path=analysis_dir_path,
                    running_mode=running_mode,
                    rounds=rounds,
                    epochs_before_aggregation=epochs_before_aggregation,
                    data_splits_count=data_splits_count,
                    options=[
                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
                    ],
                    rsc_target=rsc_name_dir[rsc_target],
                    layer=layer,
                    data_lake_param=data_lake_param,
                    quantile=quantile,
                    number_of_participants=number_of_participants,
                    test_data_percent=test_data_percent,
                    no_training_samples= no_samples,
                    lr = lr,
                    data_pipeline_parameter=data_pipeline_parameter)

    # What to print on screen
    client.feedback_type()

    # ML model summery: layers, neurons
    client.get_model_summery('CPU')

    # what extra params
    client.get_required_params()

    # send data before training
    client_file_index = 0

    # TODO: REMOVE THEM AFTER THE TEST
    # Fix the test data to the first batch
    # if running_mode["mode"].upper() == "Batch".upper():
    #     batch_size = running_mode["batch_size"]
    #     y_train = x_train[:batch_s
    #     print('y_train is assigned the first batch', y_train.shape)


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



if __name__ == '__main__':

    print("in client")
    # while True:
    #     d = get_real_time_training_data()
    #     print(d.shape)
    # 
    # #TODO####################################
    # exit()
    # #TODO####################################

    config = json.load(open(fl))
    URL = config['network']['url']
    PORT_NO = config['network']['port']
    # url = f['network']['url']  # '172.17.0.2'

    fed_config = config['fed_config']
    optim_config = config['optim_config']

    file_paths = config['file_paths']
    running_mode = config['running_mode']

    quantile = config['quantile']



    #TO output erros to file
    #f_path = file_paths['output_dir_path']
    #ff_file = f"{f_path}/errors_output"
    #print(f"error file path: {ff_file}")
    #sys.stderr = open(ff_file, "w")





    # 1=full data. 0,5= half.  The actual data size should be loaded for training from a dataset file
    no_samples = config["dataset_shape"]["no_samples"]  # used to have limited number of samples from the whole DB. In config.json, set it to a large NO. to have all samples in a DS

    data_splits_count = config['data_splits_count']
    split_label = config['split_label']  # day
    learning_rate = optim_config['lr']
    client_count = fed_config['K']  # number of participants
    rsc_target = config['rsc_target']
    layer = config['layer']
    splits_instance_label = config['split_label']

    training_data_splits_path = file_paths['train_data_splits_path']
    training_data_splits_path = f'{ROOT_DIR}/{training_data_splits_path}'

    analysis_dir_path = file_paths['analysis_dir_path']
    analysis_dir_path = f'{ROOT_DIR}/{analysis_dir_path}'
    
    hostname = os.uname()[1]
    i = int(hostname.split('-')[1]) #IF hostname = TFX-3, then i=3


    print(f"___________________ Client{i} __________________")
    print("layer", layer, "\trsc_target = ", rsc_target)

    test_data = []  # TODO: REMOVE, set it in client_multi.py file
    print_blue(fed_config)
    print_blue(running_mode)
    print_blue(config["dataset_shape"])

    print(f'Connection: {URL}:{PORT_NO}')

    NUMBER_OF_PARTICIPANTS = int(client_count)

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
    split_index = i
    day_index=1
    training_data_dir_path = f'{training_data_splits_path}/{splits_instance_label}'
    # TODO: USED WITH VALUES FROM CONFIG FILE
    # TODO======= BE CAREFUL WITH THE NO. OF SAMPLES =============
    test_data_percent = config["test_data_percentage"]  # 0.10
    # client_dataset = load_database_file_split(training_data_dir_path, day_index, no_samples, test_data_percent)

    """""
       Runining Mode 'batch' or not batch
       batch: the file is red segment by segment
       """""
    mode = running_mode['mode']
    # segement_size = running_mode['batch_size'] # TODO CAN IT BE FROM CONFIG

    sp_rounds = int(fed_config['R'])
    sp_epochs = int(fed_config['E'])
    # segment_size = client_dataset.shape[0] / sp_rounds  # Each batch is one hour

    # TODO: [comment if no setting] ADJUST PARAMETERS dependently
    if mode.lower() == "batch":
        print_blue(config["COMMENTS_"])
        settings = config['setting']
        print("Current running settings: ", end="")
        print_orange(settings)

        # TODO, Redundant
        if settings == "ROUND":  # Adjust Rounds and Fix Batch size
            # sp_rounds = math.ceil(client_dataset.shape[0] / segment_size)
            print("Current rounds setting: ", end="")
            print_orange(sp_rounds)
        elif settings == "BATCH":  # Adjust Batch size and Fix Rounds
            # segment_size = math.ceil(client_dataset.shape[0] / sp_rounds)
            # running_mode['batch_size'] = segment_size
            print("Current batch size setting : ", end="")
            # print_orange(segment_size)
        elif settings == "TIME":
            pass
        elif settings == "SAMPLE":
            pass


    print_blue(f'Client no: {i}')

    if len(sys.argv) < 3:
        self_file = sys.argv[0]
        print("Usage: python3 "+self_file+" eNDBF_server:port_number consumer_topic_name")
        sys.exit(1)
        
    bootstrap_server = str(sys.argv[1])
    topic_name = str(sys.argv[2])
    
    data_pipeline_parameter_ = {
            
            "topic_name": topic_name,  # Name of the topic from where to get the data
            "bootstrap_server":bootstrap_server,  # Ip address of the borker (eNDBF)
            "consumer_group": "preprocessed_cpu_group"
    }
    
    train(i, str(uuid.uuid4()),
          training_data_dir_path=training_data_dir_path,
          test_data=test_data,
          analysis_dir_path=analysis_dir_path,
          client_file_index=i,
          running_mode=running_mode,
          client_count=NUMBER_OF_PARTICIPANTS,
          rounds=sp_rounds,
          epochs_before_aggregation=sp_epochs,
          data_splits_count=data_splits_count,
          rsc_target=rsc_target,
          quantile=quantile,
          number_of_participants=NUMBER_OF_PARTICIPANTS,
          test_data_percent =test_data_percent,
          no_samples=no_samples,
          lr= learning_rate,
          data_pipeline_parameter=data_pipeline_parameter_)
