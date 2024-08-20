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

MAX_MESSAGE_LENGTH = 1024 * 1024 * 20

CPU = 'CPU'
MEMORY = 'Memory'
NETWORK = 'NW'
DISK = 'Disk'

ROOT_DIR = os.getcwd()
print(ROOT_DIR)
ROOT_DIR = ROOT_DIR.split("/")
ROOT_DIR = ROOT_DIR[:len(ROOT_DIR) - 1]
ROOT_DIR = "/".join(ROOT_DIR)
print(ROOT_DIR)

fl = f'{ROOT_DIR}/FLConfig/config.json'
f = json.load(open(fl))

# url = f['network']['url']  # '172.17.0.2'

fed_config = f['fed_config']
file_paths = f['file_paths']

train_data_cpu_file = file_paths['train_data_cpu_file']
train_data_cpu_file = f'{ROOT_DIR}{train_data_cpu_file}'
test_data_cpu_file = file_paths['test_data_cpu_file']
test_data_cpu_file = f'{ROOT_DIR}{test_data_cpu_file}'
output_dir_path = file_paths['output_dir_path']
output_dir_path = f'{ROOT_DIR}{output_dir_path}'
train_data_splits_cpu_file = file_paths['train_data_splits_cpu_file']
train_data_splits_cpu_file = f'{ROOT_DIR}{train_data_splits_cpu_file}'


try:
    URL = os.environ["URL"]
    print(">>>>>>>>>>>>>>>> ", URL)
except:
    URL = f['network']['url']

PORT_NO = f['network']['port']

print(f'Connection: {URL}:{PORT_NO}')

train_data_cpu = get_data_from_file(train_data_cpu_file)
test_data_cpu = get_data_from_file(test_data_cpu_file)
train_data_multi_splits_cpu = get_data_from_file(train_data_splits_cpu_file)

dir_path = output_dir_path
analysis_data = {}


# URL = '163.173.228.188'
# PORT_NO = 80
# URL = 'localhost' #[::]'
# PORT_NO = 80 # 505543

def train(client_id, client_str, all_dataset, client_file_index, client_count):
    client = Client(target=f'{URL}:{PORT_NO}',
                    client_id=client_id,
                    client_str=client_str,
                    full_dataset=train_data_cpu,
                    analysis_dir_path=dir_path,
                    options=[
                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
                    ], )

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
    print(x_train.shape)
    ad = client.send_data(CPU, x_train, y_train, test_data_cpu, client_count=client_count)
    # analysis_data[client_str] = ad

    # send if there are more params needed
    client.send_extra_params()
    print("><><")
    # start training
    # x=input("strat?")
    print("started")
    client.start_training()

import sys
if __name__ == '__main__':


    # __path = f"{output_dir_path}file______________.txt"
    # print(__path)
    # sys.stdout = open(__path, 'w')
    __path = f"{output_dir_path}/file______________.txt"
    print(__path)
    # sys.stdout = open(__path, 'w')

    try:
        client_count = os.environ['k']
    except:
        client_count = fed_config['K']  # number of participants #TODO
        print('Get client no. from config.File')

    NUMBER_OF_PARTICIPANTS = client_count

    all_dataset = do.split_data_sample_intervals_per_client(train_data_cpu, NUMBER_OF_PARTICIPANTS)

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
    try:
        for i in range(NUMBER_OF_PARTICIPANTS):
            c_id = str(uuid.uuid4())
            print(f'client {i} [{c_id}]-> [Data shape: {all_dataset[i].shape}]')
            t = thr.Thread(target=train, args=(i, c_id, all_dataset[i], i, NUMBER_OF_PARTICIPANTS))
            running_clients.append(t)
            client_ids.append(c_id)
            t.start()
    except:
        print("Exception!")

