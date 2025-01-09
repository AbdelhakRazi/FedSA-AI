import enum
import fileinput
import random
from collections import OrderedDict
import numpy as np
import subprocess
# import torch

from tensor_pb2 import (Model, Empty,
                        FunctionReturns,
                        RegistrationResponse,
                        TrainingModelAndInitializationParams)

from tensor_pb2_grpc import (FederatedLearningServicer,
                             add_FederatedLearningServicer_to_server)

import grpc
import time
import pickle

import os, stat
import platform

from queue import Queue

# import torch #TODO REMOVED IN THE CONTAINER ONLY

from concurrent import futures
import sys
from Colors.console_colors import colors as cc

import threading
import copy
from Colors import print_colors as pcolors

# as there a problem for pickling tf.keras, this is a work around
# from Data_operations.tf_keras_pickel_solution import make_keras_picklable
from Data_operations import DataAnalysis as da, dataOperations as do
#from sympy.printing import str

ROOT_DIR = os.getcwd()
ROOT_DIR = ROOT_DIR.split("/")
ROOT_DIR = ROOT_DIR[:len(ROOT_DIR) - 1]
ROOT_DIR = "/".join(ROOT_DIR)
sys.path.insert(0, f'{ROOT_DIR}/Utils')
import utils
from pathlib import Path


# tensor = th.rand(60,1000, 5) #TODO: decomment later
# tensor = torch.rand(4900, 300, 1)

# print(pickle.dumps(tensor))

gRPC_MESSAGE_MAX_LENGTH = 3000
# gRPC_MESSAGE_MAX_LENGTH = 4194304 - 4  # 4 bytes are removed

ALARM_CLIENT_DISCONNECTED_TIME = 100
ROUND_WAITING_TIME = 3  # Time to wait before curring out aggregation
# MAX_ACCEPTED_CLIENTS_FOR_TRAINING = 3
MIN_FL_CLIENTS = 2
MAX_PARTICIPANTS_ALLOWED = 2

SECONDS_IN_A_DAY = 1 * 24 * 60 * 60  # (day * 24_hours * 60_minutes * 60_seconds)
#SECONDS_IN_A_DAY = 5
# TODO: SEE IF IT POSSIBEL TO LEAVE IT THERE
# MAX_MESSAGE_LENGTH = 1024 * 1024 * 20
MAX_MESSAGE_LENGTH = 1024 * 1024 * 390

MAX_WAITING_TIME_FOR_CLIENT_CONTRIBUTION = 1000_1000  # LONG time so it can't be executed
MAX_WAITING_TIME_FOR_CLIENT_CONTRIBUTION = 70 #This is in asynchronous method

class ModelTypeStruct(enum.Enum):
    TORCh = 1
    KERAS = 2
    SCIKIT = 3


class RoundStruct:
    def __init__(self, round_num, start_time, end_time, is_current=False):
        self.round_id = round_num
        self.round_num = round_num
        self.round_start_time = start_time
        self.round_end_time = end_time
        self.is_current_round = is_current


class Participant():
    def __init__(self, client_id, name, model_transmit_time=0):
        self.name = name
        self.id = client_id
        self.rge_time = time.time()
        self.last_seen = time.time()
        self.model = 0
        self.rpc_f = None
        self.my_name = ""
        self.fullNames = ""
        self.model_transmit_time = model_transmit_time
        #
        # @property
        # def last_seen(self):
        #     return self.last_seen
        #
        # @last_seen.setter
        # def last_seen(self, val):
        #     print(f'{self.name} ={val}')
        #     self.last_seen = val


CPU = 'cpu'
MEMORY = 'Memory'
NETWORK = 'NW'
DISK = 'Disk'


class FLServer(FederatedLearningServicer):
    def __init__(self, train_data, test_data, client_count=0,
                 trained_model_folder="", rsc_target="unknown", layer="unknown"):
        self.layer = layer
        self.train_data = train_data
        self.test_data = test_data
        self.model_type = None
        self.fraction = None
        self.total_rounds = None
        self.aggregation_method=None
        self.local_epochs = None
        self.batch_size = None
        self.lr = None
        self.optimizer = None
        self.start_aggregation = None
        self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING = client_count
        self.trained_model_folder = trained_model_folder
        self.rsc_target = rsc_target
        self.init()

    def __del__(self):
        # self.heartbeat_thread.stop()
        print('tic monitoring stopped')

    def init(self):

        # tf.keras pickling work around
        # make_keras_picklable()
        self.COUNT_REGISTERED_CLIENTS = 0

        self.client_id = "#"

        self.local_model = None
        self.global_model = None  # bClass.ModelCode(1, 1)
        self.current_global_model = None  # bClass.ModelCode(1, 1)
        # self.global_model = LSTM_ML_model.LSTMAutoencoder(10, 1)
        # self.serialized_global_model = pickle.dumps(self.global_model)

        self.current_round = {CPU: 0,
                              MEMORY: 0,
                              NETWORK: 0,
                              DISK: 0}

        self.current_parmQ = {CPU: Queue(),
                              MEMORY: Queue(),
                              NETWORK: Queue(),
                              DISK: Queue()}

        self.current_clients = {CPU: {},
                                MEMORY: {},
                                NETWORK: {},
                                DISK: {}}
        self.chunk_count = 1

        # mdl = pickle.dumps(LSTM_ML_model.LSTMAutoencoder)
        # fnc = pickle.dumps(LSTM_ML_model.train_tr_model)

        self.model_code = None
        self.training_function = None

        self.a = 0

        # NOTICE: The values set to true so the
        # aggregation not start when the server first run
        self.training_end = {CPU: False,
                             MEMORY: False,
                             NETWORK: False,
                             DISK: False}

        self.resource_train_time = {CPU: 0,
                                    MEMORY: 0,
                                    NETWORK: 0,
                                    DISK: 0}

        self.client_with_local_model = {CPU: {},
                                        MEMORY: {},
                                        NETWORK: {},
                                        DISK: {}}

        self.client_with_local_model_copy = {CPU: {},
                                             MEMORY: {},
                                             NETWORK: {},
                                             DISK: {}}
        self.ready_clients = 0
        self.aggregation_started = False
        self.last_seen = time.time() + SECONDS_IN_A_DAY

        self.run_once = True
        self.keep_running = True
        self.training_start_time = 0
        self.verbose = False
        self.counter = 0

        self.do_wait = SECONDS_IN_A_DAY
        self.check_trying = True

        tformated = f'{str(self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING).zfill(3)}_%H%M%S_%d%m%y'
        self.folder_string = f'{time.strftime(tformated, time.localtime())}'
        
        self.client_to_send_global_model = {}
    
    def set_client_to_send_global_model(self,client_to_send_global_model):
        self.client_to_send_global_model = client_to_send_global_model
      
    # region CALLABLE FOR GENERAL USE
    def set_ML_model_class(self, model):
        self.global_model = model
        # self.model_type = str(type(model)).replace("<class '", "").split(".")[0]
        self.model_code = pickle.dumps(model)
        print(f'Global model size [{sys.getsizeof(self.model_code):,}]]')

    def set_training_function(self, training_function):
        self.training_function = pickle.dumps(training_function)

    def init_parameters(self, total_rounds,aggregation_method, data_splits_count, local_epochs, batch_size=24, learning_rate=0.003,
                        sampling_fraction=1,
                        optimizer='ADAM'):
        self.fraction = sampling_fraction

        # TODO [remember], as dataset is very large, is split into a number (14) files
        # the number of rounds should span all these files
        self.total_rounds = total_rounds * data_splits_count  # this will also be passed by clients.. TODO solve later
        self.aggregation_method=aggregation_method

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.optimizer = optimizer

    def run_server(self, url='[::]', port='50055', max_workers=10):
        if self.global_model is None or self.training_function is None:
            raise ValueError("Either or both ML model and training method are not provided")
        elif self.total_rounds is None or self.local_epochs is None:
            raise ValueError("Either or both total rounds and local epochs are not provided")
        elif self.model_type is None or self.model_type not in ModelTypeStruct:
            raise NotImplementedError("Model type is required, e.g. torch, Keras")
        else:

            print("run     ", self.local_epochs)
            # TODO: TRY TO FIX IT
            # ck_thread = threading.Thread(target=self.check_client_existence, args=(0,)).start()
            ag_thread = threading.Thread(target=self.aggregate_local_models_cpu, ).start()

            serve(self, url, port, max_workers)

    # endregion
    def get_MSE(self, model, train_data):
        train_predicted = model.predict(train_data)
        train_flat_x = (do.flatten(train_data))
        pred_train_flat_x = (do.flatten(train_predicted))
        train_squared_error = np.square(train_flat_x - pred_train_flat_x)  # power
        train_MSE_loss = np.mean(train_squared_error, axis=1)

        return train_MSE_loss

    def send_aggregated_model(self, resource_name, s_model, client_count):
        # time.sleep(random.uniform(0.1, 0.3))
        model_size = sys.getsizeof(s_model)
        print(f'Send global model:  [{model_size:,}] bytes')

        # get The MSR
        # mse = self.get_MSE(self.global_model, self.train_data)
        # print("=" * 80)
        # print(f'MSE: {mse}')
        # print(f'average mean: {np.mean(mse)}')
        # print("=" * 80)
        
        if (self.current_round[resource_name] + 1) < self.total_rounds:
            
            self.current_round[resource_name] += 1
            
        else:
            
            if self.counter == self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING * self.total_rounds:
                self.current_round[resource_name] += 1
            
        
        self.check_trying = True

        for i in range(client_count):
            self.current_parmQ[resource_name].put(s_model)
            # print(f'\tNOTICE: chunk no: = {i} ')

    def set_clients_local_models(self, resource_name, client_id, l_model, model_transmit_time):
        # print(resource_name, client_id, sys.getsizeof(l_model), model_transmit_time)
        model = pickle.loads(l_model)
        self.current_clients[resource_name][client_id].model = model
        self.current_clients[resource_name][client_id].model_transmit_time = model_transmit_time
        self.client_with_local_model[resource_name][client_id] = self.current_clients[resource_name][client_id]
        self.last_seen = time.time()

    # not used
    def aggregate_local_models_19_01_2022(self, resource_name):  # independent thread to
        sum = 0

        while not self.training_end[resource_name]:
            if self.total_rounds > self.current_round:
                client_count = len(self.client_with_local_model)
                # time.sleep(ROUND_WAITING_TIME)
                if client_count >= self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING:  # or self.time_is_up():
                    print("aggregate local")
                    if client_count > 0:
                        t_start = time.time()
                        if self.current_round == 0:  # log time for first round to signal start of training
                            cut_time = time.strftime('%H:%M:%S %d/%m/%Y')
                            # print(pColors.OKORANGE, f'Start time {cut_time}')
                            pcolors.print_orange(f'Start time {cut_time}')
                            self.training_start_time = t_start

                        self.client_with_local_model_copy[resource_name] = copy.copy(
                            self.client_with_local_model[
                                resource_name])  # work round to use it later to count the cliet
                        self.client_with_local_model[resource_name].clear()
                        self.last_seen = time.time() + SECONDS_IN_A_DAY  # work for setting the time

                        sampled_clients = self.sample_participants(resource_name, client_count, fraction=self.fraction)

                        self.average_fd_model_to_num_of_clients(sampled_clients, resource_name)
                        self.send_aggregated_model(resource_name, pickle.dumps(self.global_model), client_count)

                        t_end = time.time()
                        total_time = t_end - t_start

                        pcolors.print_orange(f'{client_count} participants were averaged in {total_time} seconds')


                    else:
                        print("STRANGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            else:
                self.keep_running = False

                c_count = self.current_clients.keys()
                #
                # for client in self.current_clients.copy():
                #     id = self.current_clients[client].id
                #     name = self.current_clients[client].name
                #     self.current_clients.pop(client)
                #     print(cc.Blue, f'{name} [{id}] unregistered ', cc.Color_Off)

                model_size = sys.getsizeof(self.global_model)
                t_end = time.time()
                total_time = t_end - self.training_start_time
                cut_time = time.strftime('%H:%M:%S %d/%m/%Y')
                pcolors.print_orange(f'End time {cut_time}')
                pcolors.print_orange(
                    f'Average of {client_count} participants model parameters were averaged in {total_time} seconds')

                pcolors.print_cyne(
                    f' {get_time_format(self.get_average_model_transmission_time(self.client_with_local_model_copy))}')

                # Save model to a file
                file_name = 'model_cpu_trained.h5'
                self.global_model.save(f'C:/Users/Salah/My notebooks/AD_DATA/Trained_models/{file_name}')

    def aggregate_local_models(self, resource_name):  # independent thread to
       
        if self.start_aggregation is None:
            self.start_aggregation = time.time()
        if self.total_rounds > self.current_round[resource_name]:
            client_count = len(self.client_with_local_model[resource_name])
            # time.sleep(ROUND_WAITING_TIME)

            if self.check_trying:
                self.do_wait = time.time()
                self.check_trying = False
                print("\t\tcheck_trying", self.check_trying)

            force_aggregation = client_count >= self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING and not self.check_trying

            if self.aggregation_method=="async":
            
                force_aggregation = client_count > 1 and (time.time() - self.do_wait) >= MAX_WAITING_TIME_FOR_CLIENT_CONTRIBUTION \
                                    and not self.check_trying
    
            
            #if client_count >= self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING or force_aggregation:
                      
            if force_aggregation:  # or self.time_is_up():

                #print(resource_name, ' in aggr')
                if client_count > 0:
                    t_start = time.time()

                    if self.current_round == 0:  # log time for first round to signal start of training
                        cut_time = time.strftime('%H:%M:%S %d/%m/%Y')
                        pcolors.print_orange(f'[{resource_name}] Start time {cut_time}')
                        self.training_start_time = time.time()  # t_start

                        self.resource_train_time[resource_name] = t_start

                    self.client_with_local_model_copy[resource_name] = copy.copy(
                        self.client_with_local_model[resource_name])  # work round to use it later to count the cliet
                    self.client_with_local_model[resource_name].clear()
                    self.last_seen = time.time() + SECONDS_IN_A_DAY  # work a round for setting the time

                    sampled_clients = self.sample_participants(resource_name, client_count, fraction=self.fraction)

                    model_weights = self.average_fd_model_to_num_of_clients(sampled_clients, resource_name)
                    self.current_global_model = model_weights
                    # print("<|>" * 40)
                    # print(self.current_global_model[0][:5, 4:8])
                    # print("<->" * 80)
                    # self.send_aggregated_model(resource_name, pickle.dumps(self.global_model), client_count)
                    self.send_aggregated_model(resource_name, pickle.dumps(model_weights), client_count)
                    column_names=['current_round','start_aggregation', 'end_aggregation', 'participant count', 'actual partipants']
                    column_values=[str(self.current_round[resource_name]),str(self.start_aggregation),str(time.time()), str(client_count), str(sampled_clients)]
                    utils.write_to_csv_file("aggregation_"+str(self.total_rounds)+".csv", column_names,column_values)
                    self.save_trained_model(self.current_round[resource_name])
                    self.start_aggregation = None
                    
                    
                else:
                    print("-" * 80)
        else:

            model_size = sys.getsizeof(self.global_model)
            t_end = time.time()
            resource_total_time = t_end - self.resource_train_time[resource_name]
            cut_time = time.strftime('%H:%M:%S %d/%m/%Y')
            pcolors.print_orange(f'{resource_name} end time {cut_time}')
            t_t = time.time() - self.training_start_time
            t_t = time.strftime("%H:%M:%S", time.gmtime(t_t))
            
            column_names=['current_round','start_aggregation', 'end_aggregation']
            data=[str(self.current_round[resource_name]),str(self.start_aggregation),str(time.time())]
            utils.write_to_csv_file("aggregation_"+str(self.total_rounds)+".csv", column_names,data)
            self.start_aggregation = None
            print(f'Total training time: {t_t}')
            print("=" * 80)
            

            # train_MSe, \
            # test_MSE_loss, \
            # train_anomalous_data, \
            # test_anomalous_data, \
            # threshold = self.get_data_analysis(resource_name, self.global_model, self.train_data,
            #                                    self.test_data)

            
                

            # print("<||>" * 40)
            # print(self.current_global_model[0][:5, 4:8])
            # print("<|-|>" * 40)
            #TODO: NOTICE the self.global_model
            
            #self.global_model = self.current_global_model
            self.save_trained_model(self.current_round[resource_name])
            
            print("=" * 80)

            # TODO: use later
            # y = get_time_format(self.get_average_model_transmission_time
            #                     (self.client_with_local_model_copy)),

            # Save model to a file
            # TODO
            # file_name = f'model_12012022_{resource_name}_trained.h5'
            # self.global_model.save(Path(f'C:/Users/Salah/My notebooks/AD_DATA/Trained_models/{file_name}'))

            peep()  # MAKE SOUND

            self.training_end[resource_name] = True
            
            
            # self.init()
    
    def save_trained_model(self, current_round):
        
        
        self.global_model.set_weights(self.current_global_model)
        pcolors.print_orange("\tsaving trained model")
        file_name = f'{self.rsc_target}_{self.folder_string}_{current_round}.h5'
        fldr = f'{self.trained_model_folder}/{self.layer}'
        fldr_path = Path(fldr)
        if not fldr_path.exists():
            fldr.mkdir(parents=True)
        self.global_model.save(Path(f'{fldr}/{file_name}'))
        # os.chmod(f'{fldr}/{file_name}', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        file = f'{fldr}/{file_name}'
        pcolors.print_orange(f'{file_name} saved successfully at location: {fldr}')
        for key in self.client_to_send_global_model:
            destination = self.client_to_send_global_model.get(key)
            p = subprocess.Popen(["scp", file, destination])
            sts = os.waitpid(p.pid, 0)
            pcolors.print_orange(f'\tsend file to {destination}, statistics: {sts}')
   
            
        
    def get_data_analysis(self, resource_name, model, train_x, test_x, target=""):
        test_MSE_loss, \
            train_anomalous_data, \
            test_anomalous_data, \
            threshold, \
            train_MSe = da.do_it(model, train_x, test_x)

        print(f'Average MSE ({target} {resource_name}): {np.mean(train_MSe)}')
        print(
            f'Number of anomaly samples ({target} {resource_name}) :{np.sum(test_anomalous_data)} out of {len(test_anomalous_data)}')
        print(f'Indices of anomaly samples ({target} {resource_name}): {np.where(test_anomalous_data)}')

        return train_MSe, test_MSE_loss, train_anomalous_data, test_anomalous_data, threshold

    def send_global_model(self, resource_name):
        # Work round to make it wait and don't return immediately
        # logging.debug('in send model ', self.current_parmQ.qsize())
        if len(self.client_with_local_model_copy[resource_name]) == 0:
            parms = self.current_parmQ[resource_name].get()
            print('\n-----------------send_global_model-----[None]---------')

            if self.total_rounds > self.current_round[resource_name]:
                print("FL training completed")

            return Model(parameters=parms,
                         round=self.current_round[resource_name],
                         resource_name=resource_name,
                         trainingDone=self.total_rounds > self.current_round[resource_name])

        for x in range(len(self.client_with_local_model_copy)):
            parms = self.current_parmQ[resource_name].get()
            print('\n-----------------send_global_model-----------------')
            return Model(parameters=parms,
                         round=self.current_round[resource_name],
                         resource_name=resource_name,
                         trainingDone=self.total_rounds > self.current_round[resource_name])

    def get_average_model_transmission_time(self, participants):
        time_sum = 0
        participants_count = len(participants)
        if participants_count > 0:
            for client in participants:
                print(self.client_with_local_model_copy[client].model_transmit_time)
                time_sum += self.client_with_local_model_copy[client].model_transmit_time
            print(f'{time_sum} / {participants_count} = {time_sum / participants_count}')
            return time_sum / participants_count
        else:
            return -1

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {self._round}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message)

        averaged_weights = OrderedDict()
        for it, idx in enumerate(sampled_client_indices):
            local_weights = self.current_clients[idx].model.state_dict()
            print('\t', local_weights)
            for key in self.global_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
            self.global_model.load_state_dict(averaged_weights)

        message = f"[Round: {self.current_round}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        message = '\n\nDone'
        # print(message)

    def average_fd_model_to_num_of_clients(self, sampled_client_dic_keys, resource_name):
        if self.model_type.name.lower() == 'torch':
            return self.average_torch_models(sampled_client_dic_keys, resource_name)
        elif self.model_type.name.lower() == 'keras':
            return self.average_keras_models(sampled_client_dic_keys, resource_name)

    def average_torch_models(self, sampled_client_keys, resource_name):
        raise "Not implemented"
        # if len(sampled_client_keys)==0:
        #     return
        #

        averaged_weights = OrderedDict()

        num_of_client = len(sampled_client_keys)

        for it, key_ in enumerate(sampled_client_keys):
            ts = time.perf_counter()
            # local_weights = self.client_with_local_model_copy[key_].model.state_dict()
            local_weights = self.client_with_local_model_copy[key_].model

            for key in self.global_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = local_weights[key]
                else:
                    averaged_weights[key] += local_weights[key]

        # for key in self.global_model.state_dict().keys():
        for key in self.global_model.state_dict().keys():
            averaged_weights[key] = averaged_weights[key] / num_of_client

        self.global_model.load_state_dict(averaged_weights)

        message = f"[Round: {self.current_round}] ...updated weights of {len(self.client_with_local_model_copy)} clients are " \
                  f"successfully averaged! "
        # message = '\n\nDone'
        print(message);

    def average_keras_models(self, sampled_client_keys, resource_name):
        local_params = []
        for it, key_ in enumerate(sampled_client_keys):
            ts = time.perf_counter()
            # local_weights = self.client_with_local_model_copy[key_].model.get_weights()
            local_weights = self.client_with_local_model_copy[resource_name][key_].model
            local_params.append(local_weights)
            # print(f'\t model returned from client[{idx}] in {(time.perf_counter() - ts):0.5f} seconds')

        local_params = np.array(local_params, dtype=object)
        # avg_model = np.average(local_params, axis=0, dtype=object)
        avg_model = np.mean(local_params, axis=0, dtype=object)
        # self.global_model.set_weights(avg_model)
        return avg_model

    def sample_clients_(self):
        """Select some fraction of all clients."""

        # sample clients randomly
        message = f"[Round: {self.current_round}] Select clients...!"
        print(message);
        client_count = len(self.current_clients)
        num_sampled_clients = max(int(self.fraction * client_count), 1)
        sampled_client_indices = sorted(np.random.choice(
            a=[i for i in range(client_count)],
            size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def sample_participants(self, resource_name, num_client, fraction):
        if num_client > 0:
            num_sampled_clients = max(int(fraction * num_client), 1)
            # print('sample_participants ', f'{num_sampled_clients} = max(int({fraction} * {num_client}), 1)')
            return sorted(random.sample(list(self.client_with_local_model_copy[resource_name]), k=num_sampled_clients))
        return []

    def received_messages(self, request):
        self.counter += 1
        print(f'Get local model counter= {self.counter}')

        def get_client_data():
            t_start = request.act_time
            # print(f'act_time = {t_start}')
            t_end = time.time()
            model_transmit_time = t_end - t_start

            l_model = request.parameters
            client_id = request.client.clientId
            resource_name = request.resource_name
            self.set_clients_local_models(resource_name,
                                          client_id,
                                          l_model,
                                          model_transmit_time)

            # self.training_end[resource_name] = False

            # print(f'{self.current_round} {self.total_rounds}')

        threading.Thread(target=get_client_data).start()

    def aggregate_local_models_cpu(self):
        while not self.training_end[CPU]:
            self.aggregate_local_models(CPU)

    def aggregate_local_models_memory(self):
        while not self.training_end['Memory']:
            self.aggregate_local_models('Memory')

    def aggregate_local_models_nw(self):
        while not self.training_end['NW']:
            self.aggregate_local_models('NW')

    def aggregate_local_models_disk(self):
        while not self.training_end['Disk']:
            self.aggregate_local_models('Disk')

    def TheMessageCPU(self, request, context):
        # print('Entry point')
        self.received_messages(request)
        return self.send_global_model(CPU)

        # return ms.Info(clientId = 1, msgId = CPU, result = 12, round = 1)

    def TheMessageMemory(self, request, context):
        self.received_messages(request)
        return self.send_global_model('Memory')

    def TheMessageNW(self, request, context):
        self.received_messages(request)
        return self.send_global_model('NW')

    def TheMessageDisk(self, request, context):
        self.received_messages(request)
        return self.send_global_model('Disk')

    def LocalModel(self, request, context):
        self.counter += 1
        print(f'[4] Get local model counter= {self.counter}')

        def get_client_data():
            t_start = request.act_time
            print(f'act_time = {t_start}')
            t_end = time.time()
            model_transmit_time = t_end - t_start

            l_model = request.parameters
            client_id = request.client.clientId
            self.set_clients_local_models(client_id, l_model, model_transmit_time)

            print(f'{self.current_round} {self.total_rounds}')

        threading.Thread(target=get_client_data).start()

        return self.send_global_model()

    # NOT USED ANY MORE
    def GlobalModel(self, request, context):
        idx = 0
        print(f'[3] Send global model[{2}]  [{idx}]')
        parms = self.current_parmQ.get()
        response = Model(parameters=parms, round=self.current_round, modelChunkCount=self.chunk_count)
        # response = Model(parameters=self.current_parm[idx], round=self.current_round, modelChunkCount=self.chunk_count)

        return response

    def Heartbeat(self, request, context):
        self.a += 1
        self.tic_time = time.time()
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        client_id = request.clientId
        # print(f'{self.a} tic at {current_time} for {request.clientName}')
        if client_id in self.current_clients:
            self.current_clients[client_id].last_seen = time.time()

        return FunctionReturns(response=True)

    def ClientRegistration(self, request, context):
        client_registered = False
        training_model_and_function = None
        self.verbose = request.verbose
        # MAX_ACCEPTED_CLIENTS_FOR_TRAINING = request.client_count
        clientCredentials = request.clientCredentials
        client_id = clientCredentials.clientId
        client_name = clientCredentials.clientName
        resource_name = request.resource_name
        participant = Participant(client_id=client_id, name=client_name)

        pcolors.print_purple("self.total_rounds before:", self.total_rounds)
        # self.total_rounds = int(clientCredentials.clientToken)
        # pcolors.print_purple("self.total_rounds after:", self.total_rounds)

        # Used to check if the client is still exists
        participant.last_seen = time.time()

        if client_id not in self.current_clients:
            self.current_clients[resource_name][client_id] = participant
            self.COUNT_REGISTERED_CLIENTS += 1
            print(cc.Blue, f'{self.COUNT_REGISTERED_CLIENTS} {client_name} [{client_id}] registered successfully ',
                  cc.Color_Off)
            client_registered = True
        # print(f'Participants: {MAX_ACCEPTED_CLIENTS_FOR_TRAINING}')
        return RegistrationResponse(registered=client_registered)

    # TODO NEW CHANGES
    # TODO move to a suitable place
    # TODO WORK A ROUND FOR OBJECT PICKLING
    def get_model_json_and_weights(self, model, pickled=False):
        # TO SLOVE THE PROBLEM OF KERASE pickling
        j = model.to_json()
        w = model.get_weights()
        pwj = {"json": j, "weights": w}

        if pickled:
            pwj = pickle.dumps(pwj)

        # return pwj
        return model

    def TransmitInitializationParams(self, request, context):
        print("Request for GL model and init parameters submitted by ", request.clientId)

        t = time.time()
        pickled_glm = self.get_model_json_and_weights(self.global_model)  # pickle.dumps(self.global_model)
        print(f'Object serialized in {time.time() - t} sec')
        tmf = TrainingModelAndInitializationParams(nnModel=self.model_code,
                                                   modelType=self.model_type.name,
                                                   trainingFunc=self.training_function,
                                                   initialModel=pickle.dumps(pickled_glm),
                                                   epochs=self.local_epochs,
                                                   batchSize=self.batch_size,
                                                   rounds=self.total_rounds,
                                                   lr=self.lr,
                                                   folder_string=f'_{self.folder_string}')
        print('Training model and function are placed in client ', request.clientId)
        return tmf

    def StartTraining(self, request, context):
        if self.ready_clients <= self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING:
            self.ready_clients += 1

            # TODO THIS is a stupid way of changing no of rounds {as they are variables}
            # TODO Try not to forget to change is

            print(f'start training{self.ready_clients} <= {self.MAX_ACCEPTED_CLIENTS_FOR_TRAINING}')
            return FunctionReturns(response=True)

        reason = 'The number of trained clients exceeds the allowed maximum'
        return FunctionReturns(response=False, reason=reason)

    # TODO: CHECK IF REQUEST IS APPROVED BY CLIENT
    def GetPandasStore(self, request, context):
        print("Panda size", sys.getsizeof(request.data))
        data = pickle.loads(request.data)
        print("Data:\n ", data)
        return Empty()

    def check_client_existence(self, tic_time):
        print("check_heart_beat")
        try:

            while True:
                time.sleep(ALARM_CLIENT_DISCONNECTED_TIME + 0.05)
                current_delete = []
                for client in self.current_clients:
                    last_seen = self.current_clients[client].last_seen
                    if time.time() - last_seen > ALARM_CLIENT_DISCONNECTED_TIME:
                        print(
                            f' {time.time()} - {last_seen} [{time.time() - last_seen}]> {ALARM_CLIENT_DISCONNECTED_TIME}')
                        print(cc.BRed, f'CLIENT {self.current_clients[client].name} NO LONGER CONNECTED',
                              cc.Color_Off)
                        current_delete.append(client)
                for i in current_delete:
                    print("will delete ", current_delete)
                    del self.current_clients[i]
                    print(self.current_clients)
        except Exception as ex:
            print("There is an error in check_client_existence \n Exception:", ex.args)

    def GetNumberOfClients(self, request, context):
        print(f'No. of clients still in training {len(self.current_clients)}')
        for c in self.current_clients:
            print(f'\t{c}')

        return Empty()

    # will wait for some time and start training
    def time_is_up(self, ):
        # gap = WAITING_TIME_TO_START_TRAINING
        # if self.training_started:
        #     gap = 0.5
        WAITING_TIME_TO_START_TRAINING = 4
        return time.time() - self.last_seen > WAITING_TIME_TO_START_TRAINING


def get_time_format(t):
    h = 60 * 60
    m = 1 * 60
    if (t >= h):
        return time.strftime("%H:%M:%S", time.gmtime(t))
    if (t > m):
        return time.strftime("%M:%S", time.gmtime(t)) + " minutes"
    else:
        return f'{t:0.2f}' " seconds"


# region Commented code
"""""

    def LocalModel____________(self, request, context):
        print('[4] Get local model')
        model = request.parameters
        t_model = pickle.loads(model)
        # self.current_parm.append(s_model)
        # print(f'<======={sys.getsizeof(model):,}', 'LocalModel----->', model[3:20])
        print(f'{self.current_round} {self.total_rounds}')
        self.send_aggregated_model(t_model)
        return Empty()


    def GlobalModel_(self, request, context):
        idx = 0

        while True:
            # Check if there are any new messages
            # while not self.current_parm.empty():

            while len(self.current_parm) > idx:
                parm = self.current_parm[idx]
                print(f'[3] Send global model', len(self.current_parm))
                self.current_round += 1

                # response = Model(parameters=self.current_parm.get(), round=self.current_round)
                response = Model(parameters=parm, round=self.current_round, modelChunkCount=self.chunk_count)
                idx += 1
                yield response

    def InitiateFlTraining(self, request, context):
        print(f'Request for start  is made with:')  # {request.clientId}, {request.clientToken}, {request.machineName}')

        mdl = pickle.dumps(tensor)
        fl = FLInitializationParameters(
            epochs=self.epochs,
            rounds=self.total_rounds,
            initModel=mdl
        )
        print(fl)
        return fl
        # self.send_aggregated_model()

    def TrailFunction(self, request, context):
        print(
            f'[2] Request for start  is made {request}')  # with:  {request.clientId}, {request.clientToken}, {request.machineName}')
        self.current_round = 0
        # send load as torch tensor
        self.send_aggregated_model(self.serialized_global_model)
        return FLInitializationParameters(epochs=self.epochs, rounds=self.total_rounds)
"""""


# endregion


def printIt(self, c):
    cz = b''
    for chunk_no in range(self.chunk_count):
        print(f'in if {chunk_no} ? {self.chunk_count}')
        zz = self.received_parmQ.get()
        cz += zz
        print(zz)
        print('-----------------------------------------------------------------------')
        if chunk_no + 1 == self.chunk_count:
            print(f'in if last {chunk_no} ? {self.chunk_count}')
            chunk_no = 0
            tnsr = pickle.loads(cz)
            print("----", tnsr)


def printi(s):
    print(s)


def serve(server_, url, port, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]
                         )
    add_FederatedLearningServicer_to_server(server_, server)

    print(f'Listens: {url}:{port}')
    server.add_insecure_port(f'{url}:{port}')
    server.start()
    server.wait_for_termination()


# FedAvg
def server_aggregate(global_model, client_models, client_lens):
    total = sum(client_lens)
    n = len(client_models)
    # n = num_selected
    global_dict = global_model.state_dict()
    for k in global_dict.keys():  # calculate average weight/bias --> avg_w/b
        global_dict[k] -= torch.stack([client_models[i].state_dict()[k].float() *
                                       (n * client_lens[i] / total)
                                       for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())  # local model get updated weight/bias


# FedAvgM
def server_aggregate_M(global_model, client_models, client_lens):
    total = sum(client_lens)  # 592    sum [51, 122, 162, 257]
    n = len(client_models)  # 4 local clients
    global_dict = global_model.state_dict()  # weight/bias dict --> {'encoder.0.weight': Tensor with shape torch.Size([86, 115]), 'encoder.0.bias':....} 16 items
    temp = copy.deepcopy(global_dict)  # temporary weight/bias dict
    v = {x: 1 for x in copy.deepcopy(global_dict)}  # initialise v

    for i, k in enumerate(global_dict.keys()):
        # calculate average weight/bias --> avg_w/b
        temp[k] = torch.stack([client_models[i].state_dict()[k].float() * (n * client_lens[i] / total) for i in
                               range(len(client_models))], 0).mean(0)
        temp_v = 0.9 * v[k] + temp[k]  # v = 0.9v + avg_w/b   momentum=0.9
        global_dict[k] = global_dict[k] - temp_v  # w = w - v
    global_model.load_state_dict(global_dict)


def peep():
    if platform.system() == 'Windows':
        import winsound
        duration = 500  # milliseconds
        freq = 1000  # Hz
        winsound.Beep(freq, duration)
    else:
        import os
        duration = 1  # seconds
        freq = 440  # Hz
        try:
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        except Exception:
            print('No file for sound can be played')


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime
