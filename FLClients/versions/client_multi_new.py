import json
import os
import platform
import numpy as np

import tensor_pb2 as tr
import tensor_pb2_grpc as tr_rpc
import grpc
import time
import pickle
import threading

import sys
# import torch as th
# import torch.nn as nn


# as there a problem for pickling tf.keras, this is a work around
# from tf_keras_pickel_solution import make_keras_picklable
from Data_operations.DTAT_LAKE import get_data_batch

from Colors.print_colors import (print_purple,
                                 print_blue,
                                 print_cyne,
                                 print_orange)

import Data_operations.DataOperations as do
import Data_operations.DataAnalysis as da

from keras.models import model_from_json

gRPC_MESSAGE_MAX_LENGTH = 4194304 - 4  # 4 bytes are removed

CHECK_IS_ALIVE_INTERVAL = 3  # a heart pulse every # seconds

# TODO: SEE IF IT POSSIBEL TO LEAVE IT THERE
# MAX_MESSAGE_LENGTH = 1024 * 1024 * 20
MAX_MESSAGE_LENGTH = 1024 * 1024 * 390

#TODO: MOVE THEM TO A SHARED LOCATION WITH client_drive.py : DONE
CPU = do.CPU
MEMORY = do.MEMORY
NETWORK = do.NETWORK
DISK = do.DISK

from enum import Enum


class KerasVerbosity(Enum):
    silent = 0
    progress_bar = 1
    one_line = 2


class Client:
    _options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ]

    def __init__(self, client_id, client_str, dir_path, target="localhost:50055", running_mode={}, set_rounds=0,
                 options=_options,
                 rsc_target="", layer="", data_lake_param={},
                 *args,
                 **kwargs):

        print(f"connection {target}")
        # make_keras_picklable()  # WORKROUND
        self.layer = layer
        self.client_crd = tr.ClientCredentials(
            clientId=client_str,
            clientName=platform.uname()[1],  # node
            # TODO: Replace [currently used to tell server the the calculated no of rounds]
            clientToken=str(set_rounds),
            machineName='Salah Bin',  # TODO:    Replace
            machineHW=platform.uname()[4],  # machine
        )
        self.rsc_target = rsc_target  # TODO: just for the sake of the out put: modify to be general
        self.client_id = client_id
        self.analysis_data_dir_path = dir_path
        self.client_count = 2

        self.channel = grpc.insecure_channel(target=target, options=options)
        self.conn = tr_rpc.FederatedLearningStub(self.channel)

        self.transmit_funcs = {CPU: self.conn.TheMessageCPU,
                               MEMORY: self.conn.TheMessageMemory,
                               NETWORK: self.conn.TheMessageNW,
                               DISK: self.conn.TheMessageDisk}

        # self.full_dataset = full_dataset
        self.test_dataset = {CPU: None,
                             MEMORY: None,
                             NETWORK: None,
                             DISK: None}
        self.global_model = {CPU: None,
                             MEMORY: None,
                             NETWORK: None,
                             DISK: None}
        self.local_model = {CPU: None,
                            MEMORY: None,
                            NETWORK: None,
                            DISK: None}

        self.model_type = ''
        self.current_round = 0
        self.total_rounds = set_rounds

        self.epochs = -1
        self.batch_size = -1
        self.segment_size = -1
        self.chunk_count = 1

        self.nnModel = None
        self.training_func = None
        self.training_data = {CPU: None,
                              MEMORY: None,
                              NETWORK: None,
                              DISK: None}
        self.evaluation_data = {CPU: None,
                                MEMORY: None,
                                NETWORK: None,
                                DISK: None}

        self.training_finished = False

        self.counter = 0
        self.print = {CPU: print_orange,
                      MEMORY: print_purple,
                      NETWORK: print_blue,
                      DISK: print_cyne}

        self.round_analysis = []
        self.full_analysis = {str(client_id): {}}
        self.output_folder = "output_file"
        self.model_sent_time = time.time()
        self.model_exchange_time = time.time()
        self.model_size = 0.0

        self.start_time = time.localtime()
        self.start_timestamp =time.time()

        self.running_mode = running_mode
        self.init()
        # print('Attributes Placed successfully')

    # def initialize_training(self):
    #     print('<<<<<<<<<<<<<<<<<<<<<<')
    #     qw = self.conn.InitiateFlTraining(self.client_crd)
    #     print('<<<<<<<<>>>>>>>>>>>>>>>>')
    #     # self.total_rounds = init.rounds
    #     # self.epochs = init.epochs
    #     # print({self.total_rounds}, self.epochs, init.initModel)
    #     print(">>>sssssssssssssssss>", qw.rounds)

    # TODO GET WEIGHTs
    # TODO loacl_model added
    def get_model_to_send(self, resource_name, local_model):
        # print(' In def get_model_chunks(self)')
        s_model = pickle.dumps(local_model)  # * random.randint(1, 10))
        print(f'model size = {sys.getsizeof(s_model)}')

        return tr.Model(resource_name=resource_name, parameters=s_model, client=self.client_crd, act_time=time.time())

    def transmit_local_model(self, resource_name, local_model):
        self.print[resource_name](f'local trained model will be transmitted  {resource_name}')
        t_start = time.time()
        self.counter += 1

        l_model = self.get_model_to_send(resource_name, local_model)
        ts = time.time()
        # TODO
        # THIS MUST be uncommitted when multi-threading
        # g_model = self.transmit_funcs[resource_name](l_model)

        g_model = self.transmit_funcs[resource_name](l_model)
        self.model_exchange_time = time.time() - ts
        self.get_global_model_and_train(resource_name, g_model)

    def get_global_model_and_train(self, resource_name, model):
        # def set_global_model(self, resource_name, model):
        # model = self.conn.GlobalModel(tr.Empty())  # this line will wait for new messages from the server!

        self.current_round = model.round
        # self.print[resource_name](
        #     f'Starting [Round  ({resource_name}): {self.current_round}/{self.total_rounds}]')
        g_model = model.parameters

        local_model = self.train_global_model(resource_name, g_model)
        if model.trainingDone:  # TODO : CHANGE to trainingNotDone or reverse the value from server
            self.transmit_local_model(resource_name, local_model)

            print_orange(' Global model and init params received in ', self.model_exchange_time)
        else:
            self.print[resource_name]('training_finished')
            #
            # dir_name = f'analysis_data_time_{time.strftime("%H_%M_%S-%d%m%y", time.localtime())}'

            print_cyne(f"Analysis for client[{str(self.client_id)}]:\n", self.full_analysis)
            print_cyne(f"Analysis for client[{str(self.client_id)}]:\n", self.round_analysis)

            # json_output = json.dumps(self.full_analysis)

            mode = self.running_mode['mode']
            if mode.upper() == "batch".upper():
                segment_size = self.segment_size  # self.running_mode['batch_size']
            else:
                segment_size = "-"

            training_settings = {
                "epochs": self.epochs,
                "mode": mode,
                "Rounds": self.total_rounds,
                "data_shape": self.training_data[resource_name].shape,
                "segment_size": self.segment_size
            }
            w_t = get_formatted_elapsed_time(self.start_timestamp)
            self.full_analysis[str(self.client_id)]["resource_target"] = self.rsc_target
            self.full_analysis[str(self.client_id)]["training_settings"] = training_settings
            self.full_analysis[str(self.client_id)]["model_info"] = {"model_type": self.model_type.lower(),
                                                                     "model_size": f'{self.model_size:,} bytes'}
            self.full_analysis[str(self.client_id)]["FD total_training_time"] = \
                {"total":w_t,
                 "start time": time.strftime("%H:%M:%S",self.start_time),
                 "end time": time.strftime("%H:%M:%S",time.localtime())}
            self.full_analysis[str(self.client_id)][f"rounds_anl [{self.total_rounds}]"] = self.round_analysis
            dir_path = do.create_dir(f"{self.analysis_data_dir_path}/{self.layer}",
                                     f"{self.rsc_target}{self.output_folder}")
            do.save_json_to_file(self.full_analysis, os.path.join(dir_path, f'{str(self.client_id)}.json'))
            # do.save_json_to_file(self.round_analysis, os.path.join(dir_path, f'{str(self.client_id)}.json'))

            self.training_finished = True

    def train_global_model(self, resource_name, g_model):
        # print(self.model_type, "<+><"*30)
        global_model_weights = pickle.loads(g_model)
        self.global_model[resource_name].set_weights(global_model_weights)
        # print(self.global_model)
        local_model = self.do_local_training(resource_name, self.global_model[resource_name])
        return local_model

    def do_local_training(self, resource_name, g_model):
        if self.model_type.lower() == 'torch':
            return self.train_torch_model(resource_name, g_model)
        elif self.model_type.lower() == 'keras':
            return self.train_keras_model(resource_name, g_model)

    # TODO activate hartbeat
    def send_heartbeat(self):
        try:
            while not self.training_finished:
                rsp = self.conn.Heartbeat(self.client_crd)
                time.sleep(CHECK_IS_ALIVE_INTERVAL)

            print(" training finsh -------------------------------------------------")

        except Exception:
            print("Gone ")
            print(type(Exception))  # the exception instance
            print(Exception.args)  # arguments stored in .args

    # First contact with server

    def register_client(self):
        # TODO  verbose=True  implement it at  server side
        # try:
        rsp = tr.RegistrationParams(clientCredentials=self.client_crd, verbose=True, resource_name=CPU,
                                    client_count=self.client_count)

        print_blue("Try to register client with FL server ")
        t = time.time()
        rsp = self.conn.ClientRegistration(rsp)
        if rsp.registered:
            print(f'client  registered successfully [{time.time() - t}] sec')
            self.get_training_model_and_init_params()

    # except Exception as ex:
    #     print_purple(f"Connection problem\n{ex}")

    def get_training_model_and_init_params(self):

        # try:
        print_orange(f'Getting global model and init params ... ')
        t_start = time.time()
        tmf = self.conn.TransmitInitializationParams(self.client_crd)
        self.model_exchange_time = time.time() - t_start
        print_orange(f'Global model and init params received in {self.model_exchange_time}')

        # get the model code class and training routine
        # print("Epoch === ", tmf.epochs)
        self.epochs = tmf.epochs
        self.batch_size = tmf.batchSize
        # self.nnModel = pickle.loads(tmf.nnModel)
        # self.training_func = pickle.loads(tmf.trainingFunc)
        self.training_func = keras_lstm_training  # TODO RRMOVE THE TRAINING FUNCTION FROM THE SERVER

        # Todo: removed because we need to have the total_rounds variable
        # self.total_rounds = tmf.rounds

        self.global_model[CPU] = tmf.initialModel
        self.global_model[MEMORY] = tmf.initialModel
        self.global_model[NETWORK] = tmf.initialModel
        self.global_model[DISK] = tmf.initialModel

        self.model_type = tmf.modelType

        self.output_folder = tmf.folder_string

        t_end = time.time()
        total_time = t_end - t_start
        #
        # print_orange(
        #     f'Download ML model and training function from aggregator server in {total_time} seconds',
        # )
        print_orange(f"Model size {sys.getsizeof(tmf.initialModel):,} bytes")
        self.model_size = sys.getsizeof(tmf.initialModel)

        #
        #     return True
        # except KeyboardInterrupt:
        #     print('Exiting, please wait...')
        #     print('Finished.')
        #     return False
        # except grpc.RpcError as er:
        #     print(f'RPC exception')
        #     for ar in er.args:
        #         print(ar)
        #     print(grpc.RpcContext.__name__)
        #
        #     return False
        # except Exception:
        #     print("Exception ")
        #     print(type(Exception))  # the exception instance
        #     print(Exception.args)  # arguments stored in .args
        #     return False

    def proxFed(self, l_model, g_model, mu=0):
        # for fedprox
        fed_prox_reg = 0.0
        fed_prox_reg = ((mu / 2) * (l_model - g_model) ** 2)
        return fed_prox_reg

    def train_keras_model(self, resource_name, k_model):
        if self.current_round <= self.total_rounds - 1:
            self.print[resource_name](
                f'({resource_name}|-> Starting [Round  : {self.current_round + 1}/{self.total_rounds}]')

            train_data = self.training_data[resource_name]
            eval_data = self.evaluation_data[resource_name]

            # keep if the round more than the batch no can support

            print("\t\t", len(self.running_mode.keys()))
            print("\t\t", self.running_mode["mode"])

            # TODO: CHECK IT IN DIFFERENT SETTING
            if len(self.running_mode.keys()) > 0:
                if self.running_mode["mode"].lower() == "batch":
                    print("in batch mode")
                    # TODO: NOTES if no more batches, the last one will be repeated"
                    segment_size = self.segment_size  # self.running_mode["batch_size"]
                    train_data = get_data_batch(train_data, segment_size, self.current_round)
                    # TODO: NOTES Fix the First Batch for eval during training be repeated"
                    # eval_data = get_data_batch(eval_data, batch_size, self.current_round)

            x_train = train_data
            y_train = eval_data
            print(f'\t\tdata shape x={x_train.shape} y={y_train.shape}')
            print_cyne('Got From Server')

            # GET the Mean square Error
            # mse = self.get_MSE(k_model, self.full_dataset)

            # insure they have same dimensions
            if x_train.shape[0] < y_train.shape[0]:
                y_train = y_train[:x_train.shape[0]]
            elif x_train.shape[0] > y_train.shape[0]:
                x_train = x_train[:y_train.shape[0]]

            x_train = np.flip(x_train, axis=0)
            y_train = np.flip(y_train, axis=0)

            vr = KerasVerbosity.progress_bar.value
            vr = KerasVerbosity.one_line.value
            vr = KerasVerbosity.silent.value

            s_t = time.time()
            st_t = time.strftime("%H:%M:%S", time.localtime())
            print(f'Training started at:{st_t}')

            # print_purple("<>"*40)
            # print_cyne(k_model.get_weights()[0][:5, 4:8])
            # print_purple("-" * 80)

            self.training_func(x_train, y_train,
                               self.epochs,
                               k_model,
                               batch_size=self.batch_size,
                               verbose=vr)
            # print_cyne(k_model.get_weights()[0][:5, 4:8])
            # print_purple("<>" * 40)

            # k_model.fit(x_train, y_train, self.epochs, verbose=1)

            ed_t = time.strftime("%H:%M:%S", time.localtime())
            print(f'Trining ended at:{ed_t}')
            e_t = time.time()
            t_t = e_t - s_t
            t_t = time.strftime("%H:%M:%S", time.gmtime(t_t))
            print(f'Training time {t_t}')

            # print("=" * 80)
            # print(f'[Global] MSE: {mse}')
            # print(f'[Global] average  mean: {np.mean(mse)}')
            # GET the Mean square Error
            print("=" * 80)
            train_MSe, \
            test_MSE_loss, \
            train_anomalous_data, \
            test_anomalous_data, \
            threshold = self.get_data_analysis(resource_name, k_model, x_train,
                                               self.test_dataset[resource_name])
            print("=" * 80)

            dtan = {"client_id": self.client_id,
                    "round_no": int(self.current_round),
                    "start_time": st_t,
                    "end_time": ed_t,
                    "training_time": t_t,
                    # "train_MSE": train_MSe,
                    "average_MSE": np.mean(train_MSe),
                    # "test_MSE_loss": test_MSE_loss,
                    "threshold": threshold,
                    "anomalous_count": int(np.sum(test_anomalous_data)),
                    "total_samples": int(len(test_anomalous_data)),
                    "model_round_trip": self.model_exchange_time
                    }

            self.round_analysis.append(dtan)
            # # TODO: REMOVE=====================================================
            # print("=" * 50, self.full_analysis, "=*50")
            # dir_name = f'analysis_data_time_{time.strftime("%H_%M_%S-%d%m%y", time.localtime())}'
            # dir_path = do.create_dir(self.analysis_data_dir_path, dir_name)
            # do.save_pickle_to_file(self.full_analysis, os.path.join(dir_path, str(self.client_id)))
            # #TODO===================================================
        # self.local_model[resource_name] = k_model.trainable_weights
        # return k_model.trainable_weights
        # self.local_model[resource_name] = k_model.get_weights()
        return k_model.get_weights()

    def train_torch_model(self, resource_name, nn_model):
        inputDim = 1  # takes variable 'x'
        outputDim = 1  # takes variable 'y'
        learningRate = 0.01

        # model = nn_model(inputDim, outputDim)
        model = nn_model

        ##### For GPU #######
        if th.cuda.is_available():
            model.cuda()

        criterion = nn.MSELoss()
        optimizer = th.optim.SGD(model.parameters(), lr=learningRate)

        # optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
        # criterion = nn.L1Loss()

        x_train = self.training_data[resource_name]
        y_train = self.evaluation_data[resource_name]

        local_weights = model.state_dict()

        print_cyne('Got From Server')
        for key in model.state_dict().keys():
            print_cyne(f'{key} = {local_weights[key]}')

        if not self.training_finished:
            self.training_func(x_train, y_train, self.epochs, model, optimizer, criterion)
            # self.local_model = model
            self.local_model[resource_name] = model.state_dict()

        print_cyne('Will Send To Server')
        for key in model.state_dict().keys():
            print(print_cyne(f'{key} = {local_weights[key]}'))

        """""
        self.global_model used for training and the result is saved in self.local_model 
        """""

    # TODO Be CHANGED FOR model JSON AND WEIGHTs
    def ready_to_start(self, resource_name):
        OK = True
        # try:
        OK = False

        response = self.conn.StartTraining(self.client_crd)
        print(f'Ready to start = {response}')
        self.start_timestamp = time.time()
        self.start_time = time.localtime()
        if response.response:
            self.print[resource_name]("----> ", resource_name)
            g_model = self.global_model[resource_name]
            # ----------------------------------------------
            g_model = pickle.loads(g_model)
            # self.global_model_json[resource_name]=g_model['json']
            # model_weight = g_model['weight']
            loaded_model = model_from_json(g_model['json'])
            loaded_model.compile(optimizer='adam', loss='mse')
            loaded_model.set_weights(g_model['weights'])

            self.global_model[resource_name] = loaded_model

            local_model = self.do_local_training(resource_name, loaded_model)
            # ----------------------------------------------
            self.transmit_local_model(resource_name, local_model)
        else:
            print('Can not start training right now. Try to start another time')

    # except Exception as ex:
    #     print_purple(f"Connection problem\n{ex}")
    #     # while OK:
    #     #     print_purple("will try to connect in 3 sec")
    #     #     time.sleep(3)
    #     #     self.ready_to_start(resource_name)
    #
    #
    #         print(ex)
    # What to print on screen
    def feedback_type(self):
        print('feedback_type')

    # ML model summery: layers, neurons
    # TODO MAKE IT GENERAL for all resources
    def get_model_summery(self, resource_name):
        print('model_summery: ')
        # print_cyne(pickle.loads(self.global_model[resource_name]))

    # what extra params
    def get_required_params(self):
        print('get_required_params')

    # send data before training
    def send_data(self, resource_name, training_data, evaluation_data, test_data="", segement_size =-1, client_count=2):
        self.training_data[resource_name] = training_data
        self.evaluation_data[resource_name] = evaluation_data
        self.test_dataset[resource_name] = test_data
        self.segment_size=segement_size
        self.client_count = client_count
        self.round_analysis.clear()
        # self.full_analysis.clear()
        print("segement_size: ",segement_size)
        return

    # send if there are more params needed
    def send_extra_params(self):
        pass

    def get_data_analysis(self, resource_name, model, train_x, test_x, target=""):
        test_MSE_loss, \
        train_anomalous_data, \
        test_anomalous_data, \
        threshold, \
        train_MSe = da.get_tarin_MSE_and_threshold(model, train_x, test_x)

        print(f'Average MSE ({target} {resource_name}): {np.mean(train_MSe)}')
        print(
            f'Number of anomaly samples ({target} {resource_name}) :{np.sum(test_anomalous_data)} out of {len(test_anomalous_data)}')
        print(f'Indices of anomaly samples ({target} {resource_name}): {np.where(test_anomalous_data)}')

        return train_MSe, test_MSE_loss, train_anomalous_data, test_anomalous_data, threshold

    # start training
    def start_training(self, rsc_name=""):
        self.ready_to_start(CPU)
        # TODO: uncomment when multi resources
        # threading.Thread(target=self.ready_to_start, args=(CPU,)).start()
        # threading.Thread(target=self.ready_to_start, args=(MEMORY,)).start()
        # threading.Thread(target=self.ready_to_start, args=(NETWORK,)).start()
        # threading.Thread(target=self.ready_to_start, args=(DISK,)).start()

    def init(self):
        # NOTE(gRPC Python Team): .close() is possible on a channel and should be
        # used in circumstances in which the with statement does not fit the needs
        # of the code.
        self.register_client()

        # TODO activate heartbeat
        # threading.Thread(target=self.send_heartbeat, daemon=True).start()
        # threading.Thread(target=client.get_global_model).start() # just waits for any data from server
        print("-------------- Ready --------------")

    def get_data(self):
        pass


from tensorflow.keras.callbacks import EarlyStopping


def keras_lstm_training(X, y, epochs, model, batch_size=64, verbose=1):
    # fit model

    print("Keras fit function ", batch_size)
    print("x, y shapes: ", X.shape, y.shape)
    my_callbacks = [
        EarlyStopping(monitor='loss', patience=4),
        #     (monitor="val_loss"
        # TensorBoard(log_dir='./logs'),
    ]
    model.fit(X, y, epochs=epochs,
              batch_size=batch_size,
              #     validation_split=0.1, # TODO: replace with a vildation data
              verbose=verbose,
              callbacks=my_callbacks,
              shuffle=False)

def get_formatted_elapsed_time(t):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - t))
