import os
import sys
import time
import numpy as np
from tensorflow import keras
from pathlib import Path
ROOT_DIR = os.getcwd()
ROOT_DIR = ROOT_DIR.split("/")
ROOT_DIR = ROOT_DIR[:len(ROOT_DIR) - 1]
ROOT_DIR = "/".join(ROOT_DIR)
sys.path.insert(0, f'{ROOT_DIR}/Utils')
import utils
import json
from os.path import exists


sys.path.insert(0, f'{ROOT_DIR}/FLClients')
from ProducerConsumer import Producer
from ProducerConsumer import Consumer

from numpy import loadtxt
from tensorflow.keras.models import load_model


connect_to_data_pipeline = False
target_model = f'{ROOT_DIR}/Output/Models/client/current_global_model.h5'
target_runtime_analysis_file = f'{ROOT_DIR}/Output/Analysis/physical/runtime_analysis.json'
log_file_name = 'anomaly_logs.csv'
signature_file_name = 'signature_logs.csv'


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

def flatten(data):
    flat_data = []
    for i in range(len(data)):
        flat_data.append(data[i, 0,])
    return np.stack(flat_data)

def get_real_time_training_data(data_pipeline_parameter, producer_sleep_time_interval = 0.2):
    
    global connect_to_data_pipeline
    if not connect_to_data_pipeline:
        print("Real-time data arrives")
        producer = Producer(data_pipeline_parameter, producer_sleep_time_interval)
        producer.start()
        connect_to_data_pipeline = True
    consumer = Consumer()
    consumer.start()
    consumer.join()

    # nbr_features = 15
    data_dict = {}
    data = consumer.get_training_data()
    timestamp_arrival = str(time.time())
    for d in data:
        for key, values in d.items():
            metrics = values['metrics']
            nbr_features = len(metrics)
            
            source_id = values['source_id']
            timestamp_col = values['timestamp_col']
            timestamp_pub = values['timestamp_pub']
            timestamp_prep_start = values['timestamp_prep_start']
            timestamp_prep_end = values['timestamp_prep_end']
            column_names=['Window training time', 'Source ID', 'Collection time', 'Publishing time', 'Preprocessing time start', 'Preprocessing time end', 'Data arrival time']
            column_values = [key,source_id,timestamp_col,timestamp_pub,timestamp_prep_start,timestamp_prep_end,timestamp_arrival]
            csv_file_name = 'infere_timestamp_values.csv'
            utils.write_to_csv_file(csv_file_name,column_names,column_values)
            source_id = values['source_id']
            if key in data_dict:
                # Extend the metrics list for the existing key
                data_dict[key].extend(metrics)
            else:
                # Create a new entry for the key
                data_dict[key] = metrics

        for source_id, values in data_dict.items():
            training_data = np.array(values)
            # Retransform to required shape (#samples,loockback,#features)
            #training_data = np.array(training_data).reshape(-1, lookback, nbr_features)

    print("Get data from Data Pipeline System")
    return training_data

    
    
def rt_inference(model, data, threshold_value):

        mse, flat_data = get_mse(model=model, infrence_data=data)
        is_anomaly = mse > threshold_value
        # as the value is an array with one element, use  index 0 to get it
        is_anomaly = is_anomaly[0]
        mse=mse[0]
        return mse, is_anomaly

        
    
if __name__ == '__main__':
    
    
    
    if len(sys.argv) < 3:
        self_file = sys.argv[0]
        print("Usage: python3 "+self_file+" eNDBF_server:port_number consumer_topic_name")
        sys.exit(1)


    threshold_value = 0
        
    bootstrap_server = str(sys.argv[1])
    topic_name = str(sys.argv[2])
    
    data_pipeline_parameter_ = {
            
            "topic_name": topic_name,  # Name of the topic from where to get the data
            "bootstrap_server":bootstrap_server,  # Ip address of the borker (eNDBF)
            "consumer_group": "preprocessed_cpu_group_for_inference"
    }
    
    print("Cleaning...")
    for i in range(0,10):
        csv_file_name = f'anomaly_detection_{i}.0.csv'
        if exists(csv_file_name):
            os.remove(csv_file_name)
            print("\tRemove ", csv_file_name)
        
        
    if exists(target_runtime_analysis_file):
        os.remove(target_runtime_analysis_file)
        print("\tRemove ", target_runtime_analysis_file)
    
    if exists(target_model):
        os.remove(target_model)
        print("\tRemove ", target_model)
            
        
    round_no = 0
    model_version = 0
    model = None
    while True:
            
        if exists(target_runtime_analysis_file) == False:
            continue
        
        f = json.load(open(target_runtime_analysis_file)) 
        #print(f)    
        if round_no != int(f["round_no"]):
            threshold_value = f["threshold"]
            round_no = float(f["round_no"])
        else:
            if threshold_value == 0:
                threshold_value = f["threshold"]
                round_no = float(f["round_no"])
                
           
        if exists(target_model):
            model = get_keras_model(Path(target_model))
            os.remove(target_model)
            model_version = model_version + 1
            
        elif model_version == 0:
            continue 
                    
        real_time_training_data =  get_real_time_training_data(data_pipeline_parameter_)    #self.get_segment_data(training_data, resource_name)
        mse, is_anomaly = rt_inference(model, real_time_training_data, threshold_value)
        column_names=['MODEL VERSION','MSE THRESHOLD', 'DETECTED MSE', 'IS ANOMALY']
        column_values = [model_version,threshold_value, mse, is_anomaly]
        csv_file_name = f'anomaly_detection_{round_no}.csv'
        utils.write_to_csv_file(csv_file_name,column_names,column_values)
           
            
        
    