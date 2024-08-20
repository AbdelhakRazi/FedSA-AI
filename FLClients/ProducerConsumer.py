# Python script to connect to broker, consume data and store it in a buffer

import threading
import time
import queue
from kafka import KafkaConsumer
import json 
import sys
import ast
import csv
import datetime
import ntplib
import numpy as np
 
# Shared Memory variables
WINDOW_SIZE = 5 # Number of samples to be collected
buffer = queue.Queue()
CAPACITY=10
SECOND_TO_MILLISECOND=1000

# Declaring Semaphores
mutex = threading.Semaphore()
empty = threading.Semaphore(CAPACITY)
full = threading.Semaphore(0)
 
# Producer Thread Class
class Producer(threading.Thread):
    should_stop_c = False
    
    def __init__(self, kafka_parameter,sleepTime):
        super(Producer,self).__init__()
        self.kafka_parameter =kafka_parameter
        self.sleepTime =sleepTime
    
    def run(self):
     
        global CAPACITY, buffer, in_index, out_index
        global mutex, empty, full
    
        print("Connect to DATA PIPELINE SYSTEM", self.kafka_parameter["bootstrap_server"])
        kafka_consumer = KafkaConsumer( 
        bootstrap_servers=[self.kafka_parameter["bootstrap_server"]],
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=self.kafka_parameter["consumer_group"]
        #key_deserializer=lambda x: json.loads(x.decode("utf-8")),
        #value_deserializer=lambda x: json.loads(x.decode("utf-8"))
        )
    
        kafka_consumer.subscribe([self.kafka_parameter["topic_name"]])
        items_consumed_from_kafka = 0
        for sample in kafka_consumer:
            
            dict_sample_str = sample.value.decode("UTF-8")
            sample_str = ast.literal_eval(dict_sample_str)
            empty.acquire()
            mutex.acquire()
            buffer.put(sample_str)
            mutex.release()
            full.release()
            items_consumed_from_kafka += 1
            if items_consumed_from_kafka == WINDOW_SIZE:
                items_consumed_from_kafka = 0
                if self.should_stop_c == True:
                    time.sleep(self.sleepTime/SECOND_TO_MILLISECOND)
               

# Consumer Thread Class
class Consumer(threading.Thread):
    
    def __init__(self):
        super(Consumer,self).__init__()
        self.training_data =[]
    
    def get_training_data(self):
        
        return self.training_data
                        
    def run(self):
        
        global CAPACITY, buffer, in_index, out_index, counter
        global mutex, empty, full
        self.should_stop = True
        print("Starting Consumer")
        full.acquire()
        mutex.acquire()
        sample_str = buffer.get()
        self.training_data.append(sample_str)
        #print("Consumer consumes ") #,sample_str)
        mutex.release()
        empty.release()
        self.should_stop = False

# Consumer Thread Class
class OnlineConsumer(threading.Thread):
    
    def __init__(self, end_to_end_time_threshold, max_training_data):
        super(OnlineConsumer,self).__init__()
        self.training_data_queue = queue.Queue()
        self.end_to_end_time_threshold = end_to_end_time_threshold
        self.timed_training_data = {}
        self.max_training_data = max_training_data
        self.training_data_in_queue = 0
    
    def get_training_data(self):
        
        training_data = self.training_data_queue.get()
        self.training_data_in_queue = self.training_data_in_queue -1
        return training_data
    
    def current_time(self):
        
        wait_for_response = True
        retry = 0
        while wait_for_response:
            try:
                ntp_client = ntplib.NTPClient()
                response = ntp_client.request('ntp.cnam.fr')
                wait_for_response = False
            except (ntplib.NTPException) as e:
                print('NTP client request error:', str(e))
                retry = retry + 1
                time.sleep(1)
            if retry == 3:
                wait_for_response = False
        return response.tx_timestamp

    def set_timed_training_data(self, training_data):
        
        training_data_dict = {}
        timestamp_arrival = str(self.current_time())
        preprocessing_time = 0
        column_names = []
        column_values = []
        to_discard = False
        for d in training_data:
            for key, values in d.items():
                metrics = values['metrics']    
                source_id = values['source_id'] 
                timestamp_col = values['timestamp_col']
                timestamp_pub = values['timestamp_pub']
                timestamp_prep_start = values['timestamp_prep_start']
                timestamp_prep_end = values['timestamp_prep_end']
                timestamp_polling_start = values['timestamp_polling_start']
                column_names=['Window training time', 'Source ID', 'Collection time', 'Publishing time', 'Polling time','Preprocessing time start', 'Preprocessing time end', 'Data arrival time']
                preprocessing_time = float(timestamp_prep_end) - float(timestamp_prep_start)
                end_to_end_time = float(timestamp_arrival) - float(timestamp_pub)
                column_values = [key,source_id,timestamp_col,timestamp_pub,timestamp_polling_start,timestamp_prep_start,timestamp_prep_end,timestamp_arrival]
                source_id = values['source_id']
                if  end_to_end_time > self.end_to_end_time_threshold:
                    to_discard = True
                if key in training_data_dict:
                    # Extend the metrics list for the existing key
                    training_data_dict[key].extend(metrics)
                else:
                    # Create a new entry for the key
                    training_data_dict[key] = metrics

        for source_id, values in training_data_dict.items():
            training_data = np.array(values)
        
        self.timed_training_data = {}   
        self.timed_training_data["timed_training"]=training_data
        self.timed_training_data["column_names"]=column_names
        self.timed_training_data["column_values"]=column_values
        self.timed_training_data["to_discard"]=to_discard
        self.timed_training_data["preprocessing_time"]=preprocessing_time
        self.timed_training_data["end_to_end_time"]=end_to_end_time
    
                        
    def run(self):
        
        while True:
            
            if self.training_data_in_queue == self.max_training_data:
                continue
            
            global CAPACITY, buffer, in_index, out_index, counter
            global mutex, empty, full
            self.should_stop = True
            full.acquire()
            mutex.acquire()
            training_data = buffer.get()
            self.set_timed_training_data([training_data])
            self.training_data_queue.put(self.timed_training_data)
            self.training_data_in_queue = self.training_data_in_queue + 1
            mutex.release()
            empty.release()
            self.should_stop = False
        
