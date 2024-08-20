from threading import Thread

import lstm_objects
from DB_Splitter.data_operations import load_pickle_file
rsc="cpu"
dir = "/WriteAbleData/One_Day_Data_processed/merged_output_Nov20_22/physical/"

def check_dataset_sizes(i, rsc):
    print("loading data")
    lstm_obj_file = load_pickle_file(f"{dir}/day{i}/{rsc}")
    dataset = lstm_obj_file.X_train_x
    print(f"{rsc} in day_{i}: {dataset.shape}")

if __name__ == "__main__":
    for i in range(1, 14+1):
        Thread(target= check_dataset_sizes, args=(i, rsc)).start()