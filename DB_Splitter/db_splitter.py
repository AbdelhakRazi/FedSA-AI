import os
import time
from threading import Thread

from Colors.print_colors import print_cyne, print_orange, print_blue

import init_paths as ip
from data_operations import split_dataset_to_training_files

WAIT_BEFORE_CHECK = 5  # seconds
PATIENCE_NO_ITERATIONS = 10800  # iterations to lookup for dataset in the file system


class DataSplitter:
    def __init__(self):
        self.created_files = []

    def split_dataset(self, client_count, rsc_target, no_training_samples, instant_label, split_label):

        #for day_ in range(ip.data_splits_count + 1):
        for day_ in range(ip.data_splits_count):
            day = day_ + 1
            current_split_dir = f"{split_label}{day}"  # Daya1
            if not self.check_files_exist():
                dataset_path = f"{ip.get_train_data_dir_path(current_split_dir=current_split_dir)}/{rsc_target}"  # e.i., day1

                print("data path", dataset_path)
                self.created_files = split_dataset_to_training_files(dataset_path=dataset_path,
                                                                     data_part=day,
                                                                     client_count=client_count,
                                                                     split_files_path=ip.get_train_data_splits_path(),
                                                                     rsc=rsc_target,
                                                                     instant_label=instant_label,
                                                                     no_training_samples=no_training_samples)

    
        path = f"{ip.get_train_data_splits_path()}/{instant_label}"

        for file_name in os.listdir(path):
       	    file = path + file_name
            if os.path.isfile(file):
               print('Deleting file:', file)
               os.remove(file)
        try:
            os.rmdir(path)
        except:
            print(f"cant delete dir : {path}")
                
                
    
    
    """"
    The assumption that when a node reads its data split, will delete the split file
    Keep checking till all split files are removed
    """""

    def check_files_exist(self):
        _error = False
        i = 0
        while True:
            if i > PATIENCE_NO_ITERATIONS:
                _error =True
                break

            i += 1
            ans = True
            print("exist files ", self.created_files)
            for file in self.created_files:
                ans = ans and not os.path.exists(file)
                if not ans:
                    break

            if ans:
                break  # while loop
            else:
                time.sleep(WAIT_BEFORE_CHECK)

        return _error

    def can_split(self):

        client_count = int(ip.client_count)
        rsc_target = ip.rsc_target
        instant_label = ip.splits_instance_label
        split_label = ip.split_label  # Daya1
        no_training_samples = int(ip.no_samples)
        self.split_dataset(client_count=client_count,
                           rsc_target=rsc_target,
                           no_training_samples=no_training_samples,
                           instant_label=instant_label,
                           split_label=split_label)


if __name__ == "__main__":
    spt = DataSplitter()
    spt.can_split()
