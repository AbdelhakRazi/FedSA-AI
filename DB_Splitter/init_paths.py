import json
import os

from Colors.print_colors import print_cyne, print_orange, print_blue

ROOT_DIR = os.getcwd()
print(ROOT_DIR)
ROOT_DIR = ROOT_DIR.split("/")
ROOT_DIR = ROOT_DIR[:len(ROOT_DIR) - 1]
ROOT_DIR = "/".join(ROOT_DIR)
print(ROOT_DIR)

# TODO in containers only: remove in normal
# ROOT_DIR="/"

fl = f'{ROOT_DIR}/FLConfig/config.json'

config = json.load(open(fl))

fed_config = config['fed_config']
file_paths = config['file_paths']
data_splits_count = config['data_splits_count']
split_label = config['split_label']  #day
no_samples = config["dataset_shape"]["no_samples"]

try:
    client_count = os.environ['k']
    rsc_target = os.environ['rsc_target']
    layer = os.environ['layer']
    splits_instance_label = os.environ['splits_instance_label']
except:
    client_count = fed_config['K']  # number of participants #TODO
print_cyne("________________ Splitter ________________")
print_orange("layer", layer, "\trsc_target = ", rsc_target)


output_dir_path = file_paths['output_dir_path']
output_dir_path = f'{ROOT_DIR}/{output_dir_path}'


def get_train_data_dir_path(current_split_dir):
    # i.e.,  current_split_dir = Day1
    _train_data_dir_path = file_paths[f'train_data_dir_path']
    return f'{ROOT_DIR}/{_train_data_dir_path}/{layer}/{current_split_dir}'  # not the file name (cpu,memory, network, disk or radio)

def get_train_data_splits_path():
    _train_data_splits_file = file_paths['train_data_splits_path']
    return f'{ROOT_DIR}/{_train_data_splits_file}'

