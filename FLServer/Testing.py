import os
import math
import numpy as np
from  pathlib import Path
import tensorflow.keras.models as ksm
from Data_operations.DataAnalysis import get_test_loss

from Clients.Data_operations.DataOperations import load_json_file, save_pickle_to_file


def get_check_anomaly(model, data, threshold):
    test_MSE_loss, test_anomalous_data, threshold = get_test_loss(model, data, threshold)


setting_types = ['BATCH', 'ROUND','BATCH_SINGLE', 'ROUND_SINGLE']
def assigned_model_to_threshold(modl_path, path_, setting):
    model_files = [mf for mf in os.listdir(modl_path)]
    model_obj=[]

    threshold_list, threshold_dic = get_param_from_file(path_,"threshold_np", "avg",setting)
    for i, mdl in enumerate(model_files):
        model = ksm.load_model(f'{modl_path}/{mdl}')
        model_k_no = int(mdl.split("_")[1])
        threshold_value = threshold_dic[math.log(model_k_no, 2)]
        model_obj.append({'model': model, 'threshold': threshold_value})
    return model_obj


def get_param_from_file(json_file_name, param_name, param_fun, setting):
    tag = 'analysis_data'
    # ['BATCH_SINGLE', 'ROUND_SINGLE']:

    jsn = load_json_file(f"{json_file_name}")
    sett_ = jsn[setting]

    param_arr = []
    param_2Darr = {}

    for k in sett_.keys():
        param_arr.append(sett_[k][param_name][param_fun])
        param_2Darr[int(k)]= sett_[k][param_name][param_fun]

    return param_arr, param_2Darr



if __name__ == "__main__":
    hm_dr = "/home/salah/MY_PROGRAMMING/Dockers/FLDocker_5G/Results/"
    model_path = {"BATCH":f"{hm_dr}Trained_models/BATCH",
                  "ROUND":f"{hm_dr}Trained_models/ROUND"}
    param_file =f"{hm_dr}training/training_batches_ALL.json"
    settings =""
    f_name = "trained_model_with_threshold"
    for key in model_path.keys():
       model_obj = assigned_model_to_threshold(model_path[key], param_file, key)
       save_pickle_to_file(model_obj, f"{hm_dr}models_with_thresholds/{f_name}_{key}")





