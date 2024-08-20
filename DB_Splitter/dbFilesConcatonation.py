import data_operations as h
import time
import numpy as np
import lstm_objects

main_path = "/sharing_s154/k8s/Datasets"
data_path = f"{main_path}"


if __name__ == '__main__':


    t = time.time()
    store_path = "/sharing_s154/k8s/Datasets"

    for layer in ["physical", "container"]:
        for rsc in ['cpu', 'memory']:
            all_data = np.array([])
            for day in range(1, 15):
                print(f'{data_path}/{layer}/day{day}/{rsc}')

                data = h.load_pickle_file(f'{data_path}/{layer}/day{day}/{rsc}')
                fdata = h.flatten(data.X_train_x)
                print(day, "data loadded and flattened suc..", fdata.shape, all_data.shape)

                if day == 1:
                    all_data = fdata
                else:
                    all_data = np.concatenate((all_data, fdata))

            ls = lstm_objects(X_train_x=all_data, oderedColumn_x=[], scaler_x=[], feature_size_x=[])

            h.save_pickle_to_file(ls, f"{store_path}/{layer}/All/{rsc}")
            print(f"\t{store_path}/{layer}/All/{rsc}")

    tt = h.get_formatted_elapsed_time(t)
    print(tt)

