import numpy as np

def flatten(data):
    flat_data = []
    for i in range(len(data)):
        flat_data.append(data[i, 0,])
    return np.stack(flat_data)
"""""""""""""""""""""""""""""""""""""""""
MSE¶
"""""""""""""""""""""""""""""""""""""""""


# [Train] MSE and Threshold - Train Data
def get_train_squared_error(model, train_data, n_quantile = 0.999):
    train_predicted = model.predict(train_data)
    train_flat_x = (flatten(train_data))
    pred_train_flat_x = (flatten(train_predicted))
    train_squared_error = np.square(train_flat_x - pred_train_flat_x)  # power

    return train_squared_error

def get_train_loss(model, train_data, n_quantile = 0.999):
    train_squared_error_ =  get_train_squared_error(model, train_data, n_quantile)
    train_MSE_loss = np.mean(train_squared_error_, axis=1)
    threshold = np.quantile(train_MSE_loss, n_quantile)
    # print(f'threshold = {threshold} train_squared_error={train_squared_error},train_MSE_loss={train_MSE_loss}' )
    # train_anomalous_data = train_MSE_loss > threshold

    return train_MSE_loss, threshold

# [Test ] MSE and Threshold - Test Data
def get_test_loss(model, test_data, threshhold_):
    #     train_anomalous_data , threshold_ = get_train_loss(model, train_x)
    test_predicted = model.predict(test_data)
    test_flat_x = (flatten(test_data))
    print("flatten ", test_flat_x.shape)
    pred_test_flat_x = (flatten(test_predicted))
    test_squared_error = np.square(test_flat_x - pred_test_flat_x)  # power
    test_MSE_loss = np.mean(test_squared_error, axis=1)

    test_anomalous_data = test_MSE_loss > threshhold_

    return test_MSE_loss, test_anomalous_data, threshhold_


def get_tarin_MSE_and_threshold(model, data_train_x, data_test_x):
    train_MSE, threshold_, = get_train_loss(model, data_train_x)
    #
    # test_MSE_loss, test_anomalous_data, threshold = get_test_loss(model,
    #                                                               data_test_x,
    #                                                               threshold_)

    return train_MSE, threshold_
