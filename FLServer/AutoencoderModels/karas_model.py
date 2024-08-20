import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

def keras_lstm_training(X, y ,epochs, model, batch_size= 64, verbose=1 ):
    # fit model
    print("Keras fit function",  batch_size)
    my_callbacks = [
        EarlyStopping(monitor='loss', patience=5),
        #     (monitor="val_loss"
        # TensorBoard(log_dir='./logs'),
    ]
    model.fit(X, y,  epochs=epochs,
        batch_size=batch_size,
        #     validation_split=0.1, # TODO: replace with a vildation data
        verbose=verbose,
        # callbacks=my_callbacks,
        shuffle=False)



#TODO: HAS nothing to do with anything
class GetWeights(Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            print('Layer %s has weights of shape %s and biases of shape %s' %(
                layer_i, np.shape(w), np.shape(b)))

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['w_'+str(layer_i+1)], w))
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['b_'+str(layer_i+1)], b))