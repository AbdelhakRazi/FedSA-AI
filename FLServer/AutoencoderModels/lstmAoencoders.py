#import tensorflow

import tensorflow.keras as keras
#from keras.layers.core import Dropout
#from keras.layers import Dropout
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout
#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import Sequential

#TODO: Problem using optmizer object

class KerasLSTMModel_M0:

    def __new__(cls, n_units = 128, feature_size=0, lookback=2, lr=0.001):
        return cls.model(n_units, feature_size, lookback,lr)

    def model(n_units, feature_size, lookback, lr):
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(
            LSTM(n_units, activation='elu', input_shape=(lookback, feature_size), return_sequences=True,
                 name='encode1'))
        lstm_autoencoder.add(Dropout(0.2, name='dropout_encode_1'))
        lstm_autoencoder.add(LSTM(n_units, activation='elu', name='encode2', return_sequences=False))
        lstm_autoencoder.add(RepeatVector(lookback))
        lstm_autoencoder.add(LSTM(n_units, activation='elu', return_sequences=True, name='dencode1'))
        lstm_autoencoder.add(Dropout(0.2, name='dropout_dencode_1'))
        lstm_autoencoder.add(LSTM(feature_size, activation='linear', return_sequences=True, name='dencode2'))
        # lstm_autoencoder.summary()
        optimizer = keras.optimizers.Adam(lr=lr)
        lstm_autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        return lstm_autoencoder

class KerasLSTMModel_M1:

    def __new__(cls, n_units = 128, feature_size=0, lookback=2, lr=0.001):
        return cls.model(n_units, feature_size, lookback,lr)

    def model(u_units, feature_size, lookback,lr):
        model_ = Sequential([
            LSTM(u_units, input_shape=(lookback, feature_size)),
            Dropout(0.2),
            RepeatVector(lookback),
            LSTM(u_units, return_sequences=True),
            Dropout(0.2),
            TimeDistributed(Dense(feature_size))])

        # model_.summary()
        optimizer = keras.optimizers.Adam(lr=lr)
        model_.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])

        return model_


class KerasLSTMModel_M2:

    def __new__(cls, n_units = 128, feature_size=0, lookback=2,lr=0.001):
        return cls.model(n_units, feature_size, lookback,lr)

    def model(n_units, feature_size, lookback, lr):
        print(n_units)

        model_ = keras.Sequential()
        model_.add(keras.layers.LSTM(
            units=n_units,
            input_shape=(lookback, feature_size)
        ))
        model_.add(keras.layers.Dropout(rate=0.2))
        model_.add(keras.layers.RepeatVector(n=lookback))
        model_.add(keras.layers.LSTM(units=n_units, return_sequences=True))
        model_.add(keras.layers.Dropout(rate=0.2))
        model_.add(keras.layers.TimeDistributed(keras.layers.Dense(units=feature_size)))
        optimizer = keras.optimizers.Adam(lr=lr)
        model_.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])

        # model_.summary()

        return model_


class KerasLSTMModel_M3:
    def __new__(cls, n_units=128, feature_size=0, lookback=2,lr =0.001):
        return cls.model(n_units, feature_size, lookback, lr)

    def model(n_units, feature_size, lookback, lr):
        n_unit = n_units
        latent_unit = n_unit/2
        print(n_units)
        # define model
        model = Sequential()
        model.add(LSTM(n_units, activation='relu', input_shape=(feature_size, lookback), return_sequences=True))
        model.add(LSTM(latent_unit, activation='relu', return_sequences=False))
        model.add(RepeatVector(lookback))
        model.add(LSTM(latent_unit, activation='relu', return_sequences=True))
        model.add(LSTM(n_units, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(feature_size)))
        optimizer = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=optimizer, loss='mse')
        #model.summary()

        return model



class KerasLConvModel:

    def __new__(cls, feature_size=0, lookback=2):
        return cls.model(feature_size, lookback)

    def model(feature_size, lookback):

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(lookback, feature_size)),
                keras.layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                keras.layers.Dropout(rate=0.2),
                keras.layers.Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                keras.layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                keras.layers.Dropout(rate=0.2),
                keras.layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                keras.layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        #model.summary()


class KerasLSTMModel_M4:
    def __new__(cls, n_units=128, feature_size=0, lookback=2, lr=0.001):
        return cls.model(n_units, feature_size, lookback, lr)

    def model(n_units, feature_size, lookback, lr):

        hidden_layer_size = int(feature_size * 0.8)
        print(hidden_layer_size)
        hidden_layer_size2 = int(feature_size * 0.6)
        print(hidden_layer_size2)
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(
            LSTM(hidden_layer_size, activation='elu', input_shape=(lookback, feature_size), return_sequences=True,
                 name='encode1'))
        lstm_autoencoder.add(Dropout(0.2, name='dropout_encode_1'))
        lstm_autoencoder.add(LSTM(hidden_layer_size2, activation='elu', name='encode2', return_sequences=False))
        lstm_autoencoder.add(RepeatVector(lookback))
        lstm_autoencoder.add(LSTM(hidden_layer_size, activation='elu', return_sequences=True, name='dencode1'))
        lstm_autoencoder.add(Dropout(0.2, name='dropout_dencode_1'))
        lstm_autoencoder.add(LSTM(feature_size, activation='linear', return_sequences=True, name='dencode2'))
        
        lstm_autoencoder.summary()
        
        optimizer = keras.optimizers.Adam(lr=lr)
        
        lstm_autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        

        
        return lstm_autoencoder



def keras_lstm_training(X, y ,epochs, model, verbose=1 ):
    # fit model
    model.fit(X, y, epochs= epochs, verbose=verbose)
