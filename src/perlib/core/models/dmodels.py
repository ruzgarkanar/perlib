import tensorflow as tf
from ..optimizers.Optimizers import optimizer
from tcn import TCN
from ..req_utils import *

class models():
    def __init__(self,
                 req_info,
                 pool_size = 1,
                 kernel_size = 3,
                 filters = 64,
                 show=False
                 ):

        self.req_info    = req_info
        self.input_shape = (24,1)
        self.model_multi = tf.keras.models.Sequential()
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.show = show
        if self.req_info.targetCol is None:
            check(False)
        check_period(self.req_info.period)
        if bool(self.req_info.metric):
            evaluate(self.req_info.metric)
        if self.req_info.modelname == "lstnet":
            check_D_modelname(self.req_info.layers,self.req_info.modelname)
        check_layer(self.req_info.layers,self.req_info)
        check_scaler(self.req_info.scaler)


    def layers(self,unit:int,activation:str,return_sequences=True,
               input_shape=None,filters=None,kernel_size = None):

        if self.req_info.modelname.lower() == "lstm":
            return tf.keras.layers.LSTM(units=unit,input_shape=self.input_shape,
                                        activation=activation,
                                        return_sequences=return_sequences)
        elif self.req_info.modelname.lower() == "bilstm":
            return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=unit,input_shape=self.input_shape,
                                        activation=activation,
                                        return_sequences=return_sequences))
        elif self.req_info.modelname.lower() == "convlstm":
             return tf.keras.layers.Conv1D(filters=self.filters,
                                       kernel_size= self.kernel_size,
                                       input_shape=self.input_shape,
                                       activation=activation)
        elif self.req_info.modelname.lower() == "tcn":
            return TCN(unit,input_shape=self.input_shape,
                                        activation=activation,
                                        return_sequences=return_sequences)
        elif self.req_info.modelname.lower() == "rnn":
            return tf.keras.layers.SimpleRNN(units=unit, input_shape=self.input_shape,
                                        activation=activation,
                                        return_sequences=return_sequences)
    def layer(self, tuple, index, len):
        if index == 0:
            self.model_multi.add(
                self.layers(tuple[0],kernel_size=self.kernel_size,filters=self.filters, activation=tuple[1])),
            if self.req_info.modelname.lower() == "convlstm":
                self.model_multi.add(tf.keras.layers.MaxPool1D(pool_size=1))
            self.model_multi.add(tf.keras.layers.Dropout(tuple[2])),
        elif index == len - 1:
            if self.req_info.modelname.lower() == "convlstm":
                self.model_multi.add(tf.keras.layers.LSTM(tuple[0], activation=tuple[1], return_sequences=False)),
            else:
                self.model_multi.add(self.layers(tuple[0], activation=tuple[1],input_shape=None,filters=None,
                                                         kernel_size=None,
                                                         return_sequences=False)),
            self.model_multi.add(tf.keras.layers.Dropout(tuple[2])),
            self.model_multi.add(tf.keras.layers.Dense(units=1)),
            self.model_multi.compile(optimizer=optimizer.Adam(), loss='mse')
        else:
            if self.req_info.modelname.lower() == "convlstm":
                self.model_multi.add(tf.keras.layers.LSTM(tuple[0], activation=tuple[1], return_sequences=True)),
            else:
                self.model_multi.add(self.layers(tuple[0],kernel_size=None,filters=None, activation=tuple[1],input_shape=None)),
            self.model_multi.add(tf.keras.layers.Dropout(tuple[2])),
        return self.model_multi

    def set_inputShape(self,input_shape):
        self.input_shape = input_shape

    def build_model(self):
        units = self.req_info.layers["Layer"]["unit"]
        activations = self.req_info.layers["Layer"]["activation"]
        dropouts = self.req_info.layers["Layer"]["dropout"]
        for i in range(len(self.req_info.layers["Layer"]["unit"])):
            self.model_multi = self.layer([list(zip(units, activations, dropouts))][0][i], i, len(self.req_info.layers["Layer"]["unit"]))
        if self.show:
            try:
                self.model_multi.summary()
            except:pass
        return self.model_multi




    #def build_model(self):
    #    lstm_multi = tf.keras.models.Sequential()
    #    lstm_multi.add(tf.keras.layers.LSTM(150, input_shape=self.input_shape, return_sequences=True))
    #    lstm_multi.add(tf.keras.layers.Dropout(0.5)),
    #    lstm_multi.add(tf.keras.layers.LSTM(units=100, return_sequences=False)),
    #    lstm_multi.add(tf.keras.layers.Dropout(0.5)),
    #    lstm_multi.add(tf.keras.layers.Dense(units=1)),
    #    lstm_multi.compile(optimizer=optimizer.Adam(), loss='mse')
    #    lstm_multi.summary()
    #    return lstm_multi