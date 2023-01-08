import tensorflow as tf
from ...core.optimizers import optimizer
from keras.models import Model
from keras.layers import Input, GRU, Conv2D, Dropout, Flatten, Dense, Reshape, Concatenate, Add
from keras import backend as K

class PreSkipTrans(tf.keras.layers.Layer):
    def __init__(self, pt, skip, **kwargs):
        self.pt = pt
        self.skip = skip
        super(PreSkipTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PreSkipTrans, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        batchsize = tf.shape(x)[0]
        input_shape = K.int_shape(x)
        output = x[:, -self.pt * self.skip:, :]
        output = tf.reshape(output, [batchsize, self.pt, self.skip, input_shape[2]])
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, [batchsize * self.skip, self.pt, input_shape[2]])
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.pt
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'pt': self.pt, 'skip': self.skip}
        base_config = super(PreSkipTrans, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PostSkipTrans(tf.keras.layers.Layer):
    def __init__(self, skip, **kwargs):
        self.skip = skip
        super(PostSkipTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PostSkipTrans, self).build(input_shape)

    def call(self, inputs):
        x, original_model_input = inputs
        batchsize = tf.shape(original_model_input)[0]
        input_shape = K.int_shape(x)
        output = tf.reshape(x, [batchsize, self.skip, input_shape[1]])
        output = tf.reshape(output, [batchsize, self.skip * input_shape[1]])
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.skip * shape[1]
        return tf.TransformShape(shape)

    def get_config(self):
        config = {'skip': self.skip}
        base_config = super(PostSkipTrans, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PreARTrans(tf.keras.layers.Layer):
    def __init__(self, hw, **kwargs):
        self.hw = hw
        super(PreARTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PreARTrans, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        batchsize = tf.shape(x)[0]
        input_shape = K.int_shape(x)
        output = x[:, -self.hw:, :]
        output = tf.transpose(output, perm=[0, 2, 1])
        output = tf.reshape(output, [batchsize * input_shape[2], self.hw])
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.hw
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'hw': self.hw}
        base_config = super(PreARTrans, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PostARTrans(tf.keras.layers.Layer):
    def __init__(self, m, **kwargs):
        self.m = m
        super(PostARTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PostARTrans, self).build(input_shape)

    def call(self, inputs):
        x, original_model_input = inputs
        batchsize = tf.shape(original_model_input)[0]
        input_shape = K.int_shape(x)
        output = tf.reshape(x, [batchsize, self.m])
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.m
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'m': self.m}
        base_config = super(PostARTrans, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def LSTNetModel(input_shape,
                lookback = 24,
                CNNFilters = 100,
                CNNKernel  = 6,
                GRUUnits = 100,
                SkipGRUUnits = 5,
                skip = 24,
                dropout = 0.2,
                highway = 24,
                initializer = "glorot_uniform"
                ):

    """
    :param input_shape:
    :param lookback:     Number of time values to consider in each input X Default : 24
    :param CNNFilters:   Number of output filters in the CNN layer Default : 100 If set to 0, the CNN layer will be omitted
    :param CNNKernel:    CNN filter size that will be (CNNKernel, number of multivariate timeseries) Default : 6
    :param GRUUnits:     Number of hidden states in the GRU layer Default : 100
    :param SkipGRUUnits: Number of hidden states in the SkipGRU layer Default : 5
    :param skip:         Number of timeseries to skip. 0 => do not add Skip GRU layer Default : 24
    :param dropout:      Dropout frequency Default : 0.2
    :param highway:      Number of timeseries values to consider for the linear layer (AR layer) Default : 24
    :param initializer:
    :return:
    """
    m = input_shape[2]
    tensor_shape = input_shape[1:]

    if K.image_data_format() == 'channels_last':
        ch_axis = 3
    else:
        ch_axis = 1

    X = Input(shape=tensor_shape)

    if CNNFilters > 0 and CNNKernel > 0:
        C = Reshape((input_shape[1], input_shape[2], 1))(X)
        C = Conv2D(filters=CNNFilters, kernel_size=(CNNKernel, m), kernel_initializer=initializer)(C)
        C = Dropout(dropout)(C)
        c_shape = K.int_shape(C)
        C = Reshape((c_shape[1], c_shape[3]))(C)
    else:
        C = X

    _, R = GRU(GRUUnits, activation="relu", return_sequences=False, return_state=True)(C)
    R = Dropout(dropout)(R)

    if skip > 0:
        pt = int(lookback / skip)
        S = PreSkipTrans(pt, int((lookback - CNNKernel + 1) / pt))(C)
        _, S = GRU(SkipGRUUnits, activation="relu", return_sequences=False, return_state=True)(S)
        S = PostSkipTrans(int((lookback - CNNKernel + 1) / pt))([S, X])
        R = Concatenate(axis=1)([R, S])

    Y = Flatten()(R)
    Y = Dense(m)(Y)

    # AR
    if highway > 0:
        Z = PreARTrans(highway)(X)
        Z = Flatten()(Z)
        Z = Dense(1)(Z)
        Z = PostARTrans(m)([Z, X])
        Y = Add()([Y, Z])

    model = Model(inputs=X, outputs=Y)
    model.compile(optimizer=optimizer.Adam(), loss="mse")
    model.summary()
    return model


#def ModelCompile(model,lr = 0.001, optimiser = "Adam"):
#    # Select the appropriate optimiser and set the learning rate from input values (arguments)
#    if optimiser == "SGD":
#        opt = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
#    elif optimiser == "RMSprop":
#        opt = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
#    else: # Adam
#    	opt  = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
