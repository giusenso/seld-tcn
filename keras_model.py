#
# The SELDnet architecture
#

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed


# SELD-TCN MODEL :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_seldtcn_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size, fnn_size, weights):

    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    spec_cnn = spec_start

    # CONVOLUTIONAL LAYERS =========================================================
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    resblock_input = Reshape((data_in[-2], -1))(spec_cnn)

    # TCN layer ===================================================================

    # residual blocks ------------------------
    skip_connections = []

    for d in range(10):

        # 1D convolution
        spec_conv1d = keras.layers.Convolution1D(filters=256,
                                                kernel_size=(3),
                                                padding='same',
                                                dilation_rate=2**d)(resblock_input)
        spec_conv1d = BatchNormalization()(spec_conv1d)

        # activations
        tanh_out = keras.layers.Activation('tanh')(spec_conv1d)
        sigm_out = keras.layers.Activation('sigmoid')(spec_conv1d)
        spec_act = keras.layers.Multiply()([tanh_out, sigm_out])

        # spatial dropout
        spec_drop = keras.layers.SpatialDropout1D(rate=0.5)(spec_act)

        # 1D convolution
        skip_output = keras.layers.Convolution1D(filters=128,
                                                 kernel_size=(1),
                                                 padding='same')(spec_drop)

        res_output = keras.layers.Add()([resblock_input, skip_output])

        if skip_output is not None:
            skip_connections.append(skip_output)

        resblock_input = res_output
    # ---------------------------------------

    # Residual blocks sum
    spec_sum = keras.layers.Add()(skip_connections)
    spec_sum = keras.layers.Activation('relu')(spec_sum)

    # 1D convolution
    spec_conv1d_2 = keras.layers.Convolution1D(filters=128,
                                          kernel_size=(1),
                                          padding='same')(spec_sum)
    spec_conv1d_2 = keras.layers.Activation('relu')(spec_conv1d_2)

    # 1D convolution
    spec_tcn = keras.layers.Convolution1D(filters=128,
                                          kernel_size=(1),
                                          padding='same')(spec_conv1d_2)
    spec_tcn = keras.layers.Activation('tanh')(spec_tcn)

    # SED ==================================================================
    sed = spec_tcn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    # DOA ==================================================================
    doa = spec_tcn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    model.summary()
    return model