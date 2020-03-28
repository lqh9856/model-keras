#!/usr/bin/env python
# coding: utf-8



from keras.layers import Conv2D, MaxPooling2D, Input, TimeDistributed, ConvLSTM2D, UpSampling2D, Add
from keras.models import Sequential, Model
import keras.backend as K
from keras.utils.data_utils import get_file
from config import *
from keras.layers.convolutional_recurrent import ConvLSTM2D

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def lr_schedule(epoch):
    lr = [1e-4, 1e-4,1e-4, 1e-4, 1e-4,1e-4,1e-4, 1e-4,1e-4,1e-4, 1e-4,1e-4,1e-5, 1e-5,1e-5, 1e-5, 1e-5,1e-5, 1e-5, 1e-5,1e-5, 1e-6, 1e-6,1e-6, 1e-6, 1e-6,1e-6,1e-6, 1e-6,1e-6,1e-7, 1e-7,1e-7,1e-7, 1e-7,1e-7,1e-7, 1e-7,1e-7,1e-7, 1e-7,1e-7,1e-7, 1e-7,1e-7]
    return lr[epoch]


def kl_divergence(y_true, y_pred):
    y_true_sum = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))
    y_pred_sum = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))

    y_true /= (y_true_sum + K.epsilon())
    y_pred /= (y_pred_sum + K.epsilon())
    
    return 10 * K.sum(y_pred * K.log(y_pred / (y_true + K.epsilon())))



def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_pred = y_pred / max_y_pred
    
    y_true_sum = K.expand_dims(K.repeat_elements(K.expand_dims(
        K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), rep=y_pred.shape[2], axis=2)), rep=y_pred.shape[3], axis=3))
    y_pred_sum = K.expand_dims(K.repeat_elements(K.expand_dims(
        K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), rep=y_pred.shape[2], axis=2)), rep=y_pred.shape[3], axis=3))

    y_true = y_true / (y_true_sum + K.epsilon())
    y_pred = y_pred / (y_pred_sum + K.epsilon())
    
#     cov(X,Y) = E(XY) - EXEY
    y_true_pred = K.mean(y_pred * y_true, axis=[2, 3, 4])
    y_true_mean = K.mean(y_true, axis=[2, 3, 4])
    y_pred_mean = K.mean(y_pred, axis=[2, 3, 4])
    cov_true_pred = y_true_pred - y_true_mean * y_pred_mean
    
#     varience of X = X.^2 - (X)^2
    y_true_var = K.mean(K.square(y_true), axis=[2, 3, 4]) - K.sqrt(y_true_mean) + K.epsilon()
    y_pred_var = K.mean(K.square(y_pred), axis=[2, 3, 4]) - K.sqrt(y_pred_mean) + K.epsilon()

    return -K.sum(cov_true_pred / (y_true_var * y_pred_var))



def crossNet():
    inputs = Input(shape=(img_height, img_width, 3))
    X = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    X = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(X)
    M1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(X) # 112*112*64

    X = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(M1)
    X = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(X)
    M2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(X) # 56*56*128

  
    X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(M2)
    X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(X)
    X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(X)
    M3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(X) # 28*28*256

  
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(M3)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(X)
    M4 = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', padding='same')(X) # 28*28*512

    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(M4)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(X) # 28*28*512  
    
    
    M3_E = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(M3)# 28*28*512 
    outputs = Add()([M3_E, M4, X])
    outputs = Conv2D(256, (3, 3), activation='relu', padding='same', name='block6_conv2')(outputs)
    outputs = Conv2D(256, (3, 3), activation='relu', padding='same', name='block6_conv3')(outputs)
    outputs = Conv2D(256, (3, 3), activation='relu', padding='same', name='block6_conv4')(outputs)
    outputs = UpSampling2D(size=(2, 2))(outputs)# 56*56*256 
    
    model = Model(inputs=inputs, outputs=outputs)
#     Load weights
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)

    return model


def crossLSTM(X, stateful=False):
    cross_net = crossNet()
    outs = TimeDistributed(cross_net)(X)
    
    outs = (ConvLSTM2D(filters=256, kernel_size=(3, 3),padding='same', return_sequences=True, stateful=stateful, dropout=0.4))(outs)
    outs = TimeDistributed(Conv2D(128, (3, 3), activation='sigmoid', padding='same'))(outs)
    outs = TimeDistributed(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))(outs)
    outs = TimeDistributed(UpSampling2D(size=(4, 4)))(outs)
    return [outs, outs]






