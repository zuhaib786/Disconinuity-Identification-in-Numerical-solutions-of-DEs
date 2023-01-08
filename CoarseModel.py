import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import MaxPooling2D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation,Convolution2D,GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from keras.models import Model
import matplotlib.patches as patches
from keras.callbacks import ModelCheckpoint
###############################################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def createCoarseModel():
  model = Sequential()
  model.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
  model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
  model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(100, activation = 'sigmoid'))

  adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model.compile(optimizer = "adam", loss =tf.keras.losses.BinaryCrossentropy() ,
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
  return model
