
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Convolution2D
from keras.optimizers import  Adam
def createFineModel():
  model = Sequential()
  model.add(Convolution2D(32, (2,2),strides=(1,1),input_shape=(11,11,1),activation='relu'))
  model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
  model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
  model.add(Flatten())
  model.add(Dropout(0.1))
  model.add(Dense(100))

  adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model.compile(optimizer = "adam", loss = tf.keras.losses.BinaryCrossentropy(), 
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])
  return model