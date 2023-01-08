import tensorflow as tf
from keras.layers import Conv1D, Dense, Input, Flatten
def create_model(input_size=  201):
  inp = Input(shape = (input_size, 1))
  X = Conv1D(kernel_size = 2, filters = 24,activation = 'relu' )(inp)
  X = Conv1D(kernel_size = 2, filters = 24, activation = 'relu')(X)
  X = Conv1D(kernel_size = 2, filters = 24, activation = 'relu')(X)
  X = Conv1D(kernel_size = 2, filters = 24, activation = 'relu')(X)
  X = Flatten()(X)
  X = Dense(input_size- 1, activation = 'sigmoid')(X)
  model  = tf.keras.Model(inputs = inp, outputs = X)
  return model
