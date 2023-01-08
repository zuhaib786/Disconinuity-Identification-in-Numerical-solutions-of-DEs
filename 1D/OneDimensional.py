import numpy as np
import pandas as pd
import random
import tensorflow as tf
from keras.layers import Conv1D, Dense, Input, Flatten
class DiscontDataGenerator:
  def __init__(self, pts = 201, N_F = 15):
    self.M = 5
    self.pts = pts
    self.D = list(np.linspace(-1,1,pts))
    self.N_F = N_F
  def Fourier(self):
    data = []
    for i in range(self.N_F):
      data.append(list(np.random.normal(0, 1, 2)))
    return data
  def calc(self, data, x):
    ans = 0
    for i, tup in enumerate(data):
      a, b = tup
      ans += a * np.cos((i + 1) * x) + b * np.sin((i + 1)*x)
    return ans
  def generate(self):
    discont = np.random.randint(0,self.M )
    points = random.sample(self.D[1:-1], discont)
    points.sort()
    i = 0 
    temp , label = [], []
    data = self.Fourier()
    for j in self.D:
      if i< discont and j >= points[i]:
        label.append(1)
        i += 1
        data = self.Fourier()
      else:
        label.append(0)
      temp.append(self.calc(data, j))
    return temp, label[1:]
      



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
