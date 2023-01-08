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

fineModelPath = 'Models/FineModel'
coarseModelPath = 'Models/CoarseModel'
fineModel = tf.keras.models.load_model(fineModelPath)
coarseModel = tf.keras.models.load_model(coarseModelPath)
def predict(x):
    ans = np.zeros((100, 100))
    coarse_labels = coarseModel.predict(x)
    coarse_labels = (coarse_labels>0.5).astype('float32')
    coarse_labels.reshape(x.shape[0], 10, 10)
    for n in range(x.shape[0]):
        for i in range(10):
            for j in range(10):
                if coarse_labels[n, i, j] == 1.0:
                    temp_x = x[n, 10 *i :10*(i + 1) + 1, 10*j : 10*(j + 1) + 1]
                    temp_y = fineModel.predict(temp_x)
                    temp_y = (temp_y > 0.5).astype('float32')
                    temp_y.reshape(10, 10)
                    ans[n , 10*i :10*(i + 1), 10*j : 10 *(j + 1) ] = temp_y
    return ans
