import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import MaxPooling2D,Lambda,Dense,BatchNormalization,Flatten,Reshape,Dropout,Activation,Convolution2D,GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import *
from keras.models import Model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

num=1
folder='Output/'
train_data=[]
train_label=[]
for i in range(1,num):
    temp=loadmat(folder+'image_line'+str(i)+'.mat')
    img=temp['image'] 
    img=(img-np.mean(img))/np.std(img)
    train_data.append(img.reshape(101,101,1))
    train_label.append(temp['label1'].ravel())
num=68076
# folder='/fs/project/PAS1263/data/line_curve/'
# for i in range(1,num+1):
#     temp=loadmat(folder+'image'+str(i)+'.mat')
#     img=temp['image'] 
#     img=(img-np.mean(img))/np.std(img)
#     train_data.append(img.reshape(101,101,1))
#     train_label.append(temp['label1'].ravel())

train_data=np.asarray(train_data)
train_label=np.asarray(train_label)
###############################################################################################
def root_mean_squared_error(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = Sequential()
model.add(Convolution2D(32, (4,4),strides=(1,1),input_shape=(101,101,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Convolution2D(32, (2,2),strides=(1,1),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10000))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = "adam", loss = root_mean_squared_error, metrics =["accuracy"])
filepath="Models"
filename=filepath+"/model_weights_100_100_normalization_each.h5"
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(train_data,train_label, validation_split=0.1, callbacks=callbacks_list,batch_size=1000, epochs=500, verbose=0)
