import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from matplotlib.pyplot import imshow
import matplotlib.cm as cm
import matplotlib.pylab as plt
import h5py
import gc
import numpy as np
import PIL
from PIL import Image
from PIL import ImageFilter
import tensorflow as tf
import cv2
import itertools
import random
import keras
import imutils
from imutils import paths
import os
from tensorflow.keras import optimizers
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
from keras import callbacks
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from keras import backend as K

def extractor(string):
    lis = string.split('.')[-2].split('/')[-1]
    return lis

def read_hpf5_files(train_data_file,train_lbl_file,val_data_file,val_lbl_file):
    
    with h5py.File(train_data_file, 'r') as hf:
        trainX = hf[extractor(train_data_file)][:]
    with h5py.File(train_lbl_file, 'r') as hf:
        trainY = hf[extractor(train_lbl_file)][:]
    with h5py.File(val_data_file, 'r') as hf:
        valX = hf[extractor(val_data_file)][:]
    with h5py.File(val_lbl_file, 'r') as hf:
        valY = hf[extractor(val_lbl_file)][:]
        
    return trainX,trainY,valX,valY


# put paths here

trainX,trainY,valX,valY = read_hpf5_files('/scratch/train_inputs.hdf5','/scratch/train_labels.hdf5','/scratch/val_inputs.hdf5','/scratch/val_labels.hdf5')

trainX = np.array(trainX)
trainY = np.array(trainY)
valX = np.array(valX)
valY = np.array(valY)

trainX = np.clip(trainX,0,1)
trainY = np.clip(trainY,0,1)
valX = np.clip(valX,0,1)
valY = np.clip(valY,0,1)

print(trainX.shape,trainY.shape,valX.shape,valY.shape)


# Sanity checks
counting=0
for i in valY:
    if(i[0]==0 and i[1]==0 and i[2]==0 and i[3]==0):
        counting+=1
print(counting)

for ind in range(trainY.shape[0]):
    fg=0
    for i in range(3):
        if(trainY[ind][i]==1):
            fg=1
        if fg==1:
            break
    if fg==0:
        print(trainY[ind])

print("Cool")

###

for ind in range(valY.shape[0]):
    fg=0
    for i in range(3):
        if(valY[ind][i]==1):
            fg=1
        if fg==1:
            break
    if fg==0:
        print(valY[ind])
        print("EROOORRR!!!")

K.set_image_data_format('channels_last')


    
def create_model():
  model=Sequential()

  # Cu Layers 
  model.add(Conv2D(filters=64,activation='relu',strides=(1,1),kernel_size=(3,3),name='conv_layer1',input_shape=(50,50,1)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(filters=128,activation='relu',strides=1,kernel_size=(2,2),padding='same',name='conv_layer2'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  #Cs Layers
  model.add(Conv2D(256, kernel_size=(2, 2), activation='relu',padding='same'))
  model.add(Conv2D(256, kernel_size=(2, 2), activation='relu' , padding = 'same'))


  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(2383,activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(4, activation='sigmoid'))

  return model

batch_size = 400 # batch size
epochs = 10 # epochs

model= create_model()


print(model.summary())

sgd = optimizers.SGD(lr=0.01, decay=5e-5, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

filepath="/scratch/check_deep_net"

#checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_freq='epoch',mode='min')
reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, min_lr=1e-4)


wandb.init(
    # set the wandb project where this run will be logged
    project="Autoencoder",
    name="deepfont",

    # track hyperparameters and run metadata with wandb.config
    config={
        "optimizer": "sgd",
        "loss": "bceloss",
        "metric": "accuracy",
        "epoch": 10,
        "batch_size": 400,
	"set_layer_train":"false"
    }
)

cnt=0

def func():
  model.save(f"/scratch/refinedmodel-tranfalse{cnt}.h5")
  cnt+=1
  print("MODEL SAVED !!!!")


checkpoint = keras.callbacks.ModelCheckpoint('/scratch/model{epoch:08d}.h5', period=1) 
callbacks_list = [early_stopping,checkpoint,reduce_lr_callback,WandbMetricsLogger(log_freq=batch_size)]

model.fit(trainX, trainY,shuffle=True,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valX, valY),validation_batch_size=400,callbacks=callbacks_list)

score = model.evaluate(valX, valY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#model.save("/scratch/refinedmodel-tranfalse.h5")

wandb.finish()
