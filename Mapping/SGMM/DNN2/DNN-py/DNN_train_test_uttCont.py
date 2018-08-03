# -*- coding: utf-8 -*-

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:26:06 2017

@author: seshads1
"""

import scipy.io
import numpy as np
#import ipdb
from IPython.core.debugger import Tracer
import os

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.constraints import nonneg
from keras.callbacks import Callback


def trainining_DNN_cont(x_train, y_train,DNNmodel):


    batch_size = 32
    epochs = 250  # 20 #500
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    DNNmodel.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['mse'])
    DNNmodel.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks = [earlyStopping])

    return DNNmodel        
    
currPath =  '/Users/seshads1/Documents/code/speakStyleConv/ene-end-system-SS2'
loadFile = currPath + '/mapping/DNN2/DNN-py/tmp.mat' 
x = scipy.io.loadmat(loadFile)#, variable_names='F0',)
train_x = x['train_x']
train_y = x['train_y']
test = x['test']
nameOldModel = x['nameOldModel']         
#nameOldModel[0] = '/Users/seshads1/Documents/code/speakStyleConv/ene-end-system-SS2/mapping/DNN/DNN-modFiles/Straightm10F0100DNN_model.h5'
DNNmodel = keras.models.load_model(nameOldModel[0])
#predO = DNNmodel.predict(test)
#predExpO = np.exp(predO[:,0])

DNNmodel = trainining_DNN_cont(train_x,train_y,DNNmodel)

pred = DNNmodel.predict(test)
#predExpA = np.exp(predA[:,0])
matFile = currPath + '/mapping/DNN2/DNN-py/pred.mat'
scipy.io.savemat(matFile, mdict={'pred': pred})
