#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:55:49 2017

@author: seshads1
"""
import scipy.io
import numpy as np
#import ipdb
from IPython.core.debugger import Tracer
import os
import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.constraints import nonneg
from keras.callbacks import Callback
from keras.constraints import maxnorm
from keras import regularizers


def trainining_DNN(x_train, y_train,model):

    batch_size = 100
    epochs = 30  # 20 #500
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
    #sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['mse'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks = [earlyStopping])

    return model

modPath = sys.argv[1]
str(modPath)

matFile = modPath + '/pyTrain.mat'
x = scipy.io.loadmat(matFile)
X = x['DNN_X']
Y = x['DNN_Y']
dnnModel =  x['oldModPath']
x = x['DNN_x']

DNNmodel = keras.models.load_model(dnnModel[0])
DNNmodel = trainining_DNN(X,Y,DNNmodel)

y = DNNmodel.predict(x)
savePath = modPath + '/pyPredTest.mat' 
scipy.io.savemat(savePath, mdict={'y': y})


    
