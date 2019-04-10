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

def construct_DNN(D_ip,D_op,nL,nN,dR):
    model = Sequential()
    model.add(Dense(nN, activation='relu', use_bias='True', input_dim=D_ip))#,kernel_constraint=maxnorm(10)))#,kernel_initializer='normal',kernel_constraint=maxnorm(10)))

    for i in range(1,nL):
        model.add(Dropout(dR))
        model.add(Dense(nN, activation='relu', use_bias='True'))#,kernel_constraint=maxnorm(10)))#,kernel_initializer='normal',kernel_constraint=maxnorm(3)))

    model.add(Dense(D_op, activation='linear', use_bias='True'))#,kernel_constraint=maxnorm(3)))
    model.summary()
    return model


def trainining_DNN(x_train, y_train,Nl,nN,dR):

    Dip = x_train.shape[1]
    Dop = y_train.shape[1]
    batch_size = 100
    epochs = 250  # 20 #500
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model = construct_DNN( Dip,Dop,Nl,nN,dR )
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
dnnOpts =  x['dnnOpts']
x = x['DNN_x']

DNNmodel = trainining_DNN(X,Y,int(dnnOpts[0][0]),int(dnnOpts[0][1]),float(dnnOpts[0][2])) # train model
savePath = modPath + '/DNN_model.h5'
DNNmodel.save(savePath)


y = DNNmodel.predict(x)
savePath = modPath + '/pyPredTest.mat' 
scipy.io.savemat(savePath, mdict={'y': y})


    
