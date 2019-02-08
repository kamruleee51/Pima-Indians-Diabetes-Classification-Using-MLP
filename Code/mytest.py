#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:52:51 2018

@author: francesco
"""

import time
startTime = time.time()
import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from sklearn import preprocessing
import tflearn.initializations as tfi

#%% Read CSV Data and Pre-processing(standardized and split) of Data
'data Load along with standardizations'
RawData= preprocessing.scale(pd.read_csv('hw2data.csv',header=None))
'Replace hw2data.csv by your data if you want to test new data but do not delete pre-processing'

label=RawData[:,10]
CopyLabel = label.copy()
CopyLabel[CopyLabel < 0] = 0 # Making -1 label to 0.

'From the RawData and Copy Label, you can select any rows to test the saved model but make sure number of rows in TestData should have same labels to see the accuracy'
TestData=RawData[0:500,0:10] # Your test data
Testlabel=CopyLabel[0:500] # your test Data labels

#%%================Create the Model=================================
#           'Hyperparameters in the Network'
NumberofClass=2
NumberofFeatures=10
NumberofNeuronHidden_1=25
NumberofNeuronHidden_2=25
LearningRate=0.01
BatchSize=200
NumberofEpoch=350

with tf.Graph().as_default():

    '------------------Fed the Input to the Network----------------'
    net = tflearn.input_data([None, NumberofFeatures])

    '-----Normalize the Batch for unwanted biased from the input----'
    net=tflearn.layers.normalization.batch_normalization (net, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002, trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization')

    #==============Hidden-1 layer Having 25 Neurons=================
    net = tflearn.fully_connected(net, NumberofNeuronHidden_1, activation='tanh',bias=True, weights_init=tfi.xavier (uniform=True, seed=None, dtype=tf.float32), bias_init=tfi.normal (shape=None, mean=0.0, stddev=0.02, dtype=tf.float32, seed=None), regularizer='L2', weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='FullyConnected')

    '-------------Normalize the Batch for unwanted biased from the previous Weight---------------'
    net=tflearn.layers.normalization.batch_normalization (net, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002, trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization')

    #=============Hidden-2 layer Having 25 Neurons================
    net = tflearn.fully_connected(net, NumberofNeuronHidden_2, activation='tanh',bias=True, weights_init=tfi.xavier (uniform=True, seed=None, dtype=tf.float32), bias_init=tfi.normal (shape=None, mean=0.0, stddev=0.02, dtype=tf.float32, seed=None), regularizer='L2', weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='FullyConnected')


    '-----------Normalize the Batch for unwanted biased from the previous Weight--------------'
    net=tflearn.layers.normalization.batch_normalization (net, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002, trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization')

    #=========Output Layer Having Two Class only.===================
    net = tflearn.fully_connected(net, NumberofClass, activation='softmax',bias=True, weights_init=tfi.xavier (uniform=True, seed=None, dtype=tf.float32), bias_init=tfi.normal (shape=None, mean=0.0, stddev=0.02, dtype=tf.float32, seed=None), regularizer='L2', weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='FullyConnected')

    #==================Gradient Optimizer Adam=======================
    GraddecientOptimizer = tflearn.Adam (learning_rate=LearningRate)

    #============Fed Optimizer to the Regression====================
    net = tflearn.regression(net, optimizer=GraddecientOptimizer, loss='categorical_crossentropy')

    #==========Create the Multilayer Perceptron Model===============

    model = tflearn.DNN(net,tensorboard_verbose=0)

# Loading the Saved Model------------------------
    model.load('mymodel.tfl')



print("Accuracy: {}%".format(100 * np.mean(Testlabel == np.argmax(model.predict(TestData), axis=1))))
endTime = time.time()
print('It took {0:0.1f} seconds'.format(endTime - startTime))