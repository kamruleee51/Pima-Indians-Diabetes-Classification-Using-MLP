# -*- coding: utf-8 -*-
"""
Created on Thu May 24 08:35:26 2018

@author: Md. Kamrul Hasan

"""
print(__doc__)
#%% Import all the Library for this problem
import time
startTime = time.time()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import tflearn.initializations as tfi

#%% Read CSV Data and Pre-processing(standardized and split) of Data
RawData= pd.read_csv('hw2data.csv',header=None)
RawdataArray=np.array(RawData)

#==================== Standardization of RawData====================
standardized_RawdataArray = preprocessing.scale(RawdataArray[:,0:10])

#=================== Positive and Negative Splitt ==================
PositiveData=standardized_RawdataArray[0:4000,:]
NegativeData=standardized_RawdataArray[4000:8000,:]

#========Concatenate Negative 1-4000 and Positive 4001-8000=========
Data=np.concatenate((NegativeData, PositiveData), axis=0)

#==================== Extract the Labels============================
label=RawdataArray[:,10]

#===============Make the Label -1 to 0==============================
CopyLabel = label.copy()
CopyLabel[CopyLabel < 0] = 0

#============== Positive and Negative Split of the Label===========
LabelPositive=CopyLabel[0:4000]
LabelNegative=CopyLabel[4000:8000]

#============Label concatenate. lebel 0 first then Label 1.=========
NewLabel=np.concatenate((LabelNegative, LabelPositive), axis=0)

#=================Train and Test split==============================
Data_train, Data_test, Label_train, Label_test = train_test_split(Data, NewLabel, test_size=0.25, random_state=100)

#=========One Hot Encodding of the Train and Test Label=============
LabelTrain_1Hot = to_categorical(Label_train,2)
LabelTest_1Hot = to_categorical(Label_test,2)

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

    #========Train The model with Input Train Data==================
    model.fit(Data_train, LabelTrain_1Hot, show_metric=True, batch_size=BatchSize, n_epoch=NumberofEpoch, snapshot_epoch=False)

# ==================Save a model or comment this line===============
model.save('mymodel.tfl')

#%% ===================Evaluate the Model============================
predicted=model.predict(Data_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NumberofClass):
    fpr[i], tpr[i], _ = roc_curve(LabelTest_1Hot[:,i], predicted[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure()
plt.grid(True)
plt.plot(fpr[1], tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')
plt.show()
y_pred=np.argmax(model.predict(Data_test), axis=1)
plt.figure()
skplt.metrics.plot_confusion_matrix(Label_test, y_pred, normalize=False)
plt.show()
print("Accuracy: {}%".format(100 * np.mean(Label_test == np.argmax(model.predict(Data_test), axis=1))))
auc=(roc_auc[0]+roc_auc[1])/2
print("Area Under ROC (AUC): {}".format(auc))
endTime = time.time()
print('It took {0:0.1f} seconds'.format(endTime - startTime))
#%%                  THE END..........................