e#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:41:25 2017

@author: Rachana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping
X_train = np.reshape(X_train, (1257, 1, 1))


# Building RNN 
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM

#Initialize the RNN 
#using Regressional 
regressor = Sequential()

#adding the inport layer and the LSTM layer
#4 mem unit, sigmoid activation 
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#adding output layer
regressor.add(Dense(units = 1))

#compliing the RNN
regressor.compile()