#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 00:40:39 2020

@author: starry
"""

# Recurrent Neural Network

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import  the trainning set
dataset_train = pd.read_csv('Dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with N(experienced num) timesteps an 1 output
# Set N = 60 timesteps (3 months) could achiever good result from experience.
N = 60
x_train = [] # Input: 60 previews stock prices before financial day
y_train = [] # Output: The stock proce the next financial day
# start to make predict
for i in range(N, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-N:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train) 
# Reshaping to RNN Input shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



# Part 2 - Building the RNN

# Part 3 - Making the predictions and visualising the results