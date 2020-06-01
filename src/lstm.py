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

# Import  the trainning set, 2012-2016 stock price record
dataset_train = pd.read_csv('./Dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with N(experienced num) timesteps an 1 output
# Set N = 60 timesteps (3 months) could achiever good result from experience.
N = 60
x_train = [] # Input: 60 previews stock prices before financial day
y_train = [] # Output: The stock price the next financial day - Ground Truth
# start to make predict
for i in range(N, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-N:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train) 
# Reshaping to RNN Input shape, (observation, timestamps, indicator)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



# Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# add LSTM layer and Dropout regulatisation, return_sequences default value is False
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2)) #20% of 50, 10 neurons will be ignored

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# add output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fit the RNN to the training set
regressor.fit(x_train, y_train, batch_size=32, epochs=100)

# Part 3 - Making the predictions and visualising the results
# real stock price of 2017 as the test set 
dataset_test = pd.read_csv('./Dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - N:].values
inputs = inputs.reshape(-1, 1) # len(input) = N + len(test) ?
inputs = sc.transform(inputs) # MinMaxScaler(feature_range=(0, 1))

x_test = [] # Input: N=60 previews stock prices before financial day

# start to make predict
for i in range(N, N+len(real_stock_price)):
    x_test.append(inputs[i-N:i, 0])

x_test= np.array(x_test)
# Reshaping to RNN Input shape, (observation, timestamps, indicator)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualize the results
plt.plot(real_stock_price, color="red", label='Real Google Stock Price')
plt.plot(predicted_stock_price, color="blue", label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Tuning the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def build_regressor(optimizer):
    regressor = Sequential()
    
    # add LSTM layer and Dropout regulatisation, return_sequences default value is False
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    regressor.add(Dropout(0.2)) #20% of 50, 10 neurons will be ignored
    
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    
    # add output layer
    regressor.add(Dense(units=1))
    
    # Compiling the RNN
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return regressor

regressor = KerasRegressor(build_fn = build_regressor)
    
parameters = {'batch_size': [25, 32, 64],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error')


# fit the RNN to the training set
grid_search = regressor.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




