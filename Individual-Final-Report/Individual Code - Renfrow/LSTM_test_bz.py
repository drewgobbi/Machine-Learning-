# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 10:33:48 2019

@author: grady
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
from pandas.tools.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas import Series


#------------------------------------------------------------------------------
#Part 1 
#Importing the training set 
dataset_train = pd.read_csv('fx.csv')
training_set = dataset_train.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 30 timesteps and 1 output
X_train = []
y_train = []
for i in range(30,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-30:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping - Based on shape required by Keras (batch_size, timesteps, input_dim)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

#Test Set---------------------------------------------------------------------
dataset_test = pd.read_csv('fx.csv')
real_fx_rate = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train["bz"], dataset_test['bz']), axis = 0)
inputs = dataset_total[len(dataset_total)- len(dataset_test) - 30:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(30,1136):
    X_test.append(inputs[i-30:i,0])
X_test =np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

#-----------------------------------------------------------------------------
#Part 2 - Building the RNN 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN 
mod_bz = Sequential () 

#Adding the first LSTM layer and Dropout regularization
mod_bz.add(LSTM(units = 50, activation='tanh', return_sequences = False, input_shape = (X_train.shape[1], 1)))
mod_bz.add(Dropout(0.2))
 
# Addising a second layer and same Dropout regularization
#mod_bz.add(LSTM(units = 50, return_sequences = True))  #Not used in final model 
mod_bz.add(Dropout(0.2))        
    
#Adding third layer and same regularization        
#mod_bz.add(LSTM(units = 50)) # only used for testing purposes 
mod_bz.add(Dropout(0.2))

#Adding output layer
mod_bz.add(Dense(units=1))

#Compiling the RNN
#mod_bz.compile(optimizer = "RMSprop", loss ="mse") #not used in the final model 
mod_bz.compile(optimizer = "adam", loss ="mse")

#Fitting the RNN to the Training set
mod_bz.fit(X_train, y_train, epochs = 1000, batch_size = 32)
bz_history = mod_bz.fit(X_train, y_train, epochs = 1000, batch_size = 32, validation_data=(X_test, ))

#60
mod_bz_60 = Sequential ()
mod_bz_60.add(LSTM(units = 60, activation='tanh', return_sequences = False, input_shape = (X_train.shape[1], 1)))
mod_bz_60.add(Dropout(0.2))
mod_bz_60.add(Dense(units=1))
mod_bz_60.compile(optimizer = "adam", loss ="mse")
bz_60_history = mod_bz_60.fit(X_train, y_train, epochs = 1000, batch_size = 32)

#70
mod_bz_70 = Sequential ()
mod_bz_70.add(LSTM(units = 70, activation='tanh', return_sequences = False, input_shape = (X_train.shape[1], 1)))
mod_bz_70.add(Dropout(0.2))
mod_bz_70.add(Dense(units=1))
mod_bz_70.compile(optimizer = "adam", loss ="mse")
bz_70_history = mod_bz_70.fit(X_train, y_train, epochs = 1000, batch_size = 32)

#30
mod_bz_30 = Sequential ()
mod_bz_30.add(LSTM(units = 30, activation='tanh', return_sequences = False, input_shape = (X_train.shape[1], 1)))
mod_bz_30.add(Dropout(0.2))
mod_bz_30.add(Dense(units=1))
mod_bz_30.compile(optimizer = "adam", loss ="mse")
bz_30_history = mod_bz_30.fit(X_train, y_train, epochs = 1000, batch_size = 32)

mod_bz_20 = Sequential ()
mod_bz_20.add(LSTM(units = 20, activation='tanh', return_sequences = False, input_shape = (X_train.shape[1], 1)))
mod_bz_20.add(Dropout(0.2))
mod_bz_20.add(Dense(units=1))
mod_bz_20.compile(optimizer = "adam", loss ="mse")
bz_20_history = mod_bz_20.fit(X_train, y_train, epochs = 1000, batch_size = 32)

#10
mod_bz_10 = Sequential ()
mod_bz_10.add(LSTM(units = 10, activation='tanh', return_sequences = False, input_shape = (X_train.shape[1], 1)))
mod_bz_10.add(Dropout(0.2))
mod_bz_10.add(Dense(units=1))
mod_bz_10.compile(optimizer = "RMSprop", loss ="mse")
bz_10_history = mod_bz_10.fit(X_train, y_train, epochs = 1000, batch_size = 32)

#PLot loss and epochs
plt.plot(bz_history.history['loss'], c='b')
plt.plot(bz_60_history.history['loss'], c= 'g')
plt.plot(bz_70_history.history['loss'], c='r')
plt.plot(bz_30_history.history['loss'], c='black')
plt.plot(bz_20_history.history['loss'], c='purple')
plt.plot(bz_10_history.history['loss'], c='yellow')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0,20)
plt.legend(['bz', 'bz_60' , 'bz_70' , 'bz_30', 'bz_20', 'bz_10'], loc='upper right')
plt.show()

#-----------------------------------------------------------------------------
#Part 3 - Making the prediction and visualizing the results 
dataset_test = pd.read_csv('fx.csv')
real_fx_rate = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train["bz"], dataset_test['bz']), axis = 0)
inputs = dataset_total[len(dataset_total)- len(dataset_test) - 30:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(30,1136):
   X_test.append(inputs[i-30:i,0])
X_test =np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

#Getting the predicted exchange rate 
predicted_fx_rate = mod_bz.predict(X_test)
predicted_fx_rate = sc.inverse_transform(predicted_fx_rate)

predicted_fx_rate30 = mod_bz_30.predict(X_test)
predicted_fx_rate30 = sc.inverse_transform(predicted_fx_rate30)
predicted_fx_rate60 = mod_bz_60.predict(X_test)
predicted_fx_rate60 = sc.inverse_transform(predicted_fx_rate60)
predicted_fx_rate70 = mod_bz_70.predict(X_test)
predicted_fx_rate70 = sc.inverse_transform(predicted_fx_rate70)
predicted_fx_rate10 = mod_bz_10.predict(X_test)
predicted_fx_rate10 = sc.inverse_transform(predicted_fx_rate10)
predicted_fx_rate20 = mod_bz_20.predict(X_test)
predicted_fx_rate20 = sc.inverse_transform(predicted_fx_rate20)


#Viz
plt.plot(real_fx_rate, color = 'red', label = 'Real BZ/US Exchange Rate')
plt.plot(predicted_fx_rate, color = 'purple' , label = 'Predicted BX/US Exchange Rate')
plt.plot(predicted_fx_rate30, color = 'blue' , label = 'Pred 30')
plt.plot(predicted_fx_rate60, color = 'green' , label = 'Pred 60')
plt.plot(predicted_fx_rate70, color = 'black' , label = 'Pred 70')
plt.plot(predicted_fx_rate10, color = 'yellow', label = 'Pred 10')
plt.plot(predicted_fx_rate20, color = 'purple', label = 'Pred 20')
plt.title('Brazilian Real / US Dollar Exchange Rate Prediction' )
plt.xlabel('Time')
plt.ylabel('BZ / US Exchange Rate')
plt.legend()
plt.show()
#------------------------------------------------------------------------------








