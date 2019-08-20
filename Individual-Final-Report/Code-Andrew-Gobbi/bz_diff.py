
# coding: utf-8

# ### Preprocessing

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

fx = pd.read_csv('fx_raw.csv')
bz = fx.bz


# In[2]:


from sklearn.preprocessing import MinMaxScaler
bz = bz.values
bz = bz.reshape(len(bz), 1)
sc = MinMaxScaler(feature_range = (0, 1))
bz_scl = sc.fit_transform(bz)
bz = bz_scl
bz = bz.reshape(len(bz))
bz = pd.Series(bz)


# In[3]:


bz_diff = bz.diff(1)


# In[4]:


def shp_lstm(data, lags, features=1):
    global X, y
    X = pd.DataFrame()
    y = pd.Series()
    for i in range(lags):
        X[i] = data.shift(-i)
    y = data.shift(-lags)
    X.dropna(inplace=True)
    X = X.values
    X = X.reshape(X.shape[0], X.shape[1], features)
    y.dropna(inplace=True)
    y = y.values  


# In[5]:


def train_test_idx(x, y, splits):
    from sklearn.model_selection import TimeSeriesSplit
    d = []
    global train, test
    train = []
    test = []
    tscv = TimeSeriesSplit(n_splits=splits)
    
    for train_index, test_index in tscv.split(x):
        d.append(train_index)
        d.append(test_index)
    
    for i in range(len(d)):
        if i%2 ==0:
            train.append(d[i])
        else:
            test.append(d[i])


# In[6]:


shp_lstm(bz_diff, 1)
train_test_idx(X,y,4)


# In[7]:


X_train, y_train = X[train[3]], y[train[3]]
X_test, y_test = X[test[3][:-1]], y[test[3][:-1]]


# ### Test for Optimal Neuron

# In[ ]:


get_ipython().system(' pip install keras')


# In[ ]:


get_ipython().system(' pip install tensorflow ')


# In[8]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# In[ ]:


cells = np.arange(10, 110, 10)
cv = []

for i in range(len(cells)):
    
    model = Sequential()
    model.add(LSTM(cells[i], activation='tanh', input_shape=(1,1)))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    
    print('1000 Epochs, MSE Neurons {} :'.format(cells[i]), score)
    cv.append(score)


# In[ ]:


plt.bar(cells, cv)
plt.title('BRL/USD Errors at Different Neuron Configurations', size=14)
plt.xlabel('neurons')
plt.ylabel('mse')
plt.savefig('Gridsearch_BZ_diff')


# ### Testing at optimal neuron

# In[9]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(1,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)


# In[10]:


score


# In[12]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.0001)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0000125, s='Test Set MSE :'+str(score)[:4]+'e-7')
plt.title('MSE - 20 NEUR, LSTM, BRL/USD Differences', size=14)
plt.savefig('MSE_diff_BZ')


# In[13]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[14]:


plot_acf(resid);
plt.title('ACF of Residuals - BRL/USD Differences')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('BZ_resid_diff')


# In[15]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[21]:


bz_diff = bz_diff.dropna()


# In[23]:


bz_diff = bz_diff.values


# In[24]:


bz_diff = bz_diff.reshape(len(bz_diff), 1)
yhat = yhat.reshape(len(yhat), 1)

bz_diff = sc.inverse_transform(bz_diff)
yhat = sc.inverse_transform(yhat)

yhat = yhat.reshape(len(yhat))
bz_diff = bz_diff.reshape(len(bz_diff))

yhat = pd.Series(yhat)
bz_diff = pd.Series(bz_diff)

yhat.index = test[3][:-1]

pred = pd.DataFrame([yhat, bz_diff, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[25]:


from sklearn.metrics import mean_squared_error as mse

fcst_err = mse(pred.pred[test[3][:-1]],pred.actual[test[3][:-1]])

fcst_err


# In[26]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('BRL/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('BRL/USD 2014-2019', size=18)
plt.text(x='2017-12', y=2.4, s='Test Set MSE :'+str(fcst_err)[:5]+'e-07', bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('forecast20neurbz_diff')


# ### Validation 1 

# In[27]:


X_train, y_train = X[train[2]], y[train[2]]
X_test, y_test = X[test[2]], y[test[2]]


# In[28]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(1,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)


# In[30]:


score


# In[29]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.0001)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0000125, s='Test Set MSE :'+str(score)[:4]+'e-7')
plt.title('MSE - 20 NEUR, LSTM, BRL/USD Differences, New Validation Set', size=14)
plt.savefig('MSE_diff_BZ_val1')


# In[40]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[41]:


plot_acf(resid);
plt.title('ACF of Residuals - BRL/USD Differences, New Validation Set')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('BZ_resid_diff_val1')


# In[42]:


adfuller(resid)


# In[43]:


yhat = yhat.reshape(len(yhat), 1)

yhat = sc.inverse_transform(yhat)

yhat = yhat.reshape(len(yhat))

yhat = pd.Series(yhat)

yhat.index = test[2]

pred = pd.DataFrame([yhat, bz_diff, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[45]:


fcst_err = mse(pred.pred[test[2]],pred.actual[test[2]])

fcst_err


# In[46]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('BRL/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('BRL/USD 2014-2019', size=18)
plt.text(x='2017-12', y=2.4, s='Test Set MSE :'+str(fcst_err)[:5]+'e-07', bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('forecast20neurbz_diff_val1')


# ### Backtesting

# In[47]:


#note - I am using the test list of indexes because it allows me to call individual years
X_train, y_train = X[test[2]], y[test[2]]


# In[48]:


X_test, y_test = X[test[1]], y[test[1]]


# In[49]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(1,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)


# In[50]:


score


# In[52]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.0001)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0000125, s='Test Set MSE :'+str(score)[:4]+'e-7')
plt.title('MSE - 20 NEUR, LSTM, BRL/USD Differences, Backtesting', size=14)
plt.savefig('MSE_diff_BZ_val2')


# In[57]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[58]:


plot_acf(resid);
plt.title('ACF of Residuals - BRL/USD Differences, Backtesting')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('BZ_resid_diff_val2')


# In[59]:


adfuller(resid)


# In[60]:


yhat = yhat.reshape(len(yhat), 1)

yhat = sc.inverse_transform(yhat)

yhat = yhat.reshape(len(yhat))

yhat = pd.Series(yhat)

yhat.index = test[1]

pred = pd.DataFrame([yhat, bz_diff, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[61]:


fcst_err = mse(pred.pred[test[1]],pred.actual[test[1]])

fcst_err


# In[62]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('BRL/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('BRL/USD 2014-2019, backtesting', size=18)
plt.text(x='2017-12', y=2.4, s='Test Set MSE :'+str(fcst_err)[:5]+'e-06', bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('forecast20neurbz_diff_val2')

