
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


# In[2]:


fx = pd.read_csv('fx_raw.csv')
bz = fx.bz
plt.plot(bz)


# In[ ]:


#rcParams['figure.figsize'] = 15, 15
#make huge chart of acf to determine lag length
plot_acf(bz, title = 'acf of bz');
plot_pacf(bz, title= 'pacf of bz', lags=100);
#displays AR 1 signature


# In[3]:


from sklearn.preprocessing import MinMaxScaler
bz = bz.values
bz = bz.reshape(len(bz), 1)
sc = MinMaxScaler(feature_range = (0, 1))
bz_scl = sc.fit_transform(bz)
bz = bz_scl
bz = bz.reshape(len(bz))
bz = pd.Series(bz)


# In[4]:


#check integrity of scale
plt.plot(bz)


# In[5]:


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


# In[6]:


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


# In[7]:


shp_lstm(bz, 1)
train_test_idx(X,y,4)


# In[8]:


X_train, y_train = X[train[3]], y[train[3]]
X_test, y_test = X[test[3][:-1]], y[test[3][:-1]]


# ### Testing Different Configurations of Neurons

# In[ ]:


get_ipython().system(' pip install keras')


# In[ ]:


get_ipython().system(' pip install tensorflow ')


# In[9]:


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
plt.title('Neuron Configurations - BRL/USD, 1000 Epochs')
plt.xlabel('neurons')
plt.ylabel('mse')
plt.savefig('performbz_undiff')


# ### Testing The Model at optimal neuron

# In[10]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(1,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)


# In[11]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.02)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0150, s='Test Set MSE :'+str(score)[:8])
plt.title('MSE of 20 NEUR LSTM, BRL/USD Undifferenced', size=14)
plt.savefig('mse_bz_usd_undiff')


# In[12]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[13]:


plot_acf(resid);
plt.title('ACF of Residuals - BRL/USD undifferenced')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_bz_undiff')


# In[14]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[15]:


bz = bz.values
bz = bz.reshape(len(bz), 1)
yhat = yhat.reshape(len(yhat), 1)

bz = sc.inverse_transform(bz)
yhat = sc.inverse_transform(yhat)

yhat = yhat.reshape(len(yhat))
bz = bz.reshape(len(bz))

yhat = pd.Series(yhat)
bz = pd.Series(bz)

yhat.index = test[3][:-1]

pred = pd.DataFrame([yhat, bz, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[16]:


from sklearn.metrics import mean_squared_error as mse

fcst_err = mse(pred.pred[test[3][:-1]],pred.actual[test[3][:-1]])

fcst_err


# In[17]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('BRL/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('BRL/USD 2014-2019', size=18)
plt.text(x='2017-12', y=3.0, s='Test Set MSE :'+str(fcst_err)[:8], 
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('forecast20neurbz_undiff')


# ### Using a smaller training sample and two validation sets

# In[18]:


X_train, y_train = X[train[2]], y[train[2]]
X_test, y_test = X[test[2]], y[test[2]]


# In[19]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(1,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)


# In[20]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.02)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0150, s='Test Set MSE :'+str(score)[:8])
plt.title('MSE of 20 NEUR LSTM, New Validation Set', size=14)
plt.savefig('mse_bz_usd_val1')


# In[21]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[22]:


plot_acf(resid);
plt.title('ACF of Residuals - BRL/USD, New Validation Set')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_bz_undiff_val1')


# In[23]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[24]:


plt.plot(yhat)
plt.plot(y_test)


# In[28]:


yhat = yhat.reshape(len(yhat), 1)

yhat = sc.inverse_transform(yhat)
yhat = yhat.reshape(len(yhat))

yhat = pd.Series(yhat)

yhat.index = test[2]

pred = pd.DataFrame([yhat, bz, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[29]:


fcst_err = mse(pred.pred[test[2]],pred.actual[test[2]])

fcst_err


# In[33]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('BRL/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('BRL/USD 2014-2019, new validation set', size=18)
plt.text(x='2017-12', y=3.0, s='Test Set MSE :'+str(fcst_err)[:8]+'e-7', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('forecast20neurbz_undiff_val1')


# ### Backtesting

# In[36]:


test[1]


# In[37]:


test[2]


# In[42]:


#note - I am using the test list of indexes because it allows me to call individual years
X_train, y_train = X[test[2]], y[test[2]]


# In[44]:


X_test, y_test = X[test[1]], y[test[1]]


# In[46]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(1,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)


# In[47]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.02)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0150, s='Test Set MSE :'+str(score)[:8])
plt.title('MSE of 20 NEUR LSTM, Backtesting', size=14)
plt.savefig('mse_bz_usd_val2')


# In[48]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[49]:


plot_acf(resid);
plt.title('ACF of Residuals - BRL/USD, New Validation Set')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_bz_undiff_val2')


# In[50]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[51]:


plt.plot(yhat)
plt.plot(y_test)


# In[52]:


yhat = yhat.reshape(len(yhat), 1)

yhat = sc.inverse_transform(yhat)
yhat = yhat.reshape(len(yhat))

yhat = pd.Series(yhat)

yhat.index = test[1]

pred = pd.DataFrame([yhat, bz, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[53]:


fcst_err = mse(pred.pred[test[1]],pred.actual[test[1]])

fcst_err


# In[54]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('BRL/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('BRL/USD 2014-2019, new validation set', size=18)
plt.text(x='2017-12', y=3.0, s='Test Set MSE :'+str(fcst_err)[:8], 
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('forecast20neurbz_undiff_val2')


# In[2]:




