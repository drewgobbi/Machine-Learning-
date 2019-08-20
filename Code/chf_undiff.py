
# coding: utf-8

# ### PreProcessing

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
sz = fx.sz
plt.plot(sz)


# In[3]:


#rcParams['figure.figsize'] = 15, 15
#make huge chart of acf to determine lag length
plot_acf(sz, title = 'acf of sz');
plot_pacf(sz, title= 'pacf of sz', lags=100);


# In[4]:


sz.describe()


# In[5]:


from sklearn.preprocessing import MinMaxScaler
sz = sz.values
sz = sz.reshape(len(sz), 1)
sc = MinMaxScaler(feature_range = (0, 1))
sz_scl = sc.fit_transform(sz)
sz = sz_scl
sz = sz.reshape(len(sz))
sz = pd.Series(sz)


# In[6]:


#check integrity of scale
plt.plot(sz)


# In[7]:


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


# In[8]:


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


# In[9]:


shp_lstm(sz, 2)
train_test_idx(X, y, int(len(sz)/253))


# In[10]:


X_train, y_train = X[train[3]], y[train[3]]
X_test, y_test = X[test[3][:-1]], y[test[3][:-1]]


# ### test for optimal neuron

# In[ ]:


#running on a non-persistent environment
get_ipython().system(' pip install keras')


# In[ ]:


get_ipython().system(' pip install tensorflow ')


# In[11]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

cells = np.arange(10, 110, 10)
cv = []

for i in range(len(cells)):
    
    model = Sequential()
    model.add(LSTM(cells[i], activation='tanh', input_shape=(2,1)))
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
plt.xlabel('neurons', size=12)
plt.ylabel('mse', size=12)
plt.title('1000 Epoch Performance CHF/USD Undifferenced, Various Neurons', size = 18)
plt.savefig('neuron_perform_CHF_AR2sz.png')


# ### test optimal neuron config
# Running on a non-persistent environment, so ignore install if you are not

# In[ ]:


get_ipython().system(' pip install keras')


# In[ ]:


get_ipython().system(' pip install tensorflow ')


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# In[ ]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(2,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                            verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


score


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.02)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0150, s='Test Set MSE :'+str(score)[:4]+'e-5')
plt.title('MSE of 20 NEUR LSTM - CHF/USD', size=14)
plt.savefig('MSE_CHFUSD_20NEUR')


# In[ ]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[ ]:


plot_acf(resid);
plt.title('ACF of Residuals CHF/USD')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('CHF_undiff_resid')


# In[ ]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[ ]:


sz = sz.values
sz = sz.reshape(len(sz), 1)
yhat = yhat.reshape(len(yhat), 1)

sz = sc.inverse_transform(sz)
yhat = sc.inverse_transform(yhat)


# In[ ]:


yhat = yhat.reshape(len(yhat))
sz = sz.reshape(len(sz))


# In[ ]:


yhat = pd.Series(yhat)
sz = pd.Series(sz)


# In[ ]:


yhat.index = test[3][:-1]


# In[ ]:


pred = pd.DataFrame([yhat, sz, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)


# In[ ]:


pred.DATE = pd.to_datetime(pred.DATE)


# In[ ]:


pred.set_index('DATE', inplace=True)


# In[ ]:


from sklearn.metrics import mean_squared_error as mse


# In[ ]:


fcst_err = mse(pred.pred[test[3][:-1]],pred.actual[test[3][:-1]])


# In[ ]:


fcst_err


# In[ ]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('SZ/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('CHF/USD 2014-2019', size=18)
plt.text(x='2017-12', y=.875, s='Test Set MSE :'+str(fcst_err)[:5]+'e-05', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('forecast20neur_CHFUSD')


# ### Validation 1

# In[ ]:


X_train, y_train = X[train[2]], y[train[2]]
X_test, y_test = X[test[2]], y[test[2]]


# In[ ]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(2,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                            verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.02)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0150, s='Test Set MSE :'+str(score)[:8])
plt.title('MSE of 20 NEUR LSTM, new validation set', size=14)
plt.savefig('chfusd_val1')


# In[ ]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[ ]:


plot_acf(resid);
plt.title('ACF of Residuals - CHF/USD, new validation set')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_chfusd_val1')


# In[ ]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[ ]:


yhat = yhat.reshape(len(yhat), 1)
yhat = sc.inverse_transform(yhat)


# In[ ]:


yhat = yhat.reshape(len(yhat))


# In[ ]:


yhat = pd.Series(yhat)
sz = pd.Series(sz)


# In[ ]:


yhat.index = test[2]


# In[ ]:


pred = pd.DataFrame([yhat, sz, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)


# In[ ]:


pred.DATE = pd.to_datetime(pred.DATE)


# In[ ]:


pred.set_index('DATE', inplace=True)


# In[ ]:


from sklearn.metrics import mean_squared_error as mse


# In[ ]:


fcst_error = mse(pred.pred[test[2]],pred.actual[test[2]])


# In[ ]:


fcst_error


# In[ ]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('SZ/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('CHF/USD 2014-2019, new validation set', size=18)
plt.text(x='2017-12', y=.875, s='Test Set MSE :'+str(fcst_error)[:5]+'e-05', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('chf_usd_fcst2')


# ### Val2

# In[ ]:


X_train, y_train = X[test[2]], y[test[2]]
X_test, y_test = X[test[1]], y[test[1]]


# In[ ]:


model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(2,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                            verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs', size=12)
plt.ylabel('MSE', size =12)
plt.ylim(0,.02)
plt.legend(['train', 'test'])
plt.text(x=700, y=.0150, s='Test Set MSE :'+str(score)[:8])
plt.title('MSE of 20 NEUR LSTM, new validation set', size=14)
plt.savefig('chfusd_val2')


# In[ ]:


yhat = model.predict(X_test)
yhat = yhat.reshape(len(yhat))
resid = yhat - y_test


# In[ ]:


plot_acf(resid);
plt.title('ACF of Residuals - CHF/USD, new validation set')
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_chfusd_val2')


# In[ ]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[ ]:


yhat = yhat.reshape(len(yhat), 1)
yhat = sc.inverse_transform(yhat)


# In[ ]:


yhat = yhat.reshape(len(yhat))


# In[ ]:


yhat = pd.Series(yhat)
sz = pd.Series(sz)


# In[ ]:


yhat.index = test[1]


# In[ ]:


pred = pd.DataFrame([yhat, sz, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)


# In[ ]:


pred.DATE = pd.to_datetime(pred.DATE)


# In[ ]:


pred.set_index('DATE', inplace=True)


# In[ ]:


from sklearn.metrics import mean_squared_error as mse


# In[ ]:


fcst_error = mse(pred.pred[test[1]],pred.actual[test[1]])


# In[ ]:


fcst_error


# In[ ]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('SZ/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('CHF/USD 2014-2019, new validation set', size=18)
plt.text(x='2017-12', y=.875, s='Test Set MSE :'+str(fcst_error)[:5]+'e-05', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('chf_usd_fcst2')

