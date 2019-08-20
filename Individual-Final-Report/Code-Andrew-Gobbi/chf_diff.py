
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
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller


# In[2]:


fx = pd.read_csv('fx_raw.csv')
sz = fx.sz


# In[3]:


sz = sz.values
sz = sz.reshape(len(sz), 1)
sc = MinMaxScaler(feature_range = (0, 1))
sz_scl = sc.fit_transform(sz)
sz = sz_scl
sz = sz.reshape(len(sz))
sz = pd.Series(sz)


# In[4]:


sz_diff = sz.diff(1)


# In[5]:


plt.plot(sz_diff)
plt.title(' Normalized and Differenced CHF/USD')


# In[6]:


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


# In[7]:


shp_lstm(sz_diff, 2)


# In[8]:


#there are 253 trading days in a year
print('Years in dataset', len(X)/253)


# In[9]:


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


# In[10]:


train_test_idx(X, y, int(len(X)/253))


# In[11]:


X_train, X_test = X[train[3][:-1]], X[test[3][:-1]]
y_train, y_test = y[train[3][:-1]], y[test[3][:-1]]


# ## Test for Optimal Neuron Configuration

# In[ ]:


get_ipython().system(' pip install keras')


# In[ ]:


get_ipython().system(' pip install tensorflow ')


# In[12]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# In[ ]:


cells = np.arange(10, 110, 10)
cv = []

for i in range(len(cells)):
    
    model = Sequential()
    model.add(LSTM(cells[i], activation='tanh', input_shape=(2,1)))
    model.add(Dropout(rate = 0.2))
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
plt.title('1000 Epoch Performance, Various Neurons, CHF/USD Differenced', size = 18)
plt.savefig('neuron_perform_CHF_AR2sz_diff.png')


# ### Test at optimal neuron configuration

# In[13]:


model = Sequential()
model.add(LSTM(5, activation='tanh', input_shape=(2,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)


# In[14]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim(0,.001)
plt.title('MSE of 5 NEUR LSTM - Differenced Data CHF/USD', size=14) 
plt.savefig('mse_CHF_USD_diff')


# In[15]:


yhat_diff = model.predict(X_test)


# In[16]:


plt.plot(yhat_diff)
plt.plot(y_test)
plt.legend(['predicted', 'actual'])
plt.title('Differenced CHF/USD Forecast Results')
plt.xlabel('timesteps')
plt.ylabel('Daily Differences - CHF/USD')
plt.savefig('CHF_USD_Differenced_FCST')


# In[17]:


resid = yhat_diff - y_test


# In[18]:


yhat_diff.shape


# In[19]:


y_test.shape


# In[20]:


yhat_diff = yhat_diff.reshape(len(yhat_diff))


# In[21]:


resid = yhat_diff -y_test


# In[22]:


len(resid)


# In[23]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(resid, title='ACF of Residuals - Integrated AR CHF/USD');
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_resid_CHF_diff')


# In[24]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[25]:


sz_diff = sz_diff.dropna()
sz_diff = sz_diff.values


# In[26]:


sz_diff = sz_diff.reshape(len(sz_diff), 1)
yhat_diff = yhat_diff.reshape(len(yhat_diff), 1)

sz_diff = sc.inverse_transform(sz_diff)
yhat_diff = sc.inverse_transform(yhat_diff)

yhat_diff = yhat_diff.reshape(len(yhat_diff))
sz_diff = sz_diff.reshape(len(sz_diff))

yhat_diff = pd.Series(yhat_diff)
sz_diff = pd.Series(sz_diff)

yhat_diff.index = test[3][:-1]

pred = pd.DataFrame([yhat_diff, sz_diff, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[27]:


from sklearn.metrics import mean_squared_error as mse

fcst_err = mse(pred.pred[test[3][:-1]],pred.actual[test[3][:-1]])

fcst_err


# In[28]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('CHF/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('CHF/USD 2014-2019 Diff', size=18)
#plt.text(x='2017-12', y=2.4, s='Test Set MSE :'+str(fcst_err)[:5]+'e-05', bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('chf_usd_diff_fcst')


# ### Validation 1 

# In[ ]:


X_train, y_train = X[train[2]], y[train[2]]
X_test, y_test = X[test[2]], y[test[2]]


# In[ ]:


model = Sequential()
model.add(LSTM(5, activation='tanh', input_shape=(2,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim(0,.001)
plt.title('MSE of 5 NEUR LSTM - Differenced Data CHF/USD, new validation set', size=14) 
plt.savefig('mse_CHF_USD_diff_val1')


# In[ ]:


yhat_diff = model.predict(X_test)


# In[ ]:


yhat_diff = yhat_diff.reshape(len(yhat_diff))
resid = yhat_diff - y_test


# In[ ]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(resid, title='ACF of Residuals - Integrated AR CHF/USD');
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_resid_CHF_diff_val1')


# In[ ]:


yhat_diff = yhat_diff.reshape(len(yhat_diff), 1)

yhat_diff = sc.inverse_transform(yhat_diff)

yhat_diff = yhat_diff.reshape(len(yhat_diff))

yhat_diff = pd.Series(yhat_diff)


yhat_diff.index = test[2]

pred = pd.DataFrame([yhat_diff, sz_diff, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[ ]:


fcst_err = mse(pred.pred[test[2]],pred.actual[test[2]])

fcst_err


# In[ ]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('CHF/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('CHF/USD 2014-2019 Diff, new validation set', size=18)
#plt.text(x='2017-12', y=2.4, s='Test Set MSE :'+str(fcst_err)[:5]+'e-05', bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('chf_usd_diff_fcst_val1')


# ### Validation 2

# In[ ]:


X_train, y_train = X[test[2]], y[test[2]]
X_test, y_test = X[test[1]], y[test[1]]


# In[ ]:


model = Sequential()
model.add(LSTM(5, activation='tanh', input_shape=(2,1)))
model.add(Dropout(rate = 0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
    
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                        verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim(0,.0005)
plt.title('MSE of 5 NEUR LSTM - Differenced Data CHF/USD, new validation set', size=14) 
plt.savefig('mse_CHF_USD_diff_val2')


# In[ ]:


yhat_diff = model.predict(X_test)


# In[ ]:


yhat_diff = yhat_diff.reshape(len(yhat_diff))
resid = yhat_diff - y_test


# In[ ]:


from statsmodels.tsa.stattools import adfuller
adfuller(resid)


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(resid, title='ACF of Residuals - Integrated AR CHF/USD, back testing');
plt.xlabel('lags')
plt.ylabel('autocorrelation')
plt.savefig('acf_resid_CHF_diff_val2')


# In[ ]:


yhat_diff = yhat_diff.reshape(len(yhat_diff), 1)

yhat_diff = sc.inverse_transform(yhat_diff)

yhat_diff = yhat_diff.reshape(len(yhat_diff))

yhat_diff = pd.Series(yhat_diff)


yhat_diff.index = test[1]

pred = pd.DataFrame([yhat_diff, sz_diff, fx.DATE])
pred = pred.T
pred.rename(columns={'Unnamed 0':'pred', 'Unnamed 1': 'actual'}, inplace=True)

pred.DATE = pd.to_datetime(pred.DATE)

pred.set_index('DATE', inplace=True)


# In[ ]:


fcst_err = mse(pred.pred[test[1]],pred.actual[test[1]])

fcst_err


# In[ ]:


plt.figure(figsize=[10,10])
plt.plot(pred.actual)
plt.plot(pred.pred)
plt.legend(['actual', 'forecast'], fontsize=14)
plt.ylabel('CHF/USD', size=12)
plt.xlabel('Observations (Daily)', size=12)
plt.title('CHF/USD 2014-2019 Diff, new validation set', size=18)
#plt.text(x='2017-12', y=2.4, s='Test Set MSE :'+str(fcst_err)[:5]+'e-05', bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('chf_usd_diff_fcst_val2')


# In[29]:


from platform import python_version

print(python_version())

