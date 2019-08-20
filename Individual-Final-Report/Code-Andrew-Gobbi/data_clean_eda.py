
# coding: utf-8

# In[ ]:


####### FX Data Clean and PreProcessing #########
###### Drew Gobbi 07-27-19 ######
##### Machine Learning 1 #######


# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
#from arch.unitroot import ADF 


# In[ ]:


#load data
#all data accessible here - https://fred.stlouisfed.org/categories/158
bz = pd.read_csv('/home/jovyan/Machine-Learning-/DATA/raw data/DEXBZUS.csv')
ch = pd.read_csv('/home/jovyan/Machine-Learning-/DATA/raw data/DEXCHUS.csv')
sf = pd.read_csv('/home/jovyan/Machine-Learning-/DATA/raw data/DEXSFUS.csv')
sz = pd.read_csv('/home/jovyan/Machine-Learning-/DATA/raw data/DEXSZUS.csv')


# In[ ]:


bz.rename(columns = {'VALUE' : 'bz'} , inplace = True)
ch.rename(columns = {'VALUE' : 'ch'} , inplace = True)
sf.rename(columns = {'DEXSFUS' : 'sf'} , inplace = True)
sz.rename(columns = {'DEXSZUS' : 'sz'} , inplace = True)


# In[ ]:


#make one dataframe for convenience
fx = bz.merge(ch)
fx = fx.merge(sf)
fx = fx.merge(sz)
fx.head()


# In[ ]:


#fix datatypes
fx.replace('.', np.NaN, inplace=True)
fx.DATE = pd.to_datetime(fx.DATE)
fx.bz = fx.bz.astype(float)
fx.ch = fx.ch.astype(float)
fx.sf = fx.sf.astype(float)
fx.sz = fx.sz.astype(float)
fx.dtypes


# In[ ]:


#set index
fx.set_index('DATE', inplace=True)


# In[ ]:


fx.interpolate(method='linear', inplace = True)


# In[ ]:


sc = MinMaxScaler(feature_range = (0, 1))


# In[ ]:


#scale and format
sc = MinMaxScaler(feature_range = (0, 1))
fx_scl = sc.fit_transform(fx[['bz', 'ch', 'sf', 'sz']])
fx_scl = pd.DataFrame(fx_scl)
fx_scl.set_index(fx.index, inplace=True)
for i in range(len(fx_scl.columns)):
    fx_scl.rename(columns = {fx_scl.columns[i]: fx.columns[i]}, inplace=True)
fx = fx_scl


# In[ ]:


fx.plot()


# In[ ]:


#check seasonality 
bz_dcmps = sm.tsa.seasonal_decompose(fx.bz, model='additive')
ch_dcmps = sm.tsa.seasonal_decompose(fx.ch, model='additive')
sf_dcmps = sm.tsa.seasonal_decompose(fx.sf, model='additive')
sz_dcmps = sm.tsa.seasonal_decompose(fx.sz, model='additive')

fx_dcmps = pd.DataFrame([bz_dcmps.resid,ch_dcmps.resid, sf_dcmps.resid, sz_dcmps.resid])
fx_dcmps = fx_dcmps.head().T.dropna()


# In[ ]:


#additive seasonal decomposition is picking up a seasonal trend every few days across
#all series. This suggests to me that all the decomposition is picking up is variance and 
#not a true seasonal term
fx_dcmps[0:30].plot(title='intramonth seasonality')


# In[ ]:


fx.reset_index(inplace=True)


# In[ ]:


#check if trend stationary
resbz = smf.ols(formula='fx.bz ~ fx.index', data=fx).fit()
resch = smf.ols(formula='fx.ch ~ fx.index', data=fx).fit()
ressf = smf.ols(formula='fx.sf ~ fx.index', data=fx).fit()
ressz = smf.ols(formula='fx.sz ~ fx.index', data=fx).fit()


# In[ ]:


#bz
plt.plot(fx.bz)
plt.plot(resbz.predict())
plt.title('linear trend vs observed bz')
plt.show()

#ch
plt.plot(fx.ch)
plt.plot(resch.predict())
plt.title('linear trend vs observed ch')
plt.show()

#sf
plt.plot(fx.sf)
plt.plot(ressf.predict())
plt.title('linear trend vs observed sf')
plt.show()

#sz
plt.plot(fx.sz)
plt.plot(ressf.predict())
plt.title('linear trend vs observed sz')
plt.show()

#linear regression does not appear to show a linear trend of which any of the series 
#revolves around. This suggests that the sereis is not trend stationary and powered by 
#a deterministic mean


# In[ ]:


#check for difference stationarity
tsaplots.plot_acf(fx.bz, title = 'ACF of BZ');
print('ADF BZ', ADF(fx.bz))
tsaplots.plot_acf(fx.ch, title = 'ACF of CH');
print('ADF CH', ADF(fx.ch))
tsaplots.plot_acf(fx.sf, title = 'ACF of SF');
print('ADF SF', ADF(fx.sf))
tsaplots.plot_acf(fx.sz, title = 'ACF of SZ');
print('ADF SZ', ADF(fx.sz))


# In[ ]:


#check statinarity in first differences
tsaplots.plot_acf(fx.bz.diff(1).dropna(), title = 'ACF of BZ first diff');
print('ADF BZ first diff', ADF(fx.bz.diff(1).dropna()))
tsaplots.plot_acf(fx.ch.diff(1).dropna(), title = 'ACF of CH first diff');
print('ADF CH first diff', ADF(fx.ch.diff(1).dropna()))
tsaplots.plot_acf(fx.sf.diff(1).dropna(), title = 'ACF of SF first diff');
print('ADF SF first diff', ADF(fx.sf.diff(1).dropna()))
tsaplots.plot_acf(fx.sz.diff(1).dropna(), title = 'ACF of SZ first diff');
print('ADF SZ first diff', ADF(fx.sz.diff(1).dropna()))
#appears the series is stationary in first differences from a view of the acf of the first
#differenced acf plots and ADF tests 


# In[ ]:


fx.set_index('DATE', inplace=True)


# In[ ]:


fx_diff = fx.diff(1).dropna()
fx_diff.head()


# In[ ]:


fx.to_csv(path_or_buf='/Users/drewgobbi/Documents/FX/DATA/fx.csv')
fx_diff.to_csv(path_or_buf='/Users/drewgobbi/Documents/FX/DATA/fx_diff.csv')


# In[ ]:


#notes on data processing
'''
This file leverages methods in the Forex with ANN book by Yu et al. 

Key features:
* Collects all data and stores in single dataframe for ease of use. We will probably
have to turn into arrays for KERAS, but not sure yet (this is easy to do in pandas)
* normalizes data ranges from zero to one 
* imputes missing values with linear interpolation
* checks for seasonality and determines that there is not a seasonal component to remove 
from the series
* checks for difference vs trend stationarity and find the series to be difference 
stationary across first differences

Exports:
*normalized fx dataframe of all 4 time series to be used in the analysis phase. stored in
dataframe for easy handling, but will be converted to numpy array for analysis 
*normalized and differenced fx dataframe. Yu et al. suggest training the network on the 
stationary series for obtaining better  results. Intuitively, this seems to make sense
and is similar to including an integrated term
'''

