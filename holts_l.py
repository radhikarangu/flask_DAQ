# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:58:12 2020

@author: RADHIKA
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


# Import CSV file into a dataframe
delhidata=pd.read_excel('D:\DS project Files\Delhi (1).xlsx')
#EDA    
#Index(['date', 'pm25'], dtype='object')
delhidata.head()
delhidata=delhidata.iloc[::-1]
delhidata.head()
delhidata.info()
delhidata.dtypes
delhidata['pm25']  = pd.to_numeric(delhidata['pm25'] ,errors='coerce')
delhidata.dtypes
delhidata.sort_values("date", axis = 0, ascending = True,inplace = True, na_position ='last')
delhidata1 = pd.DataFrame({'date': pd.date_range('2018-01-01', '2018-04-21', freq='1H', closed='left')})
delhidata2 = delhidata1.iloc[:2617,:]
delhidata3 = pd.merge(delhidata,delhidata2,on='date',how='right') 
delhidata3.info()
delhidata3.sort_values("date", axis = 0, ascending = True,inplace = True, na_position ='last')
sns.heatmap(delhidata.isnull(),cbar=True)
delhidata3.head()
delhidata3.tail()
delhidata3.isna().sum()
delhidata3.info()
delhidata3.set_index(['date'],inplace=True)
delhidata3.shape
delhidata3.isnull().sum()
delhidata3_linear=delhidata3.interpolate(method='linear')
delhidata3_linear.isnull().sum()
delhidata3_linear.plot()
delhidata3_linear.shape
delhidata3_linear.plot(figsize=(15,3), color="blue", title='DELHI AIR QUALITY')
delhidata3_linear.hist()
delhidata3_linear.shape
delhidata3_linear.head()
from numpy import log
X = delhidata3_linear.values
X = log(X)
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))#mean1=5.335263, mean2=4.597500
print('variance1=%f, variance2=%f' % (var1, var2))#variance1=0.519288, variance2=0.700707
#ADF test
from statsmodels.tsa.stattools import adfuller

X = delhidata3_linear.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])#-4.057066
print('p-value: %f' % result[1])# 0.001139
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#1%: -3.433
#5%: -2.863
#10%: -2.567
#Rejecting the null hypothesis means that the process has no unit root, and in turn that the
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


Train = delhidata3_linear.head(1873)
Test = delhidata3_linear.tail(744)
Train
Test   
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp) 

fit1 = Holt(delhidata3_linear.pm25).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast1 = fit1.forecast(12).rename("Holt's linear trend")

fit2 = Holt(delhidata3_linear['pm25'], exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast2 = fit2.forecast(12).rename("Exponential trend")

fit3 = Holt(delhidata3_linear['pm25'], damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
fcast3 = fit3.forecast(12).rename("Additive damped trend")
fit1.fittedvalues.plot(marker="o", color='blue')
fcast1.plot(color='blue', marker="o", legend=True)
fit2.fittedvalues.plot(marker="o", color='blue')
fcast2.plot(color='blue', marker="o", legend=True)
fit3.fittedvalues.plot(marker="o", color='blue')
fcast3.plot(color='blue', marker="o", legend=True)
pred_test = fit1.predict(start = Test.index[0],end = Test.index[-1])
pred_train = fit1.predict(start = Train.index[0],end = Train.index[-1])
MAPE_test=MAPE(pred_test,Test.pm25)
MAPE_train=MAPE(pred_train,Train.pm25)
RMSE_test=np.sqrt(np.mean((pred_test-Test.pm25)*(pred_test-Test.pm25)))
RMSE_train=np.sqrt(np.mean((pred_train-Train.pm25)*(pred_train-Train.pm25)))
print("MAPE_test: ",MAPE_test)# 53.22815506192809
print("MAPE_train: ",MAPE_train)#39.84232849702969
print("RMSE_test: ",RMSE_test)#49.59973547822482
print("RMSE_train: ",RMSE_train)#59.88897010645848

import pickle
import sklearn

# Saving model to disk
pickle.dump(fit1, open('holts_l.pkl','wb'))
fit1.forecast(steps=24)
