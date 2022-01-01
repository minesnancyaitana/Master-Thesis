#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:50:51 2021

@author: aitanatheret
"""
from ESRNN.m4_data import prepare_m4_data
from ESRNN.utils_evaluation import evaluate_prediction_owa

from ESRNN import ESRNN

batch_size = 32
window_size = int(256) # must be a multiple of batch_size
validation_size = 45 * batch_size # must be a multiple of batch_size
test_size = 45* batch_size # must be a multiple of batch_size
ma_periods = 14 # Simple Moving Average periods length
ticker = 'gbpusd' # Your data file name without extention
start_date = '2010-01-03' # Ignore any data in the file prior to this date
seed = 42 # An arbitrary value to make sure your seed is the same
model_path = f'models/{ticker}-{batch_size}-{window_size}-{ma_periods}'
scaler_path = f'scalers/{ticker}-{batch_size}-{window_size}-{ma_periods}.bin'
full_time_series_path = f'data/{ticker}.csv'
train_time_series_path = f'data/{ticker}-train.csv'
validate_time_series_path = f'data/{ticker}-validate.csv'
test_time_series_path = f'data/{ticker}-test.csv'

import pandas as pd
import numpy as np 

def get_train(values, window_size):
    X, y = [], []
    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(values[i-window_size:i])
        y.append(values[i])
    X, y = np.asarray(X), np.asarray(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(f"X {X.shape}, y {y.shape}")
    return X, y

def get_val(values, window_size):
    X = []
    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(values[i-window_size:i])
    X = np.asarray(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = values[-X.shape[0]:]
    print(f"X {X.shape}, y {y.shape}")
    return X, y

import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from ESRNN.utils_evaluation import Naive2
from ESRNN.m4_data import naive2_predictions

df_csv = pd.read_csv("/Applications/QuantDataManagerB473.app/Contents/Resources/export/JanToMarch.csv", usecols=['Date','High','Low'], 
    index_col=['Date'], parse_dates=['Date'])
df = df_csv[df_csv.index >= pd.to_datetime(start_date)]
df=df.loc['2010-01-03 ':'2010-01-08']


df["HLAvg"] = df['High'].add(df['Low']).div(2)
# Simple Moving Average
df['MA'] = df['HLAvg'].rolling(window=ma_periods).mean()
# Log Returns
df['Returns'] = np.log(df['MA']/df['MA'].shift(1))
#print(df.head(15))

df.dropna(how='any', inplace=True)
df = df[df.shape[0] % batch_size:]
print(df.head(15))

df2=df.copy()


df_train = df[:- validation_size - test_size]
df_validation = df[- validation_size - test_size - window_size:- test_size]
df_test = df[- test_size - window_size:]

#X_train_de

df2=df_train.copy()
df2=df2.reset_index()
df3=df_train.copy()
df3=df3.reset_index()

X1_train_df = pd.DataFrame(columns=['unique_id','ds','x'], index=range(1,len(df_train)))
X1_train_df['ds']= df2['Date']
X1_train_df['unique_id']= 'Y1'
X1_train_df['x']= 'Finance'

Y1_train_df= pd.DataFrame(columns=['unique_id','ds','y'], index=range(1,len(df_train)))
Y1_train_df['ds']= df2['Date']
Y1_train_df['unique_id']= 'Y1'
Y1_train_df['y']= df2['High']

X1_test_df= pd.DataFrame(columns=['unique_id','ds','x'], index=range(1,len(df_test)))
X1_test_df['ds']= df3['Date']
X1_test_df['unique_id']= 'Y1'
X1_test_df['x']= 'Finance'

Y1_test_df= pd.DataFrame(columns=['unique_id','ds','y'], index=range(1,len(df_test)))
Y1_test_df['ds']= df3['Date']
Y1_test_df['unique_id']= 'Y1'
Y1_test_df['y']= df3['High']



X2_train_df = pd.DataFrame(columns=['unique_id','ds','x'], index=range(1,len(df_train)))
X2_train_df['ds']= df2['Date']
X2_train_df['unique_id']= 'Y2'
X2_train_df['x']= 'Finance'

Y2_train_df= pd.DataFrame(columns=['unique_id','ds','y'], index=range(1,len(df_train)))
Y2_train_df['ds']= df2['Date']
Y2_train_df['unique_id']= 'Y2'
Y2_train_df['y']= df2['Low']

X2_test_df= pd.DataFrame(columns=['unique_id','ds','x'], index=range(1,len(df_test)))
X2_test_df['ds']= df3['Date']
X2_test_df['unique_id']= 'Y2'
X2_test_df['x']= 'Finance'

Y2_test_df= pd.DataFrame(columns=['unique_id','ds','y'], index=range(1,len(df_test)))
Y2_test_df['ds']= df3['Date']
Y2_test_df['unique_id']= 'Y2'
Y2_test_df['y']= df3['Low']

			
				
X3_train_df = pd.DataFrame(columns=['unique_id','ds','x'],	 index=range(1,len(df_train)))
X3_train_df['ds']= df2['Date']				
X3_train_df['unique_id']= 'Y3'				
X3_train_df['x']= 'Finance'				
				
Y3_train_df= pd.DataFrame(columns=['unique_id','ds',	'y'],	 index=range(1,	len(df_train)))
Y3_train_df['ds']= df2['Date']				
Y3_train_df['unique_id']= 'Y3'				
Y3_train_df['y']= df2['MA']				
				
X3_test_df= pd.DataFrame(columns=['unique_id',	'ds',	'x'],	 index=range(1,	len(df_test)))
X3_test_df['ds']= df3['Date']				
X3_test_df['unique_id']= 'Y3'				
X3_test_df['x']= 'Finance'				
				
Y3_test_df= pd.DataFrame(columns=['unique_id',	'ds',	'y'],	 index=range(1,	len(df_test)))
Y3_test_df['ds']= df3['Date']				
Y3_test_df['unique_id']= 'Y3'				
Y3_test_df['y']= df3['MA']				





X_train = pd.concat([X1_train_df, X2_train_df])
X_train = pd.concat([X_train, X3_train_df])

Y_train = pd.concat([Y1_train_df, Y2_train_df])
Y_train = pd.concat([Y_train, Y3_train_df])

Y_test = pd.concat([Y1_test_df, Y2_test_df])
Y_test = pd.concat([Y_test, Y3_test_df])

    

print(naive2_predictions('Hourly' , '/Users/aitanatheret/Desktop/Master thesis/data/',3, y_train_df = Y_train, y_test_df =Y_test))
















