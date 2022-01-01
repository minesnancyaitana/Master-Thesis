#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 16:32:37 2021

@author: aitanatheret
"""
# MIT License
# Copyright (c) 2020 Adam Tibi (https://linkedin.com/in/adamtibi/ , https://adamtibi.net)

batch_size = 32
window_size = 0 # must be a multiple of batch_size
validation_size = 45 * batch_size # must be a multiple of batch_size
test_size = 18 # must be a multiple of batch_size
ma_periods = 14 # Simple Moving Average periods length
ticker = 'gbpusd' # Your data file name without extention
start_date = '2010-01-01' # Ignore any data in the file prior to this date
seed = 42 # An arbitrary value to make sure your seed is the same
model_path = f'models/{ticker}-{batch_size}-{window_size}-{ma_periods}'
scaler_path = f'scalers/{ticker}-{batch_size}-{window_size}-{ma_periods}.bin'
full_time_series_path = f'data/{ticker}.csv'
train_time_series_path = f'data/{ticker}-train.csv'
validate_time_series_path = f'data/{ticker}-validate.csv'
test_time_series_path = f'data/{ticker}-test.csv'

import pandas as pd
import numpy as np

# def get_train(values, window_size):
#     X, y = [], []
#     len_values = len(values)
#     for i in range(window_size, len_values):
#         X.append(values[i-window_size:i])
#         y.append(values[i])
#     X, y = np.asarray(X), np.asarray(y)
#     X = np.reshape(X, (X.shape[0], X.shape[1], 1))
#     print(f"X {X.shape}, y {y.shape}")
#     return X, y

# def get_val(values, window_size):
#     X = []
#     len_values = len(values)
#     for i in range(window_size, len_values):
#         X.append(values[i-window_size:i])
#     X = np.asarray(X)
#     X = np.reshape(X, (X.shape[0], X.shape[1], 1))
#     y = values[-X.shape[0]:]
#     print(f"X {X.shape}, y {y.shape}")
#     return X, y



dfEU_csv = pd.read_csv("/Users/aitanatheret/Desktop/W1EURUSD-M1-NoSession.csv",
                       usecols=['Date','High','Low'], index_col=['Date'], parse_dates=['Date'])
dfEU = dfEU_csv[dfEU_csv.index >= pd.to_datetime(start_date)]
dfEU=dfEU.loc['2010-01-03 ':'2010-01-08']
print(dfEU)

dfEU["MidEURUSD"] = dfEU['High'].add(dfEU['Low']).div(2)
del dfEU['High']
del dfEU['Low']

dfEG_csv = pd.read_csv("/Users/aitanatheret/Desktop/W1EURGBP-M1-NoSession.csv",
                       usecols=['Date','High','Low'], index_col=['Date'], parse_dates=['Date'])
dfEG = dfEG_csv[dfEG_csv.index >= pd.to_datetime(start_date)]
dfEG=dfEG.loc['2010-01-03 ':'2010-01-08']
print(dfEG)

dfEG["MidEURGBP"] = dfEG['High'].add(dfEG['Low']).div(2)
print(dfEG)
del dfEG['High']
del dfEG['Low']

dfEY_csv = pd.read_csv("/Users/aitanatheret/Desktop/W1EURJPY-M1-NoSession.csv",
                       usecols=['Date','High','Low'], index_col=['Date'], parse_dates=['Date'])
dfEY = dfEY_csv[dfEY_csv.index >= pd.to_datetime(start_date)]
dfEY=dfEY.loc['2010-01-03 ':'2010-01-08']
print(dfEY)

dfEY["MidEURYJP"] = dfEY['High'].add(dfEY['Low']).div(2)

del dfEY['High']
del dfEY['Low']
print(dfEY)


df = pd.concat([dfEU, dfEG], axis=1)
df = pd.concat([df, dfEY], axis=1)
print(df)

df.dropna(how='any', inplace=True)
df = df[df.shape[0] % batch_size:]
print(df)

df_train = df[:- validation_size - test_size]
df_validation = df[- validation_size - test_size - window_size:- test_size]
df_test = df[- test_size - window_size:]
#print(f'df_train.shape {df_train.shape}, df_validation.shape {df_validation.shape},
# df_test.shape {df_test.shape}')

df_train.to_csv("/Users/aitanatheret/Desktop/data/EURUSD_train.csv")
df_validation.to_csv("/Users/aitanatheret/Desktop/data/EURUSD_validation.csv")
df_test.to_csv("/Users/aitanatheret/Desktop/data/EURUSD_test.csv")






















