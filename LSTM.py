# MIT License
# Copyright (c) 2020 Adam Tibi (https://linkedin.com/in/adamtibi/ , https://adamtibi.net)

batch_size = 32
window_size = int(256) # must be a multiple of batch_size
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


batch_size = 32
window_size = int(256) # must be a multiple of batch_size
validation_size = 45 * batch_size # must be a multiple of batch_size
test_size = 18 # must be a multiple of batch_size
ma_periods = 14 # Simple Moving Average periods length
ticker = 'gbpusd' # Your data file name without extention
start_date = '2010-01-01' # Ignore any data in the file prior to this date
seed = 42 # An arbitrary value to make sure your seed is the same
# model_path = f'models/{ticker}-{batch_size}-{window_size}-{ma_periods}'
# scaler_path = f'scalers/{ticker}-{batch_size}-{window_size}-{ma_periods}.bin'
# full_time_series_path = f'/Users/aitanatheret/Desktop/data/{ticker}.csv'Adam
# train_time_series_path = f'data/{ticker}-train.csv'
# validate_time_series_path = f'data/{ticker}-validate.csv'
# test_time_series_path = f'data/{ticker}-test.csv'



epochs = 50
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv("/Users/aitanatheret/Desktop/data/EURUSD_train.csv", dayfirst=True, index_col=['Date'], parse_dates=['Date'])
print(df)

# fig = plt.figure(figsize=(24, 18))
# ax1, ax2, ax3 = fig.subplots(3)
# ax1.set_title('HLAvg')
# ax1.set(xlabel='Date', ylabel='High-Low Average')
# ax1.plot(df['HLAvg'])
# ax2.set_title('MA')
# ax2.set(xlabel='Date', ylabel='MA')
# ax2.plot(df['MA'])
# ax3.set_title('MidEURUSD')
# ax3.set(xlabel='Date', ylabel='MidEURUSD')
# ax3.plot(df['MidEURUSD'])

scaler = MinMaxScaler(feature_range=(0,1))
train_values = scaler.fit_transform(df[['MidEURUSD']].values)

# fig = plt.figure(figsize=(24, 8))
# ax1 = fig.subplots(1)
# ax1.set_title('MidEURUSD MinMax Scaled')
# ax1.set(xlabel='Sample', ylabel='Scaled MidEURUSD')
# ax1.plot(train_values)

X,y = get_train(train_values, window_size)

tf.autograph.set_verbosity(3, True)

model = Sequential()
model.add(LSTM(76, input_shape=(X.shape[1], 1), return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(1))
optimizer = tf.keras.optimizers.Adam()
model.compile(loss="mse", optimizer=optimizer)

print(model.summary())

df_val = pd.read_csv("/Users/aitanatheret/Desktop/data/EURUSD_validation.csv", dayfirst = True, usecols=['Date','MidEURUSD'],
    index_col=['Date'], parse_dates=['Date'])
df_val['Scaled'] = scaler.transform(df_val[['MidEURUSD']].values)
X_val, y_val = get_val(df_val['Scaled'].values, window_size)

history = model.fit(X, y, validation_data=(X_val, y_val), epochs = epochs, batch_size = batch_size, shuffle=False, verbose = 2)

#model.save("/Utilisateurs/aitanatheret/Bureau/Master thesis/Python/Tentative1")
#joblib.dump(scaler, "/Utilisateurs/aitanatheret/Bureau/Master thesis/Python") 

fig = plt.figure(figsize=(12, 8))
ax1 = fig.subplots(1)
ax1.set_title('Model Loss')
ax1.set(xlabel='Epoch', ylabel='Loss')
ax1.plot(history.history['loss'][7:], label='Train Loss')
ax1.plot(history.history['val_loss'][7:], label='Val Loss')
ax1.legend()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv("/Users/aitanatheret/Desktop/data/EURUSD_test.csv", dayfirst = True, index_col=['Date'], parse_dates=['Date'], usecols=['Date','HLAvg','MA','MidEURUSD'])



#scaler = joblib.load(scaler_path)
df['Scaled'] = scaler.transform(df[['MidEURUSD']].values)

#model = load_model(model_path)

scaled = df['Scaled'].values
X = []
len_scaled = len(scaled)
for i in range(window_size, len_scaled):
    X.append(scaled[i-window_size:i])
X = np.asarray(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y_true = scaled[-X.shape[0]:]
print(f"X {X.shape}, y_real {y_true.shape}")

mse = model.evaluate(X, y_true, verbose=1)
print("Mean Squared Error:", mse)

y_pred = model.predict(X)
y_pred.shape

df['Pred_Scaled'] = np.pad(y_pred.reshape(y_pred.shape[0]), (window_size, 0), mode='constant', constant_values=np.nan)
df['Pred_MidEURUSD'] = scaler.inverse_transform(df[['Pred_Scaled']].values)
#df['Pred_MA'] = df["MA"].mul(1 + df['Pred_MidEURUSD'].shift(-1)).shift(1) # Arithmetic MidEURUSD
df['Pred_MA'] = df['MA'].mul(np.exp(df['Pred_MidEURUSD'].shift(-1))).shift(1) # Log MidEURUSD
df = df[window_size:]
df

plt.figure(figsize=(24, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df['HLAvg'][-20:], color = 'green', label = 'True HLAvg', alpha=0.5)
plt.plot(df['MA'][-20:], color = 'blue', label = 'True MA', alpha=0.5)
plt.plot(df['Pred_MA'][-20:], color = 'red', label = 'Predicted', alpha=0.5)
plt.title('True vs Predicted')
plt.xlabel('Date')
plt.ylabel('High-Low Avg')
plt.legend()
plt.show()

pred_interval = 128 # Predict every n minutes
pred_size = 30 # Prediction length into the future
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv("/Applications/QuantDataManagerB473.app/Contents/Resources/export/EURUSD_test.csv", dayfirst = True, index_col=['Date'], parse_dates=['Date'], usecols=['Date','HLAvg','MA','MidEURUSD'])

scaler = MinMaxScaler(feature_range=(0,1))
df['Scaled'] = scaler.fit_transform(df[['MidEURUSD']].values)

scaled = df['Scaled'].values

# Create empty column to store the multi predictions
df["Pred_Close_From"] =  np.nan
df["Pred_Close_To"] = np.nan

# Cache the column indices 
pred_close_from_col_index = df.columns.get_loc('Pred_Close_From')
pred_close_to_col_index = df.columns.get_loc('Pred_Close_To')

ma_col_index = df.columns.get_loc('HLAvg')
predictions_for_plot = []
df_len = df.shape[0]
for i in range(window_size, df_len - pred_size, pred_interval) :
    X = [scaled[i-window_size:i]]
    y = []
    y_ma = df.iloc[i - 1, ma_col_index]
    for _ in range(pred_size):
        X = np.asarray(X)
        X = np.reshape(X, (1, window_size, 1))
        y_pred_scaled = model.predict(X)
        y_return = scaler.inverse_transform(y_pred_scaled)
        #y_ma = y_ma * (1 + y_return) # Arithmetic MidEURUSD
        y_ma = y_ma * np.exp(y_return) # Log MidEURUSD
        y.append(float(y_ma))
        # Remove first item in the list
        X = np.delete(X, 0)
        # Add the new prediction to the end
        X = np.append(X, y_pred_scaled)

    df.iloc[i, pred_close_from_col_index] = y[0]
    df.iloc[i, pred_close_to_col_index] = y[-1]
    y_padded = np.pad(y, (i, df_len - pred_size - i), mode='constant', constant_values=np.nan)
    df_plot = pd.Series(data=y_padded,index=df.index)

    predictions_for_plot.append(df_plot)

df












