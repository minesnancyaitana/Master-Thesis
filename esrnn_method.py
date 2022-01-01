#from ESRNN.m4_data import prepare_m4_data
from ESRNN.utils_evaluation import evaluate_prediction_owa

from ESRNN import ESRNN
#X_train_df, y_train_df, X_test_df, y_test_df= m4_parser(dataset_name = 'Daily', directory='/Users/aitanatheret/Desktop/Master thesis/data', num_obs=1000000)
import pandas as pd
import os
import numpy as np


def name_data(path):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.csv'):
                n_key = str(os.sep.join([dirpath, filename]))
                list_of_files[n_key] = pd.read_csv(os.sep.join([dirpath, filename]),parse_dates=["Date"])
    return list_of_files

def prep_data(data):
    for key,value in data.items():
        if "train" in key:
            train  = value
        elif "validation" in key:
            validation = value
        elif 'test' in key:
            test = value
    data_sub = pd.concat([train,validation]) 
    return data_sub,test

def convert_data():
    path = "/Users/aitanatheret/Desktop/data"
    
    data_sub,test = prep_data(name_data(path))
    x_df = pd.DataFrame()
    y_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    
    for name_col in list(data_sub.columns):
        if name_col !='Date':
            dates = data_sub['Date']
            inter_x_df = pd.DataFrame()
            inter_y_df = pd.DataFrame()
            inter_x_df['ds'] = dates
            inter_y_df['ds'] = dates
            id_column = len(dates)*[name_col]
            inter_x_df['unique_id'] = id_column
            inter_y_df['unique_id'] = id_column
            inter_x_df['x'] = len(dates)*['Finance']
            inter_y_df['y'] = data_sub[name_col]
            x_df = x_df.append(inter_x_df)
            y_df  = y_df.append(inter_y_df)
            
    for name_col in list(test.columns):
        if name_col !='Date':
            dates = test['Date']
            inter_x_test_df = pd.DataFrame()
            inter_y_test_df = pd.DataFrame()
            inter_x_test_df['ds'] = dates
            inter_y_test_df['ds'] = dates
            id_column = len(dates)*[name_col]
            inter_x_test_df['unique_id'] = id_column
            inter_y_test_df['unique_id'] = id_column
            inter_x_test_df['x'] = len(dates)*['Finance']
            inter_y_test_df['y'] = test[name_col]
            inter_y_test_df['y_hat_naive2'] = test[name_col]
            x_test_df = x_test_df.append(inter_x_test_df)
            y_test_df  = y_test_df.append(inter_y_test_df)
    
    x_df.reset_index(inplace=True,drop=True)        
    y_df.reset_index(inplace=True,drop=True)
    x_test_df.reset_index(inplace=True,drop=True)
    y_test_df.reset_index(inplace=True,drop=True)
    
    
    x_df=x_df[['unique_id','ds','x']]        
    y_df=y_df[['unique_id','ds','y']]
    x_test_df=x_test_df[['unique_id','ds','x']]
    y_test_df=y_test_df[['unique_id','ds','y','y_hat_naive2']]
    
    return x_df,y_df,x_test_df,y_test_df 

model = ESRNN()

X_train_df, y_train_df, X_test_df, y_test_df= convert_data()

data_2 = X_train_df.copy()
data_2['y'] = y_train_df['y'].copy()
sorted_ds = np.sort(data_2['ds'].unique())
ds_map = {}
for dmap, t in enumerate(sorted_ds):
   ds_map[t] = dmap
data_2['ds_map'] = data_2['ds'].map(ds_map)
data_2 = data_2.sort_values(by=['ds_map','unique_id'])
df_wide = data_2.pivot(index='unique_id', columns='ds_map')['y']

print(y_train_df)
 # Fit model
  # If y_test_df is provided the model will evaluate predictions on this set every freq_test epochs
model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

  # Predict on test set
print('\nForecasting')
y_hat_df = model.predict(X_test_df)


final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train_df,
                                                               X_test_df, y_test_df)









