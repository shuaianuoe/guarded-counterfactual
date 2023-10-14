# -*- coding: utf-8 -*-

from carla.data.catalog import OnlineCatalog, CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances, predict_label
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
import numpy as np
import random
import time
import pickle
import lightgbm as lgb

import warnings
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

#%% preprocess
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
with open('dataset_config.yaml', 'r') as file:
    dataset_config = yaml.safe_load(file)
    
data_name = config['data_name']
instance_num = config['instance_num']
cf_method = config['cf_method']
feat_num = config['feat_num']
cfs_num = config['cfs_num']

eva_frac = config['eva_frac']
inference_frac = config['inference_frac']

print("data_name", data_name)
print("cf_method", cf_method)
print("*"*20)
# # load catalog dataset
dataset_online = OnlineCatalog(data_name)

# The first time it would be trained; then the second time it would be straight to the latest saved model
if data_name in ['adult']:
    training_params = {"lr": 0.001, "epochs": 50, "batch_size": 1024, "hidden_size": [512,288,144,72,36, 9, 3]} # adult
    imbalance_rate = 3
if data_name in ['compas']:
    training_params = {"lr": 0.001, "epochs": 10, "batch_size": 32, "hidden_size": [36, 9, 3]} # compas
    imbalance_rate = 11
if data_name in ['give_me_some_credit']:   
    training_params = {"lr": 0.001, "epochs": 50, "batch_size": 1024, "hidden_size": [512,288,144,72,36, 9, 3]} # give_me_some_credit
    imbalance_rate = 10
if data_name in ['loan']:
    training_params = {"lr": 0.01, "epochs": 100, "batch_size": 32, "hidden_size": [36, 9, 3]} # compas
    imbalance_rate = 10
    
    
ml_model_online = MLModelCatalog(
    dataset_online,
    model_type="ann",
    load_online=False,
    backend="pytorch"
)

ml_model_online.train(
    learning_rate=training_params["lr"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"],
    hidden_size=training_params["hidden_size"]
)


# get the training set
if cf_method not in ["dice","cchvae","face","gs"]:
    filename = f'privacy-data/{data_name}_our_instance{instance_num}_imbalance_rate{imbalance_rate}.csv'
else:
    filename = f'privacy-data/{data_name}_{cf_method}_instance{instance_num}_imbalance_rate{imbalance_rate}_cfs{cfs_num}.csv'

print(filename)
df_train = pd.read_csv(filename, index_col=0)
print("the size of df_train before drop_duplicates", df_train.shape[0])
df_train = df_train.drop_duplicates(keep='first')
df_train = df_train.dropna()

# get validation set
filename = f'privacy-data/{data_name}_evaluation.csv'
df_eva = pd.read_csv(filename, index_col=0)
df_eva = df_eva.reindex(columns=df_train.columns)

pred_orimodel = predict_label(ml_model_online, df_eva).astype(int).squeeze()
X_eva = df_eva.iloc[:, :-1]
Y_true_eva = df_eva.iloc[:, -1]

#%% training

X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

# 2. split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.001, random_state=42)

# 3. training	

# LR			
lr = LogisticRegression(max_iter=10) 
lr.fit(X_train, y_train)
					
pred_lr = lr.predict(X_eva)	
				
xgb = XGBClassifier(					
    learning_rate=0.01,					
    n_estimators=100, # 5 100 20					
    max_depth=3, # 2 1					
    subsample=0.1,					
    colsample_bytree=0.8					
)					
xgb.fit(X_train, y_train)										
pred_xgb = xgb.predict(X_eva)					
									
# DNN					
model = tf.keras.models.Sequential([					
    tf.keras.layers.Dense(36, activation='relu', input_shape=(X_train.shape[1],)),					
    # tf.keras.layers.Dropout(0.1), 			
    tf.keras.layers.Dense(9, activation='relu'),					
    # tf.keras.layers.Dropout(0.1),					
    tf.keras.layers.Dense(1, activation='sigmoid')					
])					
					
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])									
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.001, verbose=1)										
pred_dnn = (model.predict(X_eva) > 0.5).astype(int).reshape(-1)					
	
#%% evaluation

lr_fidelity = round(np.sum(pred_lr == pred_orimodel)/df_eva.shape[0], 3)
xgb_fidelity = round(np.sum(pred_xgb == pred_orimodel)/df_eva.shape[0], 3)
dnn_fidelity = round(np.sum(pred_dnn == pred_orimodel)/df_eva.shape[0], 3)

print("lr_fidelity", lr_fidelity)
print("xgb_fidelity", xgb_fidelity)
print("dnn_fidelity", dnn_fidelity)

lr_acc = round(np.sum(pred_lr == Y_true_eva)/df_eva.shape[0], 3)
xgb_acc =round(np.sum(pred_xgb == Y_true_eva)/df_eva.shape[0], 3)
dnn_acc =round(np.sum(pred_dnn == Y_true_eva)/df_eva.shape[0], 3)

print("lr_acc", lr_acc)
print("xgb_acc", xgb_acc)
print("dnn_acc", dnn_acc)












