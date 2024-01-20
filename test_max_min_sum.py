# -*- coding: utf-8 -*-

from carla.data.catalog import OnlineCatalog, CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances, predict_label
import yaml
import pandas as pd
import numpy as np
import time
import pickle
from evaluation import compute_distance, compute_diversity
from evaluation import l0_distance, gower_distance
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

###################################### Online catelog

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
with open('dataset_config.yaml', 'r') as file:
    dataset_config = yaml.safe_load(file)
    
data_name = config['data_name']
instance_num = config['instance_num']
cf_method = config['cf_method']
cfs_num = config['cfs_num']

print("data_name", data_name)
# print("cf_method", cf_method)
print("*"*20)
# # load catalog dataset
dataset_online = OnlineCatalog(data_name)


if data_name in ['adult','heloc']:
    training_params = {"lr": 0.001, "epochs": 50, "batch_size": 1024, "hidden_size": [512,288,144,72,36, 9, 3]} # adult, heloc
    imbalance_rate = 3
if data_name in ['compas']:
    training_params = {"lr": 0.001, "epochs": 10, "batch_size": 32, "hidden_size": [36, 9, 3]} # compas
    imbalance_rate = 11
if data_name in ['give_me_some_credit']:   
    training_params = {"lr": 0.01, "epochs": 10, "batch_size": 1024, "hidden_size": [72, 9, 3]} # give_me_some_credit
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


factuals = predict_negative_instances(ml_model_online, dataset_online.df_test)
# all test instances
df_test_all = dataset_online.df_test
    
df_test_pos = df_test_all[~df_test_all.index.isin(factuals.index)]
    

label_name = dataset_config[data_name]['target']
# df_test_factual = df_test_factual.drop(columns=[label_name])
df_test_pos = df_test_pos.drop(columns=[label_name])
factuals = factuals.drop(columns=[label_name])

print("df_test_pos size", df_test_pos.shape[0])
print("factuals size", factuals.shape[0])


#%% 
df_test_pos = df_test_pos.reindex(columns=factuals.columns)

useless_fea = dataset_config[data_name]['useless']

if useless_fea==False:
    df_test_pos = df_test_pos.drop(columns=useless_fea)
    factuals = factuals.drop(columns=useless_fea)


if data_name in ['adult']:
    thr_0 = 2  
    thr_gower = 0.3
if data_name in ['compas']:
    thr_0 = 2
    thr_gower = 0.3
if data_name in ['give_me_some_credit']:  
    thr_0 = 4
    thr_gower = 0.3
if data_name in ['loan']:
    thr_0 = 2
    thr_gower = 0.3
    
    
dfs_mc_list = [] 
cfs_size_mc_list = []
start_time = time.time()
ins = 0

immutable_fea = dataset_config[data_name]['immutable']

rows_to_add = [] 
for i, (index, fact_row) in enumerate(factuals.iterrows()):
    print("ins num:", i)
    if ins>=instance_num:
        break
    
    mask = (df_test_pos[immutable_fea] == fact_row[immutable_fea]).all(axis=1)
    selected_same_df = df_test_pos[mask]
    
    if selected_same_df.shape[0] < cfs_num: 
        print ("immutable feature continue")
        continue
    
    selected_l0_rows = [pos_row for _, pos_row in selected_same_df.iterrows() if l0_distance(fact_row, pos_row) <= thr_0]
    
    selected_l0_df = pd.DataFrame(selected_l0_rows)
    if selected_l0_df.shape[0] < cfs_num*2:  
        print ("l0 continue")
        continue
    
    mask = selected_l0_df.apply(lambda row: gower_distance(row, fact_row) < thr_gower, axis=1)
    selected_df = selected_l0_df[mask]
    if selected_df.shape[0] < cfs_num: 
        print ("gower continue")
        continue
    
    ins += 1
    rows_to_add.append(fact_row)   
    #%% 
    # max-min
    def max_min(selected_app_df, cfs_num):
    
        U = set(selected_app_df.index) 
        S = {index}  
    
        while len(S) < cfs_num + 1: 
            max_min_distance = -np.inf
            selected_u = None
            
            for u in U:
                if u in S: 
                    continue
                min_distance = np.inf
                for s in S:
                    distance = gower_distance(selected_app_df.loc[u], selected_app_df.loc[s])
                    min_distance = min(min_distance, distance)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    selected_u = u

            S.add(selected_u)
            U.remove(selected_u) 

        S.remove(index)
        
        return S
      
    # max-sum
    def max_sum(selected_app_df, cfs_num):
        U = set(selected_app_df.index)  
        S = {index} 
            
        while len(S) < cfs_num + 1:
            max_sum_distance = -np.inf
            selected_u = None
            
            total_distance = 0
            for u in U:
                if u in S: 
                    continue
                for s in S:
                    total_distance += gower_distance(selected_app_df.loc[u], selected_app_df.loc[s])
                    if total_distance > max_sum_distance:
                        total_distance = total_distance
                        selected_u = u
                        
            S.add(selected_u)
            U.remove(selected_u) 
            
        S.remove(index)
        
        return S
         
    if cf_method in ['max-sum']:
        selected_app_df = selected_df.append(fact_row, ignore_index=False)
        S = max_sum(selected_app_df, cfs_num)
    
        current_mc_df = selected_df.loc[list(S)]
        # print(current_mc_df.shape[0]) 
    
        dfs_mc_list.append(current_mc_df)
        cfs_size_mc_list.append(len(S))
        
    if cf_method in ['max-min']:
        selected_app_df = selected_df.append(fact_row, ignore_index=False)
        S = max_min(selected_app_df, cfs_num)
    
        current_mc_df = selected_df.loc[list(S)]
        # print(current_mc_df.shape[0]) 
    
        dfs_mc_list.append(current_mc_df)
        cfs_size_mc_list.append(len(S))

df_mc_cfs = pd.concat(dfs_mc_list)  
df_test_factual = pd.concat(rows_to_add, axis=1).transpose()



#%%  do evaluation: 
    
print("start evaluation")
result_l0, result_l1, result_gower = compute_distance(df_mc_cfs, df_test_factual, cfs_size_mc_list, cfs_size_mc_list)
result_diversity = compute_diversity(df_mc_cfs, df_test_factual, cfs_size_mc_list, cfs_size_mc_list)
result_time = round((time.time()-start_time)/instance_num, 2)
results_size = {
    "result_size":pd.DataFrame(cfs_size_mc_list, columns=['actual size']),
    "result_l0": result_l0,
    "result_l1": result_l1,
    "result_gower": result_gower,
    "result_diversity": result_diversity,
    "result_time": result_time
}

results_size_mean_values = {}
results_diversity_mean_values = {}

for key, value in results_size.items():
    if isinstance(value, pd.DataFrame):
        results_size_mean_values[key] = round(value.mean().iloc[0], 1)
    elif isinstance(value, list):
        results_size_mean_values[key] = round(sum(value) / len(value), 1)
    else:
        results_size_mean_values[key] = round(value, 3)
       
        
print("diversity_method:", cf_method)     
print("results_size_mean_values:", results_size_mean_values)







