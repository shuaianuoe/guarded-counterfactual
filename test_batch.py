# -*- coding: utf-8 -*-

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import yaml
import pandas as pd

import time
import pickle
from evaluation import compute_distance, compute_diversity
from evaluation import l0_distance, gower_distance
import warnings

from batch_alg import greedy_maximum_coverage, greedy_set_cover
import sys

warnings.filterwarnings("ignore", category=UserWarning)

###################################### Online catelog

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
with open('dataset_config.yaml', 'r') as file:
    dataset_config = yaml.safe_load(file)
    
data_name = config['data_name']
instance_num = config['instance_num']

# the number of required features, i.e., diversity. i.e., given a diversity, show the size, similarity, sparsity, validity.
feat_num = config['feat_num']
# the number of cfs needed, i.e., size. i.e., given a size, then go through the individual methods show the diversity, similarity, sparsity, validity.
cfs_num = config['cfs_num']

print("data_name", data_name)
print("*"*20)
# load catalog dataset
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

privacy = config['privacy']
eva_frac = config['eva_frac']
inference_frac = config['inference_frac']

if privacy==True:
    print("privacy test")
    df_test_all = dataset_online.df_test.iloc[:int(eva_frac*len(dataset_online.df_test))]
    factuals = predict_negative_instances(ml_model_online, df_test_all)

else:
    print("quality test")
    factuals = predict_negative_instances(ml_model_online, dataset_online.df_test)
    df_test_all = dataset_online.df_test
    
# Find the rows in the test set that are predicted to be positive   
df_test_pos = df_test_all[~df_test_all.index.isin(factuals.index)]
label_name = dataset_config[data_name]['target']
df_test_pos = df_test_pos.drop(columns=[label_name])
factuals = factuals.drop(columns=[label_name])

print("df_test_pos size", df_test_pos.shape[0])
print("factuals size", factuals.shape[0])
# %%  Generate a training set for testing privacy
if privacy==True:

    filename = f'privacy-data/{data_name}_our_instance{instance_num}_imbalance_rate{imbalance_rate}.csv'
    factuals = factuals.sample(frac=inference_frac, random_state=42)  # 增加生成结果的随机性
     
    factuals = factuals.sample(n=instance_num, random_state=42)
    df_test_pos = df_test_pos.sample(n=min(len(df_test_pos), int(instance_num*imbalance_rate)), random_state=42)
    factuals[label_name] = 0
    df_test_pos[label_name] = 1
    df_train = pd.concat([factuals, df_test_pos], axis=0)
    df_train.to_csv(filename, index=True)
    
    # the validation set gives a save as well.
    df_eva_all = dataset_online.df_test.iloc[int(eva_frac*len(dataset_online.df_test)):]

    filename = f'privacy-data/{data_name}_evaluation.csv'
    df_eva_all.to_csv(filename, index=True)

    sys.exit() 


#%% Step 1: First find all the samples in the test set that satisfy the various thresholds for x
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
dfs_sc_list = [] 

cfs_size_mc_list = []
cfs_size_sc_list = [] 
start_time = time.time()
ins = 0

immutable_fea = dataset_config[data_name]['immutable']

rows_to_add = []  # collect rows to be added

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
    
    #%% Step 2: Calculate the content of the features that are not the same for each sample and x in the test set obtained in the first step.
    subset_dict = {}
    
    for idx, row in selected_df.iterrows():
        diff_columns = row[row != fact_row].index.tolist()
        subset_dict[idx] = diff_columns
           
    all_values_union = set().union(*[set(value) for value in subset_dict.values()])
    if len(all_values_union) < feat_num:
        print ("diversity continue")
        continue
    
    ins += 1
    rows_to_add.append(fact_row)
        
    #%% 3.1, run dSBC
    universe = set(df_test_pos.columns)
    subset_copy_dict = subset_dict.copy()
    mc_res = greedy_maximum_coverage(universe, subset_copy_dict, cfs_num)
    current_mc_df = selected_df.loc[mc_res]

    dfs_mc_list.append(current_mc_df)
    cfs_size_mc_list.append(len(mc_res))
    
    #%% 3.2, run sDBC
    subset_copy_dict = subset_dict.copy()
    sc_res = greedy_set_cover(universe, subset_copy_dict, feat_num)

    current_sc_df = selected_df.loc[sc_res]
    dfs_sc_list.append(current_sc_df)
    cfs_size_sc_list.append(len(sc_res))
    
df_mc_cfs = pd.concat(dfs_mc_list)  # , ignore_index=True
df_sc_cfs = pd.concat(dfs_sc_list) #, ignore_index=True
df_test_factual = pd.concat(rows_to_add, axis=1).transpose()


#%%  do evaluation: 
print("strat evaluation")

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


result_l0, result_l1, result_gower = compute_distance(df_sc_cfs, df_test_factual, cfs_size_sc_list, cfs_size_sc_list)
result_diversity = compute_diversity(df_sc_cfs, df_test_factual, cfs_size_sc_list, cfs_size_sc_list)
result_time = round((time.time()-start_time)/instance_num, 2)
results_diversity = {
    "result_size":pd.DataFrame(cfs_size_sc_list, columns=['actual size']),
    "result_l0": result_l0,
    "result_l1": result_l1,
    "result_gower": result_gower,
    "result_diversity": result_diversity,
    "result_time": result_time
}

# save
filename = f'res/{data_name}_our_instance{instance_num}_cfs{cfs_num}_feat{feat_num}.pkl'
with open(filename, 'wb') as file:
    pickle.dump((results_size, results_diversity), file)

print(f"Results saved to {filename}")

results_size_mean_values = {}
results_diversity_mean_values = {}

for key, value in results_size.items():
    if isinstance(value, pd.DataFrame):
        results_size_mean_values[key] = round(value.mean().iloc[0], 1)
    elif isinstance(value, list):
        results_size_mean_values[key] = round(sum(value) / len(value), 1)
    else:
        results_size_mean_values[key] = round(value, 3)
print("dSBC res:")
print("results_size_mean_values:", results_size_mean_values)
print("*"*20)

for key, value in results_diversity.items():
    if isinstance(value, pd.DataFrame):
        results_diversity_mean_values[key] = round(value.mean().iloc[0], 1)
    elif isinstance(value, list):
        results_diversity_mean_values[key] = round(sum(value) / len(value), 1)
    else:
        results_diversity_mean_values[key] = round(value, 3)
print("sDBC res:")
print("results_diversity_mean_values:", results_diversity_mean_values)


#%%
# # Retrieve the dataset. Here is the code to get the original eigenvalues for the case study analysis
# factuals = predict_negative_instances(ml_model_online, dataset_online.df_test)
# df_test_factual1 = factuals.loc[df_test_factual.index]

# df_test_pos = df_test_all[~df_test_all.index.isin(factuals.index)]
# df_mc_cfs1 = df_test_pos.loc[df_mc_cfs.index]

# df_test_factual_ori = dataset_online.inverse_transform(df_test_factual1.copy())
# df_cfs_ori = dataset_online.inverse_transform(df_mc_cfs1.copy())

# df_test_factual_ori = df_test_factual_ori.drop(columns=[label_name])
# df_cfs_ori = df_cfs_ori.drop(columns=[label_name])




