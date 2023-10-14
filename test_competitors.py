# -*- coding: utf-8 -*-

from carla.data.catalog import OnlineCatalog
import carla.recourse_methods.catalog as recourse_catalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances, predict_label
import yaml
import pandas as pd
import random
import time
import pickle
from evaluation import compute_distance, compute_diversity, compute_validity, compute_act_size
import sys
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


###################################### Online catelog

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
with open('dataset_config.yaml', 'r') as file:
    dataset_config = yaml.safe_load(file)
    
data_name = config['data_name']
instance_num = config['instance_num']
cf_method = config['cf_method']

feat_num = config['feat_num']
cfs_num = config['cfs_num']

label_name = dataset_config[data_name]['target']

print("data_name", data_name)
print("cf_method", cf_method)
print("*"*20)
#  load catalog dataset
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
    filename = f'privacy-data/{data_name}_our_instance{instance_num}_imbalance_rate{imbalance_rate}.csv'
    df_train = pd.read_csv(filename, index_col=0)
    df_test_factual = df_train.iloc[:instance_num]
else:
    print("quality test")    
    factuals = predict_negative_instances(ml_model_online, dataset_online.df_test)
    df_test_factual = factuals.iloc[:instance_num]


results_size = {}
results_diversity = {}

#%% DICE
if cf_method=='dice':
    start_time = time.time()
    hyperparams = {"num": cfs_num, "desired_class": "opposite", "posthoc_sparsity_param": 0}
    recourse_method = recourse_catalog.Dice(ml_model_online, hyperparams)
    df_cfs = recourse_method.get_counterfactuals(df_test_factual)
    if privacy==True:
        print("cf of cf!") # int(cfs_num/2+1)
        hyperparams = {"num": cfs_num, "desired_class": "opposite", "posthoc_sparsity_param": 0}
        recourse_method = recourse_catalog.Dice(ml_model_online, hyperparams)
        df_cfs_dual = recourse_method.get_counterfactuals(df_cfs)

#%% FACE: Since FACE does not generate multiple interpretations for an instance at once, then our approach is to generate several more FACE methods by adjusting the parameters.
if cf_method=='face':
    start_time = time.time()
    hyperparams = {"mode": "epsilon", "fraction": 0.05} #  {"mode": "knn", "fraction": 0.05}
    recourse_methods = {} 

    for n in range(cfs_num): # cfs_num
        print("cfs num:", n+1)
        # The size of this value very much affects the time
        hyperparams["fraction"] = random.uniform(0.13, 0.15) 
        recourse_method = recourse_catalog.Face(ml_model_online, hyperparams)
        recourse_methods[f'recourse_method_{n + 1}'] = recourse_method

    all_dfs = []
    for method_name, recourse_method in recourse_methods.items():
        print("method_name num:", method_name)
        desired_class = 1
        df_cfs_tmp = recourse_method.get_counterfactuals(df_test_factual, desired_class)
        all_dfs.append(df_cfs_tmp)

    df_cfs = pd.DataFrame()

    for i in range(instance_num):
        rows_from_batch = [df.iloc[i] for df in all_dfs]
        batch_df = pd.concat(rows_from_batch, axis=1).transpose()
        df_cfs = pd.concat([df_cfs, batch_df], ignore_index=True)
        
    if privacy==True:
        all_dfs = []
        for method_name, recourse_method in recourse_methods.items():
            print("method_name num:", method_name)
            desired_class = 0
            df_cfs_tmp = recourse_method.get_counterfactuals(df_cfs, desired_class)
            all_dfs.append(df_cfs_tmp)
            
        df_cfs_dual = pd.DataFrame()
        for i in range(df_cfs.shape[0]):
            rows_from_batch = [df.iloc[i] for df in all_dfs]
            batch_df = pd.concat(rows_from_batch, axis=1).transpose()
            df_cfs_dual = pd.concat([df_cfs_dual, batch_df], ignore_index=True)


#%% CCHVAE. 
if cf_method=='cchvae':
    start_time = time.time()
    # "n_search_samples": 100
    hyperparams = {"data_name": data_name,"n_search_samples": 20,"p_norm": 1,"step": 0.1,"max_iter": 2000,"clamp": True,
        "binary_cat_features": True,"vae_params": {
            "layers": [len(ml_model_online.feature_input_order), 64, 16, 8], "train": True,"lambda_reg": 1e-6,"epochs": 7,"lr": 1e-3,
            "batch_size": 32,
        },
    }
          
    recourse_methods = {} 
    cchvae_cfs_num = cfs_num #3
    for n in range(cchvae_cfs_num):  
        print("cfs num:", n+1)
        hyperparams["p_norm"] = random.randint(1, 2)
        hyperparams["vae_params"]["epochs"] = random.randint(6, 7) 
        if data_name in ['heloc']:
            hyperparams["vae_params"]["epochs"] = random.randint(5, 7) 
        hyperparams["vae_params"]["lr"] = random.uniform(0.001, 0.003)
        hyperparams["max_iter"] = random.randint(2000, 2300) 
        recourse_method = recourse_catalog.CCHVAE(ml_model_online, hyperparams)
        recourse_methods[f'recourse_method_{n + 1}'] = recourse_method
    
    all_dfs = []
    for method_name, recourse_method in recourse_methods.items():
        print("method_name num:", method_name)
        df_cfs_tmp = recourse_method.get_counterfactuals(df_test_factual)
        all_dfs.append(df_cfs_tmp)

    df_cfs = pd.DataFrame()
    for i in range(instance_num):
        rows_from_batch = [df.iloc[i] for df in all_dfs]
        batch_df = pd.concat(rows_from_batch, axis=1).transpose()
        df_cfs = pd.concat([df_cfs, batch_df], ignore_index=True)
        
    if privacy==True:
        all_dfs = []
        for method_name, recourse_method in recourse_methods.items():
            print("method_name num:", method_name)
            df_cfs_tmp = recourse_method.get_counterfactuals(df_cfs)
            all_dfs.append(df_cfs_tmp)

        df_cfs_dual = pd.DataFrame()
        for i in range(df_cfs.shape[0]):
            rows_from_batch = [df.iloc[i] for df in all_dfs]
            batch_df = pd.concat(rows_from_batch, axis=1).transpose()
            df_cfs_dual = pd.concat([df_cfs_dual, batch_df], ignore_index=True)
        

#%% GS
if cf_method=='gs':
    start_time = time.time()
    all_dfs = []
    for n in range(cfs_num):
        print("cfs num:", n+1)
        recourse_method = recourse_catalog.GrowingSpheres(ml_model_online)
        df_cfs_tmp = recourse_method.get_counterfactuals(df_test_factual)
        all_dfs.append(df_cfs_tmp)

    df_cfs = pd.DataFrame()
    for i in range(instance_num):
        rows_from_batch = [df.iloc[i] for df in all_dfs]
        batch_df = pd.concat(rows_from_batch, axis=1).transpose()
        df_cfs = pd.concat([df_cfs, batch_df], ignore_index=True)
        
    if privacy==True:
        all_dfs = []
        for n in range(cfs_num):
            print("cfs num:", n+1)
            recourse_method = recourse_catalog.GrowingSpheres(ml_model_online)
            df_cfs_tmp = recourse_method.get_counterfactuals(df_cfs)
            all_dfs.append(df_cfs_tmp)

        df_cfs_dual = pd.DataFrame()
        for i in range(df_cfs.shape[0]):
            rows_from_batch = [df.iloc[i] for df in all_dfs]
            batch_df = pd.concat(rows_from_batch, axis=1).transpose()
            df_cfs_dual = pd.concat([df_cfs_dual, batch_df], ignore_index=True)



# %%  生成测试 privacy 的训练集
if privacy==True:
    df_train = pd.concat([df_train, df_cfs, df_cfs_dual], axis=0)
    df_train[label_name] = predict_label(ml_model_online, df_train).astype(int)

    filename = f'privacy-data/{data_name}_{cf_method}_instance{instance_num}_imbalance_rate{imbalance_rate}_cfs{cfs_num}.csv'
    df_train.to_csv(filename, index=True)
    
    sys.exit() 
    
#%% do evaluation: 
print("strat evaluation")
df_test_factual = df_test_factual.drop(columns=[label_name])


cfs_num_list = [cfs_num] * instance_num
cfs_size_list = [cfs_num] * instance_num

if cf_method in ['cchvae']:
    cfs_num_list = [cchvae_cfs_num] * instance_num
    cfs_size_list = [cchvae_cfs_num] * instance_num

result_l0, result_l1, result_gower = compute_distance(df_cfs, df_test_factual, cfs_num_list, cfs_size_list)
result_diversity = compute_diversity(df_cfs, df_test_factual, cfs_num_list, cfs_size_list)
result_validity = compute_validity(ml_model_online, df_cfs, cfs_num_list)
result_time = round((time.time()-start_time)/instance_num, 3)
results_size = {
    "result_l0": result_l0,
    "result_l1": result_l1,
    "result_gower": result_gower,
    "result_diversity": result_diversity,
    "result_validity": result_validity,
    "result_time": result_time
}


cfs_size_list, distinct_columns_counts = compute_act_size(df_cfs, df_test_factual, cfs_num_list, feat_num)

result_l0, result_l1, result_gower = compute_distance(df_cfs, df_test_factual, cfs_num_list, cfs_size_list)
result_diversity = compute_diversity(df_cfs, df_test_factual, cfs_num_list, cfs_size_list)
result_validity = compute_validity(ml_model_online, df_cfs, cfs_num_list)
result_time = round((time.time()-start_time)/instance_num, 3)
results_diversity = {
    "result_size":pd.DataFrame(cfs_size_list, columns=['actual size']),
    "result_l0": result_l0,
    "result_l1": result_l1,
    "result_gower": result_gower,
    "result_diversity": result_diversity,
    "result_validity": result_validity,
    "result_time": result_time
}


filename = f'res/{data_name}_{cf_method}_instance{instance_num}_cfs{cfs_num}_feat{feat_num}.pkl'
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



# %% case study 
# df_test_factual_ori = dataset_online.inverse_transform(df_test_factual.copy())
# df_cfs_ori = dataset_online.inverse_transform(df_cfs.copy())
# df_cfs_dual_ori = dataset_online.inverse_transform(df_cfs_dual.copy())