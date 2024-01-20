# -*- coding: utf-8 -*-

from carla.data.catalog import OnlineCatalog, CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances, predict_label
import yaml
import pandas as pd
import time
from evaluation import compute_distance, compute_diversity
from evaluation import l0_distance, gower_distance
import warnings

from stream_alg import stream_sDBC,stream_dSBC, gsv, gsv_postpro, gops,compute_largest_T,dr_post,dr

warnings.filterwarnings("ignore", category=UserWarning)


###################################### Online catelog
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
with open('dataset_config.yaml', 'r') as file:
    dataset_config = yaml.safe_load(file)
    
data_name = config['data_name']
instance_num = config['instance_num']
cf_method = config['cf_method']
# GSV
epsilon = config['epsilon']
# GOPS
beta = config['beta']
feat_num = config['feat_num']
cfs_num = config['cfs_num']

sDBC = True

print("data_name", data_name)
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


factuals = predict_negative_instances(ml_model_online, dataset_online.df_test)
# all test instances
df_test_all = dataset_online.df_test 
df_test_pos = df_test_all[~df_test_all.index.isin(factuals.index)]
    
label_name = dataset_config[data_name]['target']
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
dfs_sc_list = [] 

cfs_size_mc_list = []
cfs_size_sc_list = [] 

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


df_test_factual = pd.concat(rows_to_add, axis=1).transpose()


#%%

print("start streaming!")

dfs_mc_list = [] 
dfs_sc_list = [] 

cfs_size_mc_list = []
cfs_size_sc_list = [] 

start_time = time.time()

W_size = 0
cur_res_size = 0

    
# fact_row is x0; stream_fact_row is xt
for i, (index, fact_row) in enumerate(df_test_factual.iterrows()):
    print("ins num:", i)
    
    mc_res = []
    sc_res = []
    eff_dict = {key: -1 for key in df_test_pos.columns}
    
    # for  GSV algorithm
    C = set()
    I = set()
    W = {}
    # for GOPS algorithm
    cur_res = {}
    # for GS algorithm
    subset_dict = {}

    for (stream_index, stream_fact_row) in df_test_pos.iterrows():

        if any(stream_fact_row[fea] != fact_row[fea] for fea in immutable_fea):
            continue

        if l0_distance(stream_fact_row, fact_row) > thr_0:
            continue
    
        if gower_distance(stream_fact_row, fact_row) > thr_gower:
            continue
         
        if cf_method in ['our']:
            # streaming sDBC
            if len(mc_res) < cfs_num:
                mc_res.append(stream_index)
    
            mc_res = stream_sDBC(mc_res, df_test_pos, fact_row, stream_fact_row, cfs_num, stream_index)
        
        #  Test GSV algorithm compared to streaming sDBC   
        if cf_method in ['gsv']:
            W, mc_res = gsv(fact_row, stream_fact_row, cfs_num, stream_index, mc_res, W, C, epsilon)

        # Test GOPS algorithm compared to streaming sDBC    
        if cf_method in ['gops']:
            if len(cur_res) < cfs_num:
                  S  = set(fact_row.index[fact_row != stream_fact_row])
                  cur_res[stream_index] = S
                  continue
            cur_res = gops(fact_row, stream_fact_row, cur_res, cfs_num, stream_index, beta)
        
        if sDBC == True:
            if cf_method in ['our']:
                T = compute_largest_T(stream_fact_row, fact_row, eff_dict)
                if len(T) < 1:
                    continue    
                sc_res, eff_dict = stream_dSBC(sc_res, df_test_pos, fact_row, feat_num, stream_index, eff_dict, T)
            if cf_method in ['dr']:                  
                sc_res, diff_columns = dr(sc_res, stream_fact_row, fact_row,df_test_pos,stream_index)
                subset_dict[stream_index] = diff_columns
                
    if cf_method in ['gsv']:
        # Post-processing
        mc_res = gsv_postpro(W, cfs_num, mc_res)
        W_size += len(W)    
    if cf_method in ['gops']:
        # Post-processing
        sorted_keys_by_value_len = sorted(cur_res, key=lambda k: len(cur_res[k]), reverse=True)
        # Select the top cfs_num keys
        mc_res = sorted_keys_by_value_len[:cfs_num]
        cur_res_size += sum(1 for key, value in cur_res.items() if value)
    if cf_method in ['dr']: 
        sc_res = dr_post(sc_res, subset_dict, df_test_pos,feat_num)
     
       
    current_mc_df = df_test_pos.loc[mc_res]
    dfs_mc_list.append(current_mc_df)
    cfs_size_mc_list.append(len(mc_res))    
    
    if sDBC == True:
        current_sc_df = df_test_pos.loc[sc_res]
        dfs_sc_list.append(current_sc_df)
        cfs_size_sc_list.append(len(sc_res))
        df_sc_cfs = pd.concat(dfs_sc_list) 
    
    
df_mc_cfs = pd.concat(dfs_mc_list) 
if sDBC == True:
    df_sc_cfs = pd.concat(dfs_sc_list)

#%%  do evaluation: 
print("start evaluation")
result_time = round((time.time()-start_time)/instance_num, 4)
result_l0, result_l1, result_gower = compute_distance(df_mc_cfs, df_test_factual, cfs_size_mc_list, cfs_size_mc_list)
result_diversity = compute_diversity(df_mc_cfs, df_test_factual, cfs_size_mc_list, cfs_size_mc_list)
results_size = {
    "result_size":pd.DataFrame(cfs_size_mc_list, columns=['actual size']),
    "result_l0": result_l0,
    "result_l1": result_l1,
    "result_gower": result_gower,
    "result_diversity": result_diversity,
    "result_time": result_time
}

results_size_mean_values = {}

for key, value in results_size.items():
    if isinstance(value, pd.DataFrame):
        results_size_mean_values[key] = round(value.mean().iloc[0], 1)
    elif isinstance(value, list):
        results_size_mean_values[key] = round(sum(value) / len(value), 1)
    else:
        results_size_mean_values[key] = round(value, 3)
print("results_size_mean_values:", results_size_mean_values)


if sDBC == True:
    result_l0, result_l1, result_gower = compute_distance(df_sc_cfs, df_test_factual, cfs_size_sc_list, cfs_size_sc_list)
    result_diversity = compute_diversity(df_sc_cfs, df_test_factual, cfs_size_sc_list, cfs_size_sc_list)
    # result_time = round((time.time()-start_time)/instance_num, 2)
    results_diversity = {
        "result_size":pd.DataFrame(cfs_size_sc_list, columns=['actual size']),
        "result_l0": result_l0,
        "result_l1": result_l1,
        "result_gower": result_gower,
        "result_diversity": result_diversity,
        # "result_time": result_time
    }
    results_diversity_mean_values = {}
    
    for key, value in results_diversity.items():
        if isinstance(value, pd.DataFrame):
            results_diversity_mean_values[key] = round(value.mean().iloc[0], 1)
        elif isinstance(value, list):
            results_diversity_mean_values[key] = round(sum(value) / len(value), 1)
        else:
            results_diversity_mean_values[key] = round(value, 3)
    print("results_diversity_mean_values:", results_diversity_mean_values)