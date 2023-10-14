# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from gower import gower_matrix


# Define the L0 distance function
def l0_distance(row1, row2):
    # return sum(row1 != row2)
    return sum(abs(row1 - row2) > 0.0000001)


# Define the L1 distance function
def l1_distance(row1, row2):
    return sum(abs(row1 - row2))


# Define the Gower distance function
def gower_distance(row1, row2):
    return gower_matrix(pd.DataFrame([row1, row2]))[0, 1]


def compute_distance(df_cfs, df_test_factual, cfs_num_list, cfs_size_list):

    df_cfs = df_cfs.reindex(columns=df_test_factual.columns)
    

    l0_averages = []
    l1_averages = []
    gower_averages = []
    
    # Iterative df_test_factual each line 
    start_index_cfs = 0
    for i, (index, factual_row) in enumerate(df_test_factual.iterrows()):
        k = cfs_num_list[i]
        size = cfs_size_list[i]
        
        l0_distances = []
        l1_distances = []
        gower_distances = []
    
        for j in range(start_index_cfs, start_index_cfs + size):
            counterfactual_row = df_cfs.iloc[j].astype(float)
    
            l0_distances.append(l0_distance(factual_row, counterfactual_row))
            l1_distances.append(l1_distance(factual_row, counterfactual_row))
            gower_distances.append(gower_distance(factual_row, counterfactual_row))
    
        l0_averages.append(np.mean(l0_distances))
        l1_averages.append(np.mean(l1_distances))
        gower_averages.append(np.mean(gower_distances))
    
        # Update to reflect the value of the next k
        start_index_cfs += k
    
    result_l0 = pd.DataFrame(l0_averages, columns=['L0 Average Distance'])
    result_l1 = pd.DataFrame(l1_averages, columns=['L1 Average Distance'])
    result_gower = pd.DataFrame(gower_averages, columns=['Gower Average Distance'])

    return result_l0, result_l1, result_gower


# compute the diversity
def compute_diversity(df_cfs, df_test_factual, cfs_num_list, cfs_size_list):
    df_cfs = df_cfs.reindex(columns=df_test_factual.columns)
     
    distinct_columns_counts = []
    
    start_index_cfs = 0
    for i, (index, factual_row) in enumerate(df_test_factual.iterrows()):
        k = cfs_num_list[i]
        size = cfs_size_list[i]
        distinct_columns_set = set()  
    
        # calculating different column names for each k rows in df_cfs
        for j in range(start_index_cfs, start_index_cfs + size):
            counterfactual_row = df_cfs.iloc[j].astype(float)
            different_columns = factual_row.index[factual_row != counterfactual_row].tolist()
            distinct_columns_set.update(different_columns)
    
        # add the sizes of the different sets of column names for this row to the list
        distinct_columns_counts.append(len(distinct_columns_set))
    
        start_index_cfs += k
    
    result_diversity = pd.DataFrame(distinct_columns_counts, columns=['Average Diversity'])
    return result_diversity


# Predicts the data target. Assumption: Positive class label is at position 1
def predict_y(model, df, as_prob: bool = False):    
    predictions = model.predict(df)
    
    if not as_prob:
        predictions = predictions.round()
    
    return predictions


def compute_validity(ml_model_online, df_cfs, cfs_num_list):
    
    pre = predict_y(ml_model_online, df_cfs)
    start_index = 0
    result_validity = [] 
    
    for value in cfs_num_list:
        end_index = start_index + value
        mean_value = pre[start_index:end_index].mean()
        result_validity.append(mean_value)
        start_index = end_index
    
    return result_validity


def compute_act_size(df_cfs, df_test_factual, cfs_num_list, feat_num):

    df_cfs = df_cfs.reindex(columns=df_test_factual.columns)
    cfs_size_list = []
    distinct_columns_counts = []
    
    start_index_cfs = 0
    for i, (index, factual_row) in enumerate(df_test_factual.iterrows()):
        k = cfs_num_list[i]
        distinct_columns_set = set()  
        size_num = 0
        
        for j in range(start_index_cfs, start_index_cfs + k):
            size_num += 1
            counterfactual_row = df_cfs.iloc[j].astype(float)
            different_columns = factual_row.index[factual_row != counterfactual_row].tolist()
            distinct_columns_set.update(different_columns)
            if len(distinct_columns_set)>=feat_num:
                break
    
        cfs_size_list.append(size_num)
        distinct_columns_counts.append(len(distinct_columns_set))
        
        start_index_cfs += k
    
    return cfs_size_list, distinct_columns_counts