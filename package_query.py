# -*- coding: utf-8 -*-
from pulp import *
import pandas as pd
import random
from scipy.spatial import KDTree



def use_ILP_div(sets, k):
    # Create a list of all unique elements
    elements = set(e for s in sets.values() for e in s)

    # Initialize the problem
    prob = LpProblem("MaximizeUniqueElements", LpMaximize)

    # Create decision variables
    x_vars = LpVariable.dicts("x", sets.keys(), 0, 1, LpBinary)
    y_vars = LpVariable.dicts("y", elements, 0, 1, LpBinary)

    # Objective function: Maximize the sum of y_vars (unique elements)
    prob += lpSum(y_vars.values())

    # Constraint: At most k sets can be chosen
    prob += lpSum(x_vars.values()) == k

    # Constraints: If an element is in any chosen set, its corresponding y_var must be 1
    for element in elements:
        prob += y_vars[element] <= lpSum(x_vars[set_name] for set_name, set_elements in sets.items() if element in set_elements)

    # Solve the problem using the default solver
    prob.solve()
    
    # print("prob status", prob.status)

    # Output the chosen sets and the number of unique elements
    chosen_sets = [set_name for set_name in sets if x_vars[set_name].varValue == 1]
    unique_elements_count = sum(y_vars[element].varValue for element in elements)

    return chosen_sets, unique_elements_count


def offline_partition_div(selected_df, tau):
    n_rows = selected_df.shape[0]
    n_groups = min(n_rows, int(1/tau))  

    group_sizes = [n_rows // n_groups] * n_groups
    for i in range(n_rows % n_groups):
        group_sizes[i] += 1

    group_dict = {} 
    group_start = 0
    avg_rows = []
    for i, size in enumerate(group_sizes):
        group_end = group_start + size
        group_df = selected_df.iloc[group_start:group_end]
        avg_row = group_df.mean().to_dict()
        avg_row['Group'] = f'Group {i}'
        avg_rows.append(avg_row)
        group_dict[f'Group {i}'] = list(selected_df.index[group_start:group_end])
        group_start = group_end

    average_df = pd.DataFrame(avg_rows)
    group_status = {f'Group {i}': 0 if size == 1 else 1 for i, size in enumerate(group_sizes)}

    return average_df, group_status, group_dict


def sketch_div(average_df, fact_row, cfs_num):
    subset_dict = {}
    
    for idx, row in average_df.drop(columns=['Group']).iterrows():
        diff_columns = row[row != fact_row].index.tolist()
        subset_dict[idx] = diff_columns
    
    mc_res_tmp, _ = use_ILP_div(subset_dict, cfs_num)                 
    return mc_res_tmp


def refine_div(average_df, group_status, group_dict, cfs_num, selected_df, mc_res_tmp, fact_row):
    total_status_sum = sum(group_status[average_df.loc[idx, 'Group']] for idx in mc_res_tmp)
    
    if total_status_sum == 0:
        mc_res = []
        for i in mc_res_tmp:
            mc_res.extend(group_dict[average_df.loc[i, 'Group']])  

    while(total_status_sum>0):
        actual_df = pd.DataFrame()
        for i in mc_res_tmp:
            actual_df = pd.concat([actual_df, selected_df.loc[group_dict[average_df.loc[i, 'Group']]]])
            mc_res_tmp.remove(i)
            
        subset_tmp_dict = {}
        
        for idx, row in actual_df.iterrows():
            diff_columns = row[row != fact_row].index.tolist()
            subset_tmp_dict[idx] = diff_columns
            
        mc_res, _ = use_ILP_div(subset_tmp_dict, cfs_num)
            
        total_status_sum = sum(group_status[average_df.loc[idx, 'Group']] for idx in mc_res_tmp)


    subset_dict = {}
    
    for idx, row in selected_df.iterrows():
        diff_columns = row[row != fact_row].index.tolist()
        subset_dict[idx] = diff_columns
        
    # Check if the list size is less than cfs_num
    if len(mc_res) < cfs_num:
        # Find keys from the dictionary that are not in the list and add them until the list has 5 elements
        keys_list = list(subset_dict.keys())
        random.shuffle(keys_list)
        for key in keys_list:
        # for key in subset_dict.keys():
            if key not in mc_res:
                mc_res.append(key)
                # Check if we have reached 5 elements
                if len(mc_res) == cfs_num:
                    break
    
    return mc_res



def use_ILP_suc(sets, p):
    elements = set(e for s in sets.values() for e in s)

    # Initialize the problem
    prob = LpProblem("MinimizeSets", LpMinimize)

    # Create decision variables
    x_vars = LpVariable.dicts("x", sets.keys(), 0, 1, LpBinary)
    y_vars = LpVariable.dicts("y", elements, 0, 1, LpBinary)

    # Objective function: Minimize the sum of x_vars (number of sets chosen)
    prob += lpSum(x_vars.values())

    # Constraint: At least p should be covered
    prob += lpSum(y_vars.values()) >= p

    # Constraints: Linking y_vars and x_vars
    for element in elements:
        prob += y_vars[element] <= lpSum(x_vars[set_name] for set_name, set_elements in sets.items() if element in set_elements)

    # Solve the problem using the default solver
    prob.solve()

    # Output the chosen sets
    chosen_sets = [set_name for set_name in sets if x_vars[set_name].varValue == 1]

    return chosen_sets


def sketch_suc(average_df, fact_row, feat_num):
    subset_dict = {}

    for idx, row in average_df.drop(columns=['Group']).iterrows():
        diff_columns = row[row != fact_row].index.tolist()
        subset_dict[idx] = diff_columns
    
    
    mc_res_tmp = use_ILP_suc(subset_dict, feat_num)
                     
    return mc_res_tmp


def refine_suc(average_df, group_status, group_dict, feat_num, selected_df, sc_res_tmp, fact_row):
    
    total_status_sum = sum(group_status[average_df.loc[idx, 'Group']] for idx in sc_res_tmp)

    if total_status_sum == 0:
        sc_res = []
        for i in sc_res_tmp:
            sc_res.extend(group_dict[average_df.loc[i, 'Group']])  

    while(total_status_sum>0):
        actual_df = pd.DataFrame()
       
        for i in sc_res_tmp:
            actual_df = pd.concat([actual_df, selected_df.loc[group_dict[average_df.loc[i, 'Group']]]])
            sc_res_tmp.remove(i)
            
        subset_tmp_dict = {}
        
        for idx, row in actual_df.iterrows():
            diff_columns = row[row != fact_row].index.tolist()
            subset_tmp_dict[idx] = diff_columns
            
        sc_res = use_ILP_suc(subset_tmp_dict, feat_num)
            
        total_status_sum = sum(group_status[average_df.loc[idx, 'Group']] for idx in sc_res_tmp)
    
    subset_dict = {}

    for idx, row in selected_df.iterrows():
        diff_columns = row[row != fact_row].index.tolist()
        subset_dict[idx] = diff_columns
    

    union_of_elements = set()
    for key in sc_res:
        union_of_elements.update(subset_dict[key])
    
    # Check if the union contains 4 or fewer unique elements
    if len(union_of_elements) <= feat_num:
        # Iterate over keys in subset_dict to find additional keys to satisfy the condition
        for key, elements in subset_dict.items():
            if key not in sc_res:
                # Check if adding this key's elements to the union will satisfy the condition
                new_elements = set(elements) - union_of_elements
                if new_elements:
                    # If new elements are found, update the union and extend sc_res
                    union_of_elements.update(new_elements)
                    sc_res.append(key)
                    # If we have reached at least 4 unique elements, we can stop
                    if len(union_of_elements) >= feat_num:
                        break
    return sc_res
    