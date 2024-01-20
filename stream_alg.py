# -*- coding: utf-8 -*-

from our_tfd import greedy_maximum_coverage,greedy_set_cover
import math
import random

# GSV
def gsv(fact_row, stream_fact_row, cfs_num, stream_index, mc_res, W, C, epsilon):
    z = (cfs_num * math.log(len(stream_fact_row), 2)) / (epsilon ** 2) / 180
    diff_col_set  = set(fact_row.index[fact_row != stream_fact_row])
   
    if len(diff_col_set-C) < z / cfs_num :
        W[stream_index] = diff_col_set - C
    if len(diff_col_set-C) >= z / cfs_num and len(mc_res) < cfs_num:
        mc_res.append(stream_index)
        C = C | diff_col_set

    return W, mc_res



def gsv_postpro(W, cfs_num, mc_res):
    if len(mc_res) >= cfs_num:
        return mc_res
    
    universe = set.union(*W.values())
    mc_res.extend(greedy_maximum_coverage(universe, W, cfs_num-len(mc_res))) 
    if len(mc_res) < cfs_num:
        potential_keys = [key for key in W if key not in mc_res]
        mc_res.extend(random.sample(potential_keys, cfs_num - len(mc_res)))
    return mc_res
    


# GOPS
def gops(fact_row, stream_fact_row, cur_res, cfs_num, stream_index, beta):
    beta = 1.1
    S  = set(fact_row.index[fact_row != stream_fact_row])
    S_copy = S.copy().add('a')

    us = [] 
    while(len(S)>0 and S_copy!=S):
        S_copy = S.copy()           
        for key, value_set in cur_res.items():             
            if len(value_set) > (len(S) / beta):
                us.append(key)
        for key in us:
            S = S - cur_res[key]
    # update LS        
    ls = [key for key in cur_res.keys() if key not in us]       
    for key in ls:
        cur_res[key] = cur_res[key] - S
    
    cur_res[stream_index] = set(fact_row.index[fact_row != stream_fact_row])

    return cur_res



# DR
def dr(sc_res, stream_fact_row, fact_row,df_test_pos,stream_index):
        ks = 1
        delta = 0.01
        ini_num = ks*(stream_fact_row.shape[0]**delta) * math.log(df_test_pos.shape[0], 2) *  math.log(stream_fact_row.shape[0], 2)
        diff_columns = stream_fact_row[fact_row != stream_fact_row].index.tolist()

        if len(diff_columns) >= ini_num/ks:
            sc_res.append(stream_index)
            
        return sc_res, diff_columns
                
                
def dr_post(sc_res, subset_dict, df_test_pos,feat_num):
        union_of_values = set()
        for key in sc_res:
            union_of_values = union_of_values.union(subset_dict[key])
        
        if len(union_of_values)>feat_num:
            print("aa")
            return sc_res
        else:
            for key in sc_res:
                if key in subset_dict:
                    del subset_dict[key]
            universe = set(df_test_pos.columns)
            sc_res1 = greedy_set_cover(universe, subset_dict, feat_num-len(union_of_values))
     
            return sc_res+sc_res1



def compute_fewpri_ins(mc_res, df_test_pos, fact_row):
    pri_num = 10000
    pri_index = mc_res[0]
    for cur_i in mc_res:
        mc_res_cop = mc_res.copy()
             
        mc_res_cop.remove(cur_i)
        selected_row = df_test_pos.loc[mc_res_cop] 
        different_columns_expi = selected_row.columns[selected_row.ne(fact_row, axis=1).any()].tolist()
   
        cur_row = df_test_pos.loc[cur_i]
        different_columns_inci = cur_row.index[cur_row.ne(fact_row)].tolist()
        
        pri_diff_set = set(different_columns_inci)-set(different_columns_expi)
        if len(pri_diff_set) < pri_num:
            pri_num = len(pri_diff_set)
            pri_index = cur_i
            if len(pri_diff_set) == 0:
                break
            
    return pri_index


def stream_sDBC(mc_res, df_test_pos, fact_row, stream_fact_row, cfs_num, stream_index):
    pri_index = compute_fewpri_ins(mc_res, df_test_pos, fact_row)
        
    selected_row = df_test_pos.loc[mc_res] 
    different_columns_now = selected_row.columns[selected_row.ne(fact_row, axis=1).any()].tolist()        

    selected_row = df_test_pos.loc[pri_index] 
    different_columns_pri = selected_row.index[selected_row.ne(fact_row)].tolist()
     
    different_columns_stream = stream_fact_row.index[stream_fact_row.ne(fact_row)].tolist()
    
    if len(set(different_columns_now) - set(different_columns_pri) | set(different_columns_stream)) > (1+1/cfs_num) * len(set(different_columns_now)):
        mc_res.remove(pri_index)
        mc_res.append(stream_index)
    
    return mc_res


def compute_largest_T(stream_fact_row, fact_row, eff_dict):   
    different_columns_stream = stream_fact_row.index[stream_fact_row.ne(fact_row)].tolist()

    sorted_feas = [
        key for key, _ in sorted(
            {key: eff_dict[key] for key in different_columns_stream}.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]
    
    T = []
    for fea_inx in range(len(sorted_feas)):

        # print("len(sorted_feas)-fea_inx", len(sorted_feas)-fea_inx)
        if len(sorted_feas)-fea_inx == 1:
            log_value = 0
        else:
            log_value = math.ceil(math.log(len(sorted_feas)-fea_inx, 2))
        
        if eff_dict[sorted_feas[fea_inx]] < log_value:
            T = sorted_feas[fea_inx:]
            break
            
    return T


def stream_dSBC(sc_res, df_test_pos, fact_row, feat_num, stream_index, eff_dict, T, fur_re_flag=False):

    sc_res.append(stream_index)
    if len(T) == 1:
        log_value = 0
    else:
        log_value = math.ceil(math.log(len(T), 2))
        
    for fea in T:
        eff_dict[fea] = log_value
    
    # start removing redundancy
    sc_res_cop = sc_res.copy()
    sc_res_cop.remove(stream_index)
    if len(sc_res_cop) > 0:
        for red_inx in sc_res_cop:
            selected_row = df_test_pos.loc[red_inx] 
            different_columns_red = selected_row.index[selected_row.ne(fact_row)].tolist()
            if len(set(different_columns_red) - set(T)) == 0:
                sc_res.remove(red_inx)
    
    if fur_re_flag==True:
        pri_index = compute_fewpri_ins(sc_res, df_test_pos, fact_row)
        sc_res_cop = sc_res.copy()
        sc_res_cop.remove(pri_index)   
        selected_row = df_test_pos.loc[sc_res_cop] 
        different_columns_now = selected_row.columns[selected_row.ne(fact_row, axis=1).any()].tolist()   
        while len(set(different_columns_now))>= feat_num:
            sc_res.remove(pri_index)
            pri_index = compute_fewpri_ins(sc_res, df_test_pos, fact_row)
            sc_res_cop = sc_res.copy()
            sc_res_cop.remove(pri_index)    
            selected_row = df_test_pos.loc[sc_res_cop] 
            different_columns_now = selected_row.columns[selected_row.ne(fact_row, axis=1).any()].tolist()
        
    return sc_res, eff_dict


