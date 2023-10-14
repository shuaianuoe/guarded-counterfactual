# -*- coding: utf-8 -*-

import math

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


