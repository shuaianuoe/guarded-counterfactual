# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:17:56 2023

@author: lenovo
"""

from itertools import combinations

def find_opt_min_cover(universe, subsets, feat_num):
 
    for i in range(1, len(subsets) + 1):

        for combo in combinations(subsets.keys(), i):
            combined = set()  
            for key in combo:
                combined.update(subsets[key])
         
            if len(combined) >= feat_num:  
                return list(combo)
    return None  
    
   
    
def find_opt_max_coverage(universe, subsets, cfs_num):

    best_coverage = 0
    best_combination = None
    

    for combo in combinations(subsets.keys(), cfs_num):
        coverage = set()
        for key in combo:
            coverage.update(subsets[key])
        if len(coverage) > best_coverage:
            best_coverage = len(coverage)
            best_combination = combo
            
            
            if coverage == universe:
                break
    
    return list(best_combination)


def greedy_maximum_coverage(universe, subsets_dict, cfs_num):

    uncovered_elements = set(universe)

    selected_subsets = []

    for _ in range(cfs_num):
        if not uncovered_elements:
            break

        best_subset = max(subsets_dict.keys(), key=lambda s: len(set(subsets_dict[s]) & uncovered_elements))
        selected_subsets.append(best_subset)

        uncovered_elements -= set(subsets_dict[best_subset])

        del subsets_dict[best_subset]

    return selected_subsets


def greedy_set_cover(universe, subsets, feat_num):

    uncovered = set(universe)
    
    chosen_subsets = []

    covered = set()

    while len(covered) < feat_num and uncovered:
        
        best_subset_key = max(subsets, key=lambda k: len(uncovered & set(subsets[k])))
        chosen_subsets.append(best_subset_key)
        
        covered |= set(subsets[best_subset_key])
        
        uncovered -= set(subsets[best_subset_key])
        
        del subsets[best_subset_key]

    return chosen_subsets