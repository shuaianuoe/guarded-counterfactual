# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:17:56 2023

@author: lenovo
"""

def greedy_maximum_coverage(universe, subsets_dict, cfs_num):
    # 开始时，未被覆盖的元素集合是全集
    uncovered_elements = set(universe)

    # 存放被选择的子集的集合
    selected_subsets = []

    for _ in range(cfs_num):
        # 如果所有元素都被覆盖，结束
        if not uncovered_elements:
            break

        # 按照子集覆盖的未被覆盖的元素数量，找到最佳子集
        best_subset = max(subsets_dict.keys(), key=lambda s: len(set(subsets_dict[s]) & uncovered_elements))
        selected_subsets.append(best_subset)

        # 更新未被覆盖的元素集合
        uncovered_elements -= set(subsets_dict[best_subset])
        # 从子集字典中删除已经选择的子集，避免再次选择
        del subsets_dict[best_subset]

    return selected_subsets


def greedy_set_cover(universe, subsets, feat_num):
    """
    universe: 一个集合，代表需要被覆盖的全集
    subsets: 一个字典，其中key为子集序号，value为子集（list形式）
    feat_num: 当选中的子集的并集的元素的个数达到这个值时，迭代将停止
    
    返回：选择的子集序号的列表
    """
    # 未被覆盖的元素集合初始化为全集
    uncovered = set(universe)
    # 存储已选择子集的序号
    chosen_subsets = []
    # 已覆盖的元素
    covered = set()

    while len(covered) < feat_num and uncovered:
        # 贪心选择能覆盖最多未覆盖元素的子集
        best_subset_key = max(subsets, key=lambda k: len(uncovered & set(subsets[k])))
        chosen_subsets.append(best_subset_key)
        # 更新已覆盖的元素集合
        covered |= set(subsets[best_subset_key])
        # 更新未覆盖的元素集合
        uncovered -= set(subsets[best_subset_key])
        # 从子集字典中移除已选择的子集，为了在下次迭代中不再考虑
        del subsets[best_subset_key]

    return chosen_subsets