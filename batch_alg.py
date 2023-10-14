# -*- coding: utf-8 -*-

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