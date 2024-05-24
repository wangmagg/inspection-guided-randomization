
import numpy as np


def rand_mdl_subdir_order(rand_mdl_names):
    benchmark_names = [name for name in rand_mdl_names if "restricted" not in name]
    rand_mdl_names_sorted = np.sort(rand_mdl_names)
    igr_names = [
        name
        for name in rand_mdl_names_sorted
        if "restricted" in name and "genetic" not in name
    ]
    igrg_names = [
        name for name in rand_mdl_names_sorted if "restricted-genetic" in name
    ]
    return benchmark_names + igr_names + igrg_names

def perc_of_cr(group, col):
    ref_value = group[group["rand_mdl"].str.contains("rand-complete")][col].values[0]
    perc_reduc = group[col].apply(lambda x: x / ref_value * 100)
    return perc_reduc