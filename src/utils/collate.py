from pathlib import Path
import pandas as pd
from typing import List

def perc_of_benchmark(bench_design, group, col):
    ref_value = group[group["design"].str.contains(bench_design)][col].values[0]
    perc_reduc = group[col].apply(lambda x: x / ref_value * 100)
    return perc_reduc

def collect_res_csvs(save_dir, bench_design="CR", n_arms=2):
    res_dfs = []
    res_df_files = Path(save_dir).rglob("*res.csv")
    for res_df_file in res_df_files:
        res_dfs.append(pd.read_csv(res_df_file, index_col=None))

    res = pd.concat(res_dfs, ignore_index=True)
    bias_cols = [col for col in res.columns if col.startswith("bias")]
    rmse_cols = [col for col in res.columns if col.startswith("rmse")]
    rr_cols = [col for col in res.columns if col.startswith("rr")]

    group_cols = res.columns[~res.columns.str.contains("^bias.*?$|^rmse.*?$|^var.*?$|^rr.*?$|^design$|^data_iter")]
    design_group_cols = group_cols.tolist() + ["design"]
    iter_group_cols = group_cols.tolist() + ["data_iter"]

    res.sort_values(by=iter_group_cols, inplace=True)
    res_grouped = res.groupby(iter_group_cols, dropna=False)
    
    if res_grouped.ngroups == 1:
        for bias_col in bias_cols:
            res[f"perc_{bench_design}_{bias_col}"] = perc_of_benchmark(bench_design, res, bias_col)
        for rmse_col in rmse_cols:
            res[f"perc_{bench_design}_{rmse_col}"] = perc_of_benchmark(bench_design, res, rmse_col)
    else:
        for bias_col in bias_cols:
            res[f"perc_{bench_design}_{bias_col}"] = (
                res_grouped.apply(lambda x: perc_of_benchmark(bench_design, x, bias_col))
                .reset_index(drop=True)
                .sort_index().values
            )
        for rmse_col in rmse_cols:
            res[f"perc_{bench_design}_{rmse_col}"] = (
                res_grouped.apply(lambda x: perc_of_benchmark(bench_design, x, rmse_col))
                .reset_index(drop=True)
                .sort_index().values
            )
    res.to_csv(save_dir / "res_collated.csv", index=False)

    agg_dict = {}
    for bias_col in bias_cols:
        agg_dict[bias_col] = ['mean', 'std']
        agg_dict[f"perc_{bench_design}_{bias_col}"] = ['mean', 'std']
    for rmse_col in rmse_cols:
        agg_dict[rmse_col] = ['mean', 'std']
        agg_dict[f"perc_{bench_design}_{rmse_col}"] = ['mean', 'std']
    for rr_col in rr_cols:
        agg_dict[rr_col] = ['mean', 'std']

    res_agg = res.groupby(design_group_cols).agg(agg_dict).reset_index()
    res_agg.columns = ["_".join(a).strip('_') for a in res_agg.columns.to_flat_index()]
    res_agg.to_csv(save_dir / "res_collated_aggregated.csv", index=False)
