from pathlib import Path
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from typing import Union

def perc_of_benchmark(bench_design: str, group: Union[DataFrameGroupBy, pd.DataFrame], col):
    """
    Calculate percentage of a metric compared to a benchmark design
    
    Args:
        - bench_design: name of benchmark design
        - group: group of results from dataframe
        - col: column name of metric
    """
    ref_value = group[group["design"].str.contains(bench_design)][col].values[0]
    perc_reduc = group[col].apply(lambda x: x / ref_value * 100)
    return perc_reduc

def collect_res_csvs(save_dir: str, bench_design="CR", variance=False) -> None:
    """
    Collect and collate results from multiple res.csv files across
    different simulation parameter settings

    Args:
        - save_dir: directory to save collated results to
        - bench_design: name of benchmark design
        - variance: whether to calculate variance of metrics
    """
    res_dfs = []
    res_df_files = Path(save_dir).rglob("*res.csv")
    for res_df_file in res_df_files:
        res_dfs.append(pd.read_csv(res_df_file, index_col=None))

    res = pd.concat(res_dfs, ignore_index=True)
    bias_cols = [col for col in res.columns if col.startswith("bias")]
    rmse_cols = [col for col in res.columns if col.startswith("rmse")]
    rr_cols = [col for col in res.columns if col.startswith("rr")]

    # Calculate variance as the difference between RMSE^2 and bias^2
    if variance:
        var_cols = []
        for (bias_col, rmse_col) in zip(bias_cols, rmse_cols):
            var = res[rmse_col]**2 - res[bias_col] ** 2
            if len(bias_col.split("_")) == 2:
                var_col = f"var_{bias_col.split('_')[1]}"
            else:
                var_col = f"var"
            res[var_col] = var
            var_cols.append(var_col)

    group_cols = res.columns[~res.columns.str.contains("^bias.*?$|^rmse.*?$|^var.*?$|^rr.*?$|^design$|^data_iter")]
    design_group_cols = group_cols.tolist() + ["design"]
    iter_group_cols = group_cols.tolist() + ["data_iter"]

    res.sort_values(by=iter_group_cols, inplace=True)
    res_grouped = res.groupby(iter_group_cols, dropna=False)
    
    # Calculate percentage of metrics compared to benchmark design
    if res_grouped.ngroups == 1:
        for bias_col in bias_cols:
            res[f"perc_{bench_design}_{bias_col}"] = perc_of_benchmark(bench_design, res, bias_col)
        for rmse_col in rmse_cols:
            res[f"perc_{bench_design}_{rmse_col}"] = perc_of_benchmark(bench_design, res, rmse_col)
        if variance:
            for var_col in var_cols:
                res[f"perc_{bench_design}_{var_col}"] = perc_of_benchmark(bench_design, res, var_col)
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
        if variance:
            for var_col in var_cols:
                res[f"perc_{bench_design}_{var_col}"] = (
                    res_grouped.apply(lambda x: perc_of_benchmark(bench_design, x, var_col))
                    .reset_index(drop=True)
                    .sort_index().values
                )

    # Save collated results
    res.to_csv(save_dir / "res_collated.csv", index=False)

    agg_dict = {}
    for bias_col in bias_cols:
        agg_dict[bias_col] = ['mean', 'std']
        agg_dict[f"perc_{bench_design}_{bias_col}"] = ['mean', 'std']
    for rmse_col in rmse_cols:
        agg_dict[rmse_col] = ['mean', 'std']
        agg_dict[f"perc_{bench_design}_{rmse_col}"] = ['mean', 'std']
    if variance:
        for var_col in var_cols:
            agg_dict[var_col] = ['mean', 'std']
            agg_dict[f"perc_{bench_design}_{var_col}"] = ['mean', 'std']
    for rr_col in rr_cols:
        agg_dict[rr_col] = ['mean', 'std']

    res_agg = res.groupby(design_group_cols).agg(agg_dict).reset_index()
    res_agg.columns = ["_".join(a).strip('_') for a in res_agg.columns.to_flat_index()]
    res_agg.to_csv(save_dir / "res_collated_aggregated.csv", index=False)
