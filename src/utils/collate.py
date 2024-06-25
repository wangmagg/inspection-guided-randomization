from pathlib import Path
import pandas as pd

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
    group_cols = res.columns[~res.columns.str.contains("^bias.*?$|^rmse.*?$|^rr.*?$|^design$|^data_iter")]
    design_group_cols = group_cols.tolist() + ["design"]
    iter_group_cols = group_cols.tolist() + ["data_iter"]

    res.sort_values(by=iter_group_cols, inplace=True)
    res_grouped = res.groupby(iter_group_cols, dropna=False)
    if res_grouped.ngroups == 1:
        if n_arms > 2:
            for i in range(n_arms):
                res[f"perc_{bench_design}_bias_{i}"] = perc_of_benchmark(bench_design, res, f"bias_{i}")
                res[f"perc_{bench_design}_rmse_{i}"] = perc_of_benchmark(bench_design, res, f"rmse_{i}")
        else:
            res[f"perc_{bench_design}_bias"] = perc_of_benchmark(bench_design, res, "bias")
            res[f"perc_{bench_design}_rmse"] = perc_of_benchmark(bench_design, res, "rmse")
    else:
        if n_arms > 2:
            for i in range(n_arms):
                res[f"perc_{bench_design}_bias_{i}"] = (
                    res.groupby(iter_group_cols, dropna=False)
                    .apply(lambda x: perc_of_benchmark(bench_design, x, f"bias_{i}"))
                    .reset_index(drop=True)
                    .sort_index().values
                    )
                res[f"perc_{bench_design}_rmse_{i}"] = (
                    res.groupby(iter_group_cols, dropna=False)
                    .apply(lambda x: perc_of_benchmark(bench_design, x, f"rmse_{i}"))
                    .reset_index(drop=True)
                    .sort_index().values
                )
        else:
            res[f"perc_{bench_design}_bias"] = (
                res.groupby(iter_group_cols, dropna=False)
                .apply(lambda x: perc_of_benchmark(bench_design, x, "bias"))
                .reset_index(drop=True)
                .sort_index().values
                )
            res[f"perc_{bench_design}_rmse"] = (
                res.groupby(iter_group_cols, dropna=False)
                .apply(lambda x: perc_of_benchmark(bench_design, x, "rmse"))
                .reset_index(drop=True)
                .sort_index().values
            )
    res.to_csv(save_dir / "res_collated.csv", index=False)

    if n_arms > 2:
        agg_dict = {}
        for i in range(n_arms):
            agg_dict[f"bias_{i}"] = ['mean', 'std']
            agg_dict[f"rmse_{i}"] = ['mean', 'std']
            agg_dict[f"rr_{i}"] = ['mean', 'std']
            agg_dict[f"perc_{bench_design}_bias_{i}"] = ['mean', 'std']
            agg_dict[f"perc_{bench_design}_rmse_{i}"] = ['mean', 'std']
    else:
        agg_dict = {
            'bias': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'rr': ['mean', 'std'],
            f'perc_{bench_design}_bias': ['mean', 'std'],
            f'perc_{bench_design}_rmse': ['mean', 'std'],
        }
    res_agg = res.groupby(design_group_cols).agg(agg_dict).reset_index()
    res_agg.columns = ["_".join(a).strip('_') for a in res_agg.columns.to_flat_index()]
    res_agg.to_csv(save_dir / "res_collated_aggregated.csv", index=False)
