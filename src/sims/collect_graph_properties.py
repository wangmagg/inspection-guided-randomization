import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--exp-subdir", type=str, default="kenya")
    parser.add_argument("--res-dir", type=str, default="res")
    args = parser.parse_args()

    data_exp_subdir = Path(args.data_dir) / args.exp_subdir
    all_graph_props = []

    for param_subdir in data_exp_subdir.glob("*"):
        for net_mdl_subdir in param_subdir.glob("net-*"):
            for data_fname in net_mdl_subdir.glob("*.pkl"):
                print(data_fname)
                with open(data_fname, "rb") as f:
                    data = pickle.load(f)
                _, _, X, X_school, _, A = data
                n_edges = np.sum(A) / 2
                degs = np.sum(A, axis=0)

                mean_deg, std_deg = np.mean(degs), np.std(degs)
                max_deg, min_deg = np.max(degs), np.min(degs)
                med_deg = np.median(degs)

                degs_diff_sch = np.zeros(X.shape[0])
                for i in range(X.shape[0]):
                    sch_id = X.loc[i, "school_id"]
                    sch_id_mask = X["school_id"] == sch_id
                    degs_diff_sch[i] = A[i, :].sum() - A[i, sch_id_mask].sum()
                mean_deg_diff_sch, std_deg_diff_sch = np.mean(degs_diff_sch), np.std(
                    degs_diff_sch
                )
                max_deg_diff_sch, min_deg_diff_sch = np.max(degs_diff_sch), np.min(
                    degs_diff_sch
                )
                med_deg_diff_sch = np.median(degs_diff_sch)

                degs_diff_set = np.zeros(X.shape[0])
                for i in range(X.shape[0]):
                    set_id = X.loc[i, "settlement_id"]
                    set_id_mask = X["settlement_id"] == set_id
                    degs_diff_set[i] = A[i, :].sum() - A[i, set_id_mask].sum()
                mean_deg_diff_set, std_deg_diff_set = np.mean(degs_diff_set), np.std(
                    degs_diff_set
                )
                max_deg_diff_set, min_deg_diff_set = np.max(degs_diff_set), np.min(
                    degs_diff_set
                )
                med_deg_diff_set = np.median(degs_diff_set)

                graph_props = {
                    "param_subdir": param_subdir,
                    "net_mdl_subdir": net_mdl_subdir,
                    "data_rep": data_fname.stem,
                    "n_edges": n_edges,
                    "mean_deg": mean_deg,
                    "std_deg": std_deg,
                    "max_deg": max_deg,
                    "min_deg": min_deg,
                    "med_deg": med_deg,
                    "mean_deg_diff_sch": mean_deg_diff_sch,
                    "std_deg_diff_sch": std_deg_diff_sch,
                    "max_deg_diff_sch": max_deg_diff_sch,
                    "min_deg_diff_sch": min_deg_diff_sch,
                    "med_deg_diff_sch": med_deg_diff_sch,
                    "mean_deg_diff_set": mean_deg_diff_set,
                    "std_deg_diff_set": std_deg_diff_set,
                    "max_deg_diff_set": max_deg_diff_set,
                    "min_deg_diff_set": min_deg_diff_set,
                    "med_deg_diff_set": med_deg_diff_set,
                }

                all_graph_props.append(graph_props)
                
    all_graph_props_df = pd.DataFrame.from_records(all_graph_props)
    res_exp_subdir = Path(args.res_dir) / args.exp_subdir
    if not res_exp_subdir.exists():
        res_exp_subdir.mkdir(parents=True)

    save_fname = res_exp_subdir / f"{args.exp_subdir}_graph_props.csv"
    print(f"Saving to {save_fname}")
    all_graph_props_df.to_csv(save_fname, index=False)
