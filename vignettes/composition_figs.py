from argparse import ArgumentParser
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
from vignettes.multarm_figs import multarm_bal_boxplot, multarm_pairwise_bal_boxplot


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="res/vig1_composition")
    parser.add_argument("--n-enum", type=int, default=int(1e5))
    parser.add_argument("--n-accept", type=int, default=500)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.3, 0.7])
    parser.add_argument("--data-iter", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir) 
    rho_str = "-".join([str(rho) for rho in args.rhos])
    dgp_subdir = f"rhos-{rho_str}" 
    enum_subdir = f"n_enum-{args.n_enum}"
    accept_subdir = f"n_accept-{args.n_accept}"

    # Create directory for saving figures
    res_dir = out_dir / dgp_subdir / f"{args.data_iter}" / enum_subdir / accept_subdir 
    fig_dir = res_dir / 'res_figs'
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    # Load data and set parameters
    X = pd.read_csv(out_dir / dgp_subdir / f"{args.data_iter}" / "X.csv")
    X_no_gender = X.drop(columns="gender")
    designs = ['GFR', 'IGR', 'IGRg']
    metric_lbls = ["SumMaxAbsSMD", "MaxMahalanobis"]
    n_arms = 2

    n_groups = n_arms * len(args.rhos)
    same_rho_sets = np.arange(n_groups).reshape(len(args.rhos), n_arms)
    same_rho_pairs = np.concatenate([np.array(list(combinations(same_rho, 2))) for same_rho in same_rho_sets])
    same_z_pairs = np.concatenate([np.array(list(combinations(same_z, 2))) for same_z in same_rho_sets.transpose()])
    comps = np.vstack((same_rho_pairs, same_z_pairs))

    # Plot overall balance on each covariate across all group comparison
    multarm_bal_boxplot(
        designs,
        metric_lbls,
        n_groups,
        X_no_gender,
        comps,
        res_dir,
        fig_dir
    )

    # Plot balance on each covariate across pairwise comparisons between groups
    multarm_pairwise_bal_boxplot(
        designs,
        metric_lbls,
        n_groups,
        X,
        comps,
        res_dir,
        fig_dir
    )