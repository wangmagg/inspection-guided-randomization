from argparse import ArgumentParser
import glob
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from tqdm import tqdm

from src.data import gen_multarm_data, get_multarm_y_obs
from src.estimators import (
    get_tau_true, 
    diff_in_means_mult_arm, 
    get_pval_mult_arm, 
    qb_diff_in_means_mult_arm, 
    get_pval_qb
)
from src.igr import igr_enumeration, igr_restriction
from src.igr_checks import discriminatory_power, overrestriction
from src.metrics import get_metric

from src.utils.aesthetics import setup_fig
from src.utils.collate import collect_res_csvs
from src.utils.genetic_algorithms import get_genetic_kwargs

def multarm_config():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-iter", type=int, default=0)

    parser.add_argument("--n-stu", type=int, default=80)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--tau-sizes", type=float, nargs="+", default=[0, 0.3, 0.6])
    parser.add_argument("--sigma", type=float, default=0.1)

    parser.add_argument("--n-enum", type=int, default=int(1e4))
    parser.add_argument("--n-accept", type=int, default=500)
    parser.add_argument("--metric", type=str, nargs="+", default=["MaxMahalanobis", "SumMaxAbsSMD"])

    parser.add_argument("--genetic-iters", type=int, default=3)
    parser.add_argument("--tourn-size", type=int, default=2)
    parser.add_argument("--cross-k", type=int, default=2)
    parser.add_argument("--cross-rate", type=float, default=0.95)
    parser.add_argument("--mut-rate", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.1)

    parser.add_argument("--min-block-factor", type=int, default=2)
    parser.add_argument("--qb-dir", type=str, default="qb")

    parser.add_argument("--out-dir", type=str, default="res/vig1_multarm")

    args = parser.parse_args()
    genetic_kwargs = get_genetic_kwargs(args)

    comps = np.column_stack((np.zeros(args.n_arms - 1, dtype=int), np.arange(1, args.n_arms, dtype=int)))
    metric_kwargs = {"n_arms": args.n_arms, "comps": comps, "X": None}
    subdir_dict = {"n_enum": args.n_enum, "n_accept": args.n_accept}
    kwargs = {
        "metric": metric_kwargs,
        "genetic": genetic_kwargs,
        "save": subdir_dict
    }
    save_dir_dgp = Path(args.out_dir) / f"n-{args.n_stu}_arms-{args.n_arms}"
    save_subdirs = [f"{k}-{v}" for k, v in subdir_dict.items()]
    save_dir_data = save_dir_dgp / str(args.data_iter)
    save_dir_res = save_dir_dgp / str(args.data_iter) / Path(*save_subdirs)

    if not save_dir_res.exists():
        save_dir_res.mkdir(parents=True)

    return args, kwargs, save_dir_data, save_dir_res

def restriction(
    design,
    z_pool,
    n_accept,
    save_dir,
    save_path_data=None,
    metric=None,
    metric_kwargs=None,
    genetic_kwargs=None,
    seed=None
):
    save_dir_design = save_dir / design
    if not save_dir_design.exists():
        save_dir_design.mkdir()

    if design == "QB":
        save_path_z = save_dir_design / f"z.csv"
    elif design == "CR":
        save_path_z = save_dir_design / f"z.pkl"
    else:
        save_path_z = save_dir_design / f"{metric.__name__}_z.pkl"

    save_path_scores = save_dir_design / f"{metric.__name__}_scores.pkl"
    save_path_scores_pool = save_dir_design / f"{metric.__name__}_scores_pool.pkl"

    print(f"{design}: Restricting...")

    if save_path_z.exists() and save_path_scores.exists() and save_path_scores_pool.exists():
        print(f"{save_path_z.name} already exists, loading...")
        if design == "QB":
            z_accepted = pd.read_csv(save_path_z).to_numpy()
        else:
            with open(save_path_z, "rb") as f:
                z_accepted = pickle.load(f)
        with open(save_path_scores, "rb") as f:
            scores_accepted = pickle.load(f)
        with open(save_path_scores_pool, "rb") as f:
            scores_pool = pickle.load(f)
        return z_accepted, scores_pool, scores_accepted
    
    if save_path_scores_pool.exists():
        print(f"{save_path_scores_pool.name} already exists, loading...")
        with open(save_path_scores_pool, "rb") as f:
            scores_pool = pickle.load(f)
    else:
        scores_pool = None
    
    if design == "QB":
        subprocess.run(
                [
                    "Rscript", "src/utils/qb.R",
                    "--data-path", save_path_data,
                    "--z-path", save_path_z,
                    "--blocks-path", save_dir_design / "blocks.csv",
                    "--n-arms", str(args.n_arms),
                    "--n-accept", str(args.n_accept),
                    "--min-block-factor", str(args.min_block_factor),
                    "--seed", str(seed),
                ]
            )
        z_accepted = pd.read_csv(save_path_z).to_numpy()
        scores_accepted = metric(z_accepted, **metric_kwargs)
    else:
        z_accepted, (scores_pool, _), (scores_accepted, _) = igr_restriction(
            z_pool,
            n_accept,
            random=("CR" in design),
            metric_1=metric,
            scores_pool_1=scores_pool,
            mirror_type="all",
            genetic=("IGRg" in design),
            metric_1_kwargs=metric_kwargs,
            genetic_kwargs=genetic_kwargs
        )

    if design != "QB":
        with open(save_path_z, "wb") as f:
            pickle.dump(z_accepted, f)
    with open(save_path_scores, "wb") as f:
        pickle.dump(scores_accepted, f)
    with open(save_path_scores_pool, "wb") as f:
        pickle.dump(scores_pool, f)

    return z_accepted, scores_pool, scores_accepted


def run_trial_and_analyze(
        design, 
        y_0, y_1, z_accepted, comps, 
        save_dir, data_iter,
        metric_lbl=None,
        subdir_dict=None):
    
    save_dir_design = save_dir / design
    if not save_dir_design.exists():
        save_dir_design.mkdir()

    if metric_lbl is not None:
        tau_hat_path = save_dir_design / f"{metric_lbl}_tau_hat.pkl"
        res_path = save_dir_design / f"{metric_lbl}_res.csv"
    else:
        tau_hat_path = save_dir_design / "tau_hat.pkl"
        res_path = save_dir_design / "res.csv"

    if tau_hat_path.exists() and res_path.exists():
        print(f"{res_path.name} already exists, skipping estimation...")
        return
    else:
        print(f"{design}: Estimating...")
        y_obs_accepted = get_multarm_y_obs(y_0, y_1, z_accepted)

        if design == "QB":
            qb_blocks = pd.read_csv(save_dir_design / "blocks.csv")
            blocks = qb_blocks["cluster_label"].to_numpy()
            kwargs = {"blocks": blocks, 
                      "n_blocks": len(np.unique(blocks)), 
                      "weights": np.bincount(blocks) / len(blocks)}
            tau_hat_fn = qb_diff_in_means_mult_arm
            p_val_fn = get_pval_qb
        else:
            kwargs = {}
            tau_hat_fn = diff_in_means_mult_arm
            p_val_fn = get_pval_mult_arm
            
        tau_hat = np.array([tau_hat_fn(z, y_obs, comps=comps, **kwargs) for z, y_obs in zip(z_accepted, y_obs_accepted)])
        p_val = Parallel(n_jobs=-1, verbose=1)(delayed(p_val_fn)(z_accepted, y_obs_accepted, idx, comps=comps, **kwargs)
                                               for idx in tqdm(range(z_accepted.shape[0])))

        tau_true = get_tau_true(y_0, y_1, comps)
        bias = np.mean(tau_hat, axis=0) - tau_true
        rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2, axis=0))
        rr = np.mean(np.array(p_val) < 0.05, axis=0)

        if metric_lbl is not None:
            design = f"{design} - {metric_lbl}"
        res_dict = {"data_iter": data_iter,
                    "design": design}
        res_dict.update(subdir_dict)
        for i, (bias, rmse, rr) in enumerate(zip(bias, rmse, rr)):
            res_dict[f"bias_{i}"] = bias
            res_dict[f"rmse_{i}"] = rmse
            res_dict[f"rr_{i}"] = rr
        res_df = pd.DataFrame([res_dict])

        with open(tau_hat_path, "wb") as f:
            pickle.dump(tau_hat, f)
        res_df.to_csv(res_path, index=False)

if __name__ == "__main__":
    args, kwargs, save_dir_data, save_dir_res = multarm_config()

    # Generate data
    y_0, y_1, X = gen_multarm_data(args.n_stu, args.tau_sizes, args.sigma, args.seed + args.data_iter)
    kwargs["metric"]["X"] = X.to_numpy()
    X.to_csv(save_dir_data / f"X.csv", index=False)

    # Generate pool of candidate treatment allocations
    z_pool = igr_enumeration(args.n_stu, args.n_arms, args.n_enum, args.seed)

    # Run IGR and IGRg for each metric
    dp_fig, dp_axs = setup_fig(ncols=len(args.metric), sharex=False, sharey=True)
    or_fig, or_axs = setup_fig(ncols=len(args.metric), sharex=True, sharey=True)

    for i, metric_name in enumerate(args.metric):
        metric = get_metric(metric_name)

        # Run CR benchmark
        z_accepted_cr, scores_pool_cr, scores_accepted_cr = restriction(
            "CR", 
            z_pool, 
            args.n_accept, 
            save_dir_res,
            metric=metric,
            metric_kwargs=kwargs["metric"])
        run_trial_and_analyze(
            "CR", 
            y_0, y_1, z_accepted_cr, 
            kwargs["metric"]["comps"], 
            save_dir_res, args.data_iter, 
            subdir_dict=kwargs["save"]
            )

        # Run QB benchmark
        z_accepted_qb, scores_pool_qb, scores_accepted_qb = restriction(
            "QB",
            None,
            args.n_accept,
            save_dir_res,
            save_path_data=save_dir_data / "X.csv",
            metric=metric,
            metric_kwargs=kwargs["metric"],
            seed=args.seed
        )
        run_trial_and_analyze(
            "QB", 
            y_0, y_1, z_accepted_qb, 
            kwargs["metric"]["comps"], 
            save_dir_res, args.data_iter, 
            subdir_dict=kwargs["save"])
        
        # Run IGR
        z_accepted_igr, scores_pool_igr, scores_accepted_igr = restriction(
            "IGR",
            z_pool,
            args.n_accept,
            save_dir_res,
            metric=metric,
            metric_kwargs=kwargs["metric"],
        )
        run_trial_and_analyze(
            "IGR",
            y_0, y_1, z_accepted_igr, 
            kwargs["metric"]["comps"], 
            save_dir_res, args.data_iter,
            metric_lbl = metric_name,
            subdir_dict=kwargs["save"])

        # Run IGRg
        z_accepted_igr_g, scores_pool_igrg, scores_accepted_igrg = restriction(
            "IGRg",
            z_pool,
            args.n_accept,
            save_dir_res,
            metric=metric,
            metric_kwargs=kwargs["metric"],
            genetic_kwargs=kwargs["genetic"],
        )
        run_trial_and_analyze(
            f"IGRg", 
            y_0, y_1, z_accepted_igr_g, 
            kwargs["metric"]["comps"], 
            save_dir_res, args.data_iter, 
            metric_lbl = metric_name,
            subdir_dict=kwargs["save"])

        # Perform IGR checks
        discriminatory_power(
            fitness_lbl=metric_name,
            scores_1=scores_pool_igr,
            scores_1_g=scores_pool_igrg,
            n_accept=args.n_accept,
            ax=dp_axs[i]
        )
        overrestriction(
            fitness_lbl=metric_name,
            design_to_z_accepted={
                "CR": z_accepted_cr,
                "QB": z_accepted_qb,
                "IGR": z_accepted_igr,
                "IGRg": z_accepted_igr_g,
            },
            ax=or_axs[i]
        )

        # Save IGR checks 
        if not (save_dir_res / "igr_checks").exists():
            (save_dir_res / "igr_checks").mkdir()
        dp_fig.savefig(save_dir_res / "igr_checks" / f"discriminatory_power.svg", transparent=True, bbox_inches="tight")
        or_fig.savefig(save_dir_res / "igr_checks" / f"overrestriction.svg", transparent=True, bbox_inches="tight")
        plt.close(dp_fig)
        plt.close(or_fig)

        # Save estimation and inference results 
        collect_res_csvs(save_dir_data.parent, n_arms=args.n_arms - 1)
        
