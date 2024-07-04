from argparse import ArgumentParser
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path

from src.estimators import (
    get_tau_true_composition,
    diff_in_means_mult_arm, 
    get_pval_mult_arm,
)
from src.aesthetics import setup_fig
from src.igr import igr_paired_gfr_enumeration, igr_restriction
from src.igr_checks import discriminatory_power, overrestriction
from src.igr_enhancements import get_genetic_kwargs
from src.metrics import get_metric

from vignettes.data import gen_composition_data, get_composition_y_obs
from vignettes.collate import collect_res_csvs

def composition_config():
    """
    Set up configuration for composition
    """
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-iter", type=int, default=0)

    parser.add_argument("--n-stu", type=int, default=120)
    parser.add_argument("--n-arms", type=int, default=2)
    parser.add_argument('--tau-sizes-per-rho',type=float, nargs='+', default = [0.3, 0.5, 0.1])
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.3, 0.7])
    parser.add_argument("--prop-male", type=float, default=0.5)

    parser.add_argument("--n-enum", type=int, default=int(1e5))
    parser.add_argument("--n-accept", type=int, default=500)
    parser.add_argument("--metric", type=str, nargs="+", default=["MaxMahalanobis", "SumMaxAbsSMD"])

    parser.add_argument("--genetic-iters", type=int, default=3)
    parser.add_argument("--tourn-size", type=int, default=2)
    parser.add_argument("--cross-k", type=int, default=2)
    parser.add_argument("--cross-rate", type=float, default=0.95)
    parser.add_argument("--mut-rate", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.1)

    parser.add_argument("--out-dir", type=str, default="res/vig2_composition")

    args = parser.parse_args()

    # Set up kwargs
    genetic_kwargs = get_genetic_kwargs(args)

    n_groups = args.n_arms * len(args.rhos)
    same_rho_sets = np.arange(n_groups).reshape(len(args.rhos), args.n_arms)
    same_rho_pairs = np.concatenate([np.array(list(combinations(same_rho, 2))) for same_rho in same_rho_sets])
    same_z_pairs = np.concatenate([np.array(list(combinations(same_z, 2))) for same_z in same_rho_sets.transpose()])
    comps = np.vstack((same_rho_pairs, same_z_pairs))

    metric_kwargs = {"n_arms": n_groups, "comps": comps, "X": None}
    mirror_kwargs = {"gfr": True, "same_rho_pairs": same_rho_pairs}
    subdir_dict = {"n_enum": args.n_enum, "n_accept": args.n_accept}
    kwargs = {
        "metric": metric_kwargs,
        "genetic": genetic_kwargs,
        "mirror": mirror_kwargs,
        "save": subdir_dict
    }

    # Define and create save directories
    save_dir_dgp = Path(args.out_dir) / f"rhos-{'-'.join([str(rho) for rho in args.rhos])}"
    save_subdirs = [f"{k}-{v}" for k, v in subdir_dict.items()]
    save_dir_data = save_dir_dgp / str(args.data_iter)
    save_dir_res = save_dir_dgp / str(args.data_iter) / Path(*save_subdirs)
    if not save_dir_res.exists():
        save_dir_res.mkdir(parents=True)

    return args, kwargs, save_dir_data, save_dir_res

def restriction(
    design: str,
    z_pool: np.ndarray,
    n_accept: int,
    save_dir: Path,
    metric: callable=None,
    metric_kwargs: dict=None,
    genetic_kwargs: dict=None,
    mirror_kwargs: dict=None
):
    """
    Restrict candidate pool of treatment allocations down 
    to the set of accepted allocations

    Args:
        - design: name of design
        - z_pool: candidate treatment allocation pool
        - n_accept: number of treatment allocations to accept
        - save_dir: directory to save results
        - metric: fitness function
        - metric_kwargs: keyword arguments for fitness function
        - genetic_kwargs: keyword arguments for genetic algorithm
        - mirror_kwargs: keyword arguments for mirror algorithm
    """

    # Create save directory for design
    save_dir_design = save_dir / design
    if not save_dir_design.exists():
        save_dir_design.mkdir()

    # Define save paths
    save_path_scores_pool = save_dir_design / f"scores_pool.pkl"
    save_path_scores = save_dir_design / f"{metric.__name__}_scores.pkl"
    if design == "GFR":
        save_path_z = save_dir_design / f"z.pkl"
    else:
        save_path_z = save_dir_design / f"{metric.__name__}_z.pkl"

    print(f"{design}: Restricting...")

    # Load and return accepted allocations and scores if they already exist
    if save_path_z.exists() and save_path_scores.exists() and save_path_scores_pool.exists():
        print(f"{save_path_z.name} already exists, loading...")
        with open(save_path_z, "rb") as f:
            z_accepted = pickle.load(f)
        with open(save_path_scores, "rb") as f:
            scores_accepted = pickle.load(f)
        with open(save_path_scores_pool, "rb") as f:
            scores_pool = pickle.load(f)
        return z_accepted, scores_pool, scores_accepted
    
    # Load scores pool if it already exists
    if save_path_scores_pool.exists():
        with open(save_path_scores_pool, "rb") as f:
            scores_pool = pickle.load(f)
    else:
        scores_pool = None

    # Run restriction
    z_accepted, (scores_pool, _), (scores_accepted, _) = igr_restriction(
        z_pool,
        n_accept,
        random=("GFR" in design),
        metric_1=metric,
        scores_pool_1=scores_pool,
        mirror_type="all",
        genetic=("IGRg" in design),
        metric_1_kwargs=metric_kwargs,
        mirror_kwargs=mirror_kwargs,
        genetic_kwargs=genetic_kwargs
    )

    # Save and return the pool of accepted allocations, the scores of the accepted allocations,
    # and the scores of the pool of candidate allocations
    with open(save_path_z, "wb") as f:
        pickle.dump(z_accepted, f)
    with open(save_path_scores, "wb") as f:
        pickle.dump(scores_accepted, f)
    if not save_path_scores_pool.exists():
        with open(save_path_scores_pool, "wb") as f:
                pickle.dump(scores_pool, f)

    return z_accepted, scores_pool, scores_accepted


def run_trial_and_analyze(
        design: str, 
        y: np.ndarray, 
        z_accepted: np.ndarray, 
        comps: np.ndarray, 
        save_dir: Path, 
        data_iter: int, 
        metric_lbl: str=None, 
        subdir_dict: dict=None):
    """
    "Run" a trial and get estimation and inference results
    Args:
        - design: name of design
        - y: potential outcomes
        - z_accepted: accepted treatment allocations
        - comps: list of pairs of arms to compare
        - save_dir: directory to save results to
        - data_iter: data sample iteration number
        - metric_lbl: label for metric
        - subdir_dict: dictionary of subdirectory names,
            which will be included in the CSV results file
            to distinguish between runs of the same design under
            different simulation parameter settings
    """
    
    # Create save directory for design
    save_dir_design = save_dir / design

    # Define save paths
    if metric_lbl is not None:
        tau_hat_path = save_dir_design / f"{metric_lbl}_tau_hat.pkl"
        res_path = save_dir_design / f"{metric_lbl}_res.csv"
    else:
        tau_hat_path = save_dir_design / "tau_hat.pkl"
        res_path = save_dir_design / "res.csv"

    # Load estimated effects and results CSV if they already exist
    if tau_hat_path.exists() and res_path.exists():
        print(f"{res_path.name} already exists, skipping estimation...")
        return
    else:
        print(f"{design}: Estimating...")

        # Get observed outcomes
        y_obs_accepted = get_composition_y_obs(y, z_accepted)

        # Get difference in means estimates
        tau_hat = np.array([diff_in_means_mult_arm(z, y_obs, comps) for z, y_obs in zip(z_accepted, y_obs_accepted)])

        # Get p-values
        p_val = Parallel(n_jobs=-2, verbose=1)(delayed(get_pval_mult_arm)(z_accepted, y_obs_accepted, idx, comps)
                                    for idx in range(z_accepted.shape[0]))
        
        # Get bias, RMSE, and rejection rate
        tau_true = get_tau_true_composition(y, comps)
        bias = np.mean(tau_hat, axis=0) - tau_true
        rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2, axis=0))
        rr = np.mean(np.array(p_val) < 0.05, axis=0)

        # Save results
        if metric_lbl is not None:
            design = f"{design} - {metric_lbl}"
        res_dict = {"design": design, 
                    "data_iter": data_iter,
                    **subdir_dict}
        for i, (bias, rmse, rr) in enumerate(zip(bias, rmse, rr)):
            res_dict[f"bias_{i}"] = bias
            res_dict[f"rmse_{i}"] = rmse
            res_dict[f"rr_{i}"] = rr
        res_df = pd.DataFrame([res_dict])

        with open(tau_hat_path, "wb") as f:
            pickle.dump(tau_hat, f)
        res_df.to_csv(res_path, index=False)

if __name__ == "__main__":
    args, kwargs, save_dir_data, save_dir_res = composition_config()

    # Generate data
    n_grps = args.n_arms * len(args.rhos)
    tau_sizes_per_group = np.zeros(n_grps)
    tau_sizes_per_group[kwargs["mirror"]["same_rho_pairs"][:, 1]] = args.tau_sizes_per_rho
    y, X = gen_composition_data(
        args.n_stu, 
        args.prop_male,
        tau_sizes_per_group,
        args.sigma, 
        args.seed + args.data_iter)
    kwargs["metric"]["X"] = X.drop(columns="gender").to_numpy()
    X.to_csv(save_dir_data / "X.csv", index=False)

    # Generate pool of candidate treatment allocations
    attr_arr = X["gender"].to_numpy().astype(bool)
    z_pool = igr_paired_gfr_enumeration(args.n_stu, args.n_arms, args.n_enum, args.rhos, attr_arr, seed=args.seed)


    # Run IGR and IGRg for each metric
    dp_fig, dp_axs = setup_fig(ncols=len(args.metric), sharex=False, sharey=True)
    or_fig, or_axs = setup_fig(ncols=len(args.metric), sharex=True, sharey=True)

    for i, metric_name in enumerate(args.metric):
        metric = get_metric(metric_name)

        # Run GFR benchmark
        z_accepted_gfr, scores_pool_gfr, scores_accepted_gfr = restriction(
            "GFR", 
            z_pool, 
            args.n_accept, 
            save_dir_res, 
            metric=metric,
            metric_kwargs=kwargs["metric"],
            mirror_kwargs=kwargs["mirror"])
        
        run_trial_and_analyze(
            "GFR", 
            y, z_accepted_gfr, 
            kwargs["mirror"]["same_rho_pairs"],              
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
            mirror_kwargs=kwargs["mirror"]
        )
        run_trial_and_analyze(
            "IGR", 
            y, z_accepted_igr, 
            kwargs["mirror"]["same_rho_pairs"],              
            save_dir_res, args.data_iter,
            metric_lbl=metric_name,
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
            mirror_kwargs=kwargs["mirror"]
        )
        run_trial_and_analyze(
            "IGRg", 
            y, z_accepted_igr_g, 
            kwargs["mirror"]["same_rho_pairs"], 
            save_dir_res, args.data_iter,
            metric_lbl=metric_name,
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
                "GFR": z_accepted_gfr,
                "IGR": z_accepted_igr,
                "IGRg": z_accepted_igr_g,
            },
            ax=or_axs[i]
        )

        # Save IGR checks (do save for each subplot to track progress)
        if not (save_dir_res / "igr_checks").exists():
            (save_dir_res / "igr_checks").mkdir()
        dp_fig.savefig(save_dir_res / "igr_checks" / "discriminatory_power.svg", transparent=True, bbox_inches="tight")
        or_fig.savefig(save_dir_res / "igr_checks" / "overrestriction.svg", transparent=True, bbox_inches="tight")
        plt.close(dp_fig)
        plt.close(or_fig)

        # Save estimation and inference results (do save for each fitness fn to track progress)
        collect_res_csvs(save_dir_data.parent, bench_design="GFR")
