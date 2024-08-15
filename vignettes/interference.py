from argparse import ArgumentParser
from itertools import product
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm

from src.aesthetics import setup_fig, adjust_joint_grid_limits, save_joint_grids
from src.estimators import get_tau_true, diff_in_means, get_pval
from src.igr import igr_enumeration, igr_restriction
from src.igr_checks import (
    discriminatory_power, 
    overrestriction, 
    desiderata_tradeoffs, 
    desiderata_tradeoffs_pool, 
)
from src.igr_enhancements import get_genetic_kwargs
from src.metrics import get_metric
from src.aggregators import get_agg

from vignettes.data import gen_kenya_data, gen_kenya_network, get_kenya_sch_y_obs
from vignettes.collate import collect_res_csvs

def interference_config():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-iter", type=int, default=0)

    parser.add_argument("--n-sch-per-set", type=int, default=20)
    parser.add_argument("--n-stu-per-sch", type=int, default=40)
    parser.add_argument("--coords-range", type=float, default=0.005)
    parser.add_argument("--p-same-sch", type=float, default=0.2)
    parser.add_argument("--p-same-set-diff-sch", type=float, default=0.02)
    parser.add_argument("--p-diff-set-diff-sch", type=float, default=0.01)
    parser.add_argument("--beta", type=float, nargs="+", default=[1, 1])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--sigma-sch-in-set", type=float, default=0.2)
    parser.add_argument("--sigma-stu-in-sch", type=float, default=0.1)
    parser.add_argument("--tau-size", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.015)

    parser.add_argument("--n-enum", type=int, default=int(1e5))
    parser.add_argument("--n-accept", type=int, default=500)
    parser.add_argument("--mirror-type", type=str, default="all")
    parser.add_argument("--balance-metric", type=str, nargs="+", default=["MaxMahalanobis"])
    parser.add_argument("--interference-metric", type=str, nargs="+", default=["FracExpo", "InvMinEuclidDist"])
    parser.add_argument("--agg", type=str, default="LinComb")
    parser.add_argument("--w1", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0, 0.125, 0.375, 0.625, 0.875, 1])
    parser.add_argument("--w2", type=float, nargs="+", default=[0.75, 0.5, 0.25, 1, 0.875, 0.625, 0.375, 0.125, 0])

    parser.add_argument("--q", type=float, default=0.25)

    parser.add_argument("--genetic-iters", type=int, default=3)
    parser.add_argument("--tourn-size", type=int, default=2)
    parser.add_argument("--cross-k", type=int, default=2)
    parser.add_argument("--cross-rate", type=float, default=0.95)
    parser.add_argument("--mut-rate", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.05)

    parser.add_argument("--out-dir", type=str, default="res/vig2_interference")

    args = parser.parse_args()
    
    # Define covariate means and lat/long centroid coordinates of each settlement
    set_mus = np.array(
        [[0.25, 0.25],
        [0, 0.75],
        [0.05, 0],
        [0, 0.25],
        [1, 0]]
    )
    set_coords = np.array(
        [[36.7879475, -1.3114845],
        [36.884217, -1.3213091],
        [36.8721139, -1.2564898],
        [36.8909401, -1.2503642],
        [36.9025814, -1.2482996]]
    )

    # Set up  kwargs 
    data_kwargs = {"set_mus": set_mus, "set_coords": set_coords}
    b_metric_kwargs = {"n_arms": 2, "comps": [[0, 1]]}
    exposure_kwargs = {"q": args.q}
    genetic_kwargs = get_genetic_kwargs(args)
    subdir_dict = {
        "tau_size": args.tau_size,
        "n_enum": args.n_enum, 
        "n_accept": args.n_accept, 
        "mirror_type": args.mirror_type
        }

    kwargs = {
        "data": data_kwargs,
        "b_metric": b_metric_kwargs,
        "i_metric": {},
        "agg": {},
        "exposure": exposure_kwargs,
        "genetic": genetic_kwargs,
        "save": subdir_dict
    }

    # Define and create save directories
    save_dir_dgp = Path(args.out_dir) / f"gamma-{args.gamma}"
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
    mirror_type: str="all",
    fitness_lbl: str=None, 
    b_metric: callable=None,
    i_metric: callable=None,
    agg_fn: callable=None,
    b_metric_kwargs: dict={},
    i_metric_kwargs: dict={}, 
    genetic_kwargs: dict={},
    agg_kwargs: dict={}
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Restrict candidate pool of treatment allocations down 
    to the set of accepted allocations

    Args:
        - design: name of design
        - z_pool: candidate treatment allocation pool
        - n_accept: number of treatment allocations to accept
        - save_dir: directory to save results to 
        - b_metric: balance metric
        - i_metric: interference metric
        - agg_fn: aggregation function for combining balance and interference metrics
        - b_metric_kwargs: keyword arguments for balance metric
        - i_metric_kwargs: keyword arguments for interference metric
        - genetic_kwargs: keyword arguments for genetic algorithm
        - mirror_kwargs: keyword arguments for mirror algorithm
    """
        
    # Create save directory for design
    save_dir_design = save_dir / design
    if not save_dir_design.exists():
        save_dir_design.mkdir(parents=True)

    # Define save paths for the accepted allocations, the scores for the accepted allocations,
    # and the scores for the pool of candidate allocations
    save_path_scores_pool = save_dir_design / f"{b_metric.__name__} + {i_metric.__name__}_scores_pool.pkl"
    if design == "CR":
        save_path_z = save_dir_design / f"z.pkl"
        save_path_scores = save_dir_design / f"{b_metric.__name__} + {i_metric.__name__}_scores.pkl"
    else:
        save_path_z = save_dir_design / f"{fitness_lbl}_z.pkl"
        save_path_scores = save_dir_design / f"{fitness_lbl}_scores.pkl"

    print(f"{design}_{fitness_lbl}: Restricting...")

    # Load and return accepted allocations and scores if they already exist
    if save_path_z.exists() and save_path_scores.exists() and save_path_scores_pool.exists():
        print(f"{save_path_z.name} already exists, skipping restriction and loading existing...")
        with open(save_path_z, "rb") as f:
            z_accepted = pickle.load(f)
        with open(save_path_scores_pool, "rb") as f:
            scores_pool = pickle.load(f)
        with open(save_path_scores, "rb") as f:
            scores_accepted = pickle.load(f)
        return z_accepted, scores_pool, scores_accepted
    
    # Load scores for the pool of candidate allocations if they already exist
    if save_path_scores_pool.exists():
        with open(save_path_scores_pool, "rb") as f:
            scores_pool_1, scores_pool_2 = pickle.load(f)
    else:
        scores_pool_1, scores_pool_2 = None, None

    # Run restriction
    z_accepted, scores_pool, scores_accepted = igr_restriction(
        z_pool,
        n_accept,
        random=("CR" in design),
        metric_1=b_metric,
        metric_2=i_metric,
        scores_pool_1=scores_pool_1,
        scores_pool_2=scores_pool_2,
        agg_fn=agg_fn,
        mirror_type=mirror_type,
        genetic=("IGRg" in design),
        metric_1_kwargs=b_metric_kwargs,
        metric_2_kwargs=i_metric_kwargs,
        agg_kwargs=agg_kwargs,
        genetic_kwargs=genetic_kwargs
    )

    # Save and return the accepted allocations, the scores of the accepted allocations,
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
        tau_true: float,
        z_accepted: np.ndarray, 
        save_dir: Path, 
        data_iter: int, 
        exposure_kwargs: dict={},
        fitness_lbl: str=None,
        subdir_dict: dict={}
    ) -> None:
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
    if not save_dir_design.exists():
        save_dir_design.mkdir(parents=True)

    # Define save paths for the estimated treatment effects and the bias/rmse/rr results
    if fitness_lbl is not None:
        tau_hat_path = save_dir_design / f"{fitness_lbl}_tau_hat.pkl"
        res_path = save_dir_design / f"{fitness_lbl}_res.csv"
    else:
        tau_hat_path = save_dir_design / "tau_hat.pkl"
        res_path = save_dir_design / "res.csv"

    # Load estimated effects and results CSV if they already exist
    if tau_hat_path.exists() and res_path.exists():
        print(f"{res_path.name} already exists, skipping estimation...")
        return
    else:
        if not save_dir.exists():
            save_dir.mkdir()
        print(f"{data_iter}_{design}: Estimating...")

        # Get mean observed outcomes at the school-level
        sch_y_obs_accepted = get_kenya_sch_y_obs(y_0, y_1, z_accepted, **exposure_kwargs)

        # Get difference in means estimates
        tau_hat = np.array([diff_in_means(z, sch_y_obs) for z, sch_y_obs in zip(z_accepted, sch_y_obs_accepted)])

        # Get p-values
        p_val = Parallel(n_jobs=-2, verbose=1)(delayed(get_pval)(z_accepted, sch_y_obs_accepted, idx) 
                                               for idx in tqdm(range(z_accepted.shape[0])))
        
        # Get bias, RMSE, and rejection rate
        if fitness_lbl is not None:
            design = f"{design} - {fitness_lbl}"
        bias = np.mean(tau_hat) - tau_true
        rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
        rr = np.mean(np.array(p_val) < 0.05)

        # Save results
        res_dict = {
            "design": design,
            "data_iter": data_iter,
            "bias": bias,
            "rmse": rmse,
            "rr": rr,
            **subdir_dict
        }
        res_df = pd.DataFrame([res_dict])
        
        with open(tau_hat_path, "wb") as f:
            pickle.dump(tau_hat, f) 
        res_df.to_csv(res_path, index=False)

if __name__ == "__main__":
    args, kwargs, save_dir_data, save_dir_res = interference_config()

    n_set = kwargs["data"]["set_mus"].shape[0] # total number of settlements
    n = args.n_stu_per_sch * args.n_sch_per_set * n_set # total number of students
    n_sch = args.n_sch_per_set * n_set # total number of schools
    sch_sizes = np.repeat(args.n_stu_per_sch, n_sch) # number of students per school
    beta = np.array(args.beta) # coefficients for the covariates

    # Generate data
    print(save_dir_data)
    print("Generating network...")
    A, D = gen_kenya_network(
        kwargs["data"]["set_coords"],
        args.coords_range,
        args.n_sch_per_set,
        args.gamma, 
        args.p_same_sch, 
        args.p_same_set_diff_sch, 
        args.p_diff_set_diff_sch,
        block_sizes=sch_sizes,
        seed=args.seed + args.data_iter
    )
    with open(save_dir_data / "A.pkl", "wb") as f:
        pickle.dump(A, f)
    print("Generating covariates...")
    y_0, y_1, X_df = gen_kenya_data(
        kwargs["data"]["set_mus"],
        args.n_sch_per_set, args.n_stu_per_sch,
        args.sigma_sch_in_set, args.sigma_stu_in_sch,
        A,
        beta,
        args.tau_size, args.sigma,
        args.seed + args.data_iter
    )
    tau_true = get_tau_true(y_0, y_1, comps = [[0, 1]])
    X_df.to_csv(save_dir_data / "X.csv", index=False)
    with open(save_dir_data / f"tau_size-{args.tau_size}" / "tau_true.pkl", "wb") as f:
        pickle.dump(tau_true, f)

    # Enumerate pool of candidate allocations
    print("Enumerating...")
    z_pool = igr_enumeration(n_sch, 2, args.n_enum, seed=args.seed)

    # Run IGR and IGRg for each combination of balance and interference metrics
    kwargs["exposure"]["A"] = A
    kwargs["exposure"]["cluster_lbls"] = X_df["sch"].to_numpy()
    kwargs["b_metric"]["X"] = X_df.to_numpy()
    kwargs["b_metric"]["cluster_lbls"] = kwargs["exposure"]["cluster_lbls"]
    i_metric_kwargs_opts = {
        "FracExpo": kwargs["exposure"],
        "InvMinEuclidDist":  {"D": D}
    }
    for b_metric_name, i_metric_name in product(args.balance_metric, args.interference_metric):
        b_metric = get_metric(b_metric_name)
        i_metric = get_metric(i_metric_name)            
        agg_fn = get_agg(args.agg)
        kwargs["i_metric"] = i_metric_kwargs_opts[i_metric_name]

        # Run CR benchmark 
        z_accepted_cr, scores_pool_cr, scores_accepted_cr = restriction(
            "CR", 
            z_pool, args.n_accept, 
            save_dir_res, 
            b_metric=b_metric, 
            i_metric=i_metric,
            b_metric_kwargs=kwargs["b_metric"],
            i_metric_kwargs=kwargs["i_metric"])
        run_trial_and_analyze(
            "CR",
            tau_true, z_accepted_cr,
            save_dir_res, args.data_iter,
            exposure_kwargs=kwargs["exposure"],
            subdir_dict=kwargs["save"]
        )

        # Initialize plots for IGR checks
        if not (save_dir_res / "igr_checks").exists():
            (save_dir_res / "igr_checks").mkdir()
        dp_pool_fig, dp_pool_ax = setup_fig(ncols = 1, sharex=False, sharey=True)
        dp_fig, dp_axs = setup_fig(ncols = len(args.w1), sharex=False, sharey=True)
        or_fig, or_axs = setup_fig(ncols = len(args.w1), sharex=True, sharey=True)
        dt_grids = []
        dt_grid_save_fnames = []

        # IGR check: Desiderata tradeoff in pool of candidate allocations
        desiderata_tradeoffs_pool(
            metric_lbls = [b_metric_name, i_metric_name],
            scores_pool = scores_pool_cr,
            ax = dp_pool_ax[0],
            title=rf"$\gamma = {args.gamma}$"
        )
        dt_pool_save_fname = f"{args.data_iter}_{b_metric_name} + {i_metric_name}_desiderata_tradeoffs_pool.svg"
        dp_pool_fig.savefig(save_dir_res / "igr_checks" / dt_pool_save_fname,
                            transparent=True, bbox_inches="tight")

        # Iterate over metric weighting schemes
        for i, (w1, w2) in enumerate(zip(args.w1, args.w2)):
            kwargs["agg"] = {"w1": w1, "w2": w2}
            fitness_lbl = f"{w1:.2f}*{b_metric_name} + {w2:.2f}*{i_metric_name}"

            # Run IGR
            z_accepted_igr, scores_pool_igr, scores_accepted_igr = restriction(
                "IGR",
                z_pool, args.n_accept, 
                save_dir_res, 
                fitness_lbl=fitness_lbl,
                b_metric=b_metric, 
                i_metric=i_metric, 
                agg_fn=agg_fn, 
                mirror_type=args.mirror_type,
                b_metric_kwargs=kwargs["b_metric"],
                i_metric_kwargs=kwargs["i_metric"],
                agg_kwargs=kwargs["agg"]
            )
            run_trial_and_analyze(
                "IGR",
                tau_true, z_accepted_igr,
                save_dir_res, args.data_iter,
                fitness_lbl=fitness_lbl,
                exposure_kwargs=kwargs["exposure"],
                subdir_dict=kwargs["save"]
            )

            # Run IGRg
            z_accepted_igrg, scores_pool_igrg, scores_accepted_igrg = restriction(
                "IGRg",
                z_pool, args.n_accept, 
                save_dir_res, 
                fitness_lbl=fitness_lbl,
                b_metric=b_metric, b_metric_kwargs=kwargs["b_metric"],
                i_metric=i_metric, i_metric_kwargs=kwargs["i_metric"],
                agg_fn=agg_fn, agg_kwargs=kwargs["agg"],
                mirror_type=args.mirror_type,
                genetic_kwargs=kwargs["genetic"]
            )
            run_trial_and_analyze(
                "IGRg",
                tau_true, z_accepted_igrg,
                save_dir_res, args.data_iter,
                fitness_lbl=fitness_lbl,
                exposure_kwargs=kwargs["exposure"],
                subdir_dict=kwargs["save"]
            )

            # IGR check 1: Discriminatory power
            discriminatory_power(
                fitness_lbl=fitness_lbl,
                scores_1=scores_pool_igr[0], scores_1_g=scores_pool_igrg[0],
                scores_2=scores_pool_igr[1], scores_2_g=scores_pool_igrg[1],
                n_accept=args.n_accept,
                agg_fn=agg_fn, agg_kwargs=kwargs["agg"],
                ax=dp_axs[i]
            )
            # IGR check 2: Overrestriction
            overrestriction(
                fitness_lbl = fitness_lbl,
                design_to_z_accepted={"CR": z_accepted_cr, 
                                        "IGR": z_accepted_igr, 
                                        "IGRg": z_accepted_igrg},
                ax=or_axs[i]
            )
            # IGR check 3: Navigating desiderata tradeoffs
            dt_grid = desiderata_tradeoffs(
                metric_lbls = [b_metric_name, i_metric_name],
                fitness_lbl = fitness_lbl,
                design_to_scores={"CR": scores_accepted_cr,
                                    "IGR": scores_accepted_igr,
                                    "IGRg": scores_accepted_igrg},
            )
            dt_grids.append(dt_grid)

            # Save IGR checks
            metrics_lbl = f"{b_metric_name} + {i_metric_name}"
            dp_fig_save_fname = f"{metrics_lbl}_discriminatory_power.svg"
            or_fig_save_fname = f"{metrics_lbl}_overrestriction.svg"
            dt_grid_save_fname = f"{fitness_lbl}_desiderata_tradeoffs.svg"
            dt_grid_save_fnames.append(dt_grid_save_fname)

            dp_fig.savefig(save_dir_res / "igr_checks" / dp_fig_save_fname, transparent=True, bbox_inches="tight")
            or_fig.savefig(save_dir_res / "igr_checks" / or_fig_save_fname, transparent=True, bbox_inches="tight")
            dt_grid.savefig(save_dir_res / "igr_checks" / dt_grid_save_fname, transparent=True, bbox_inches="tight")

            plt.close(dp_fig)
            plt.close(or_fig)
            plt.close(dt_grid.figure) 

            # Collect estimation and inference results
            collect_res_csvs(save_dir_data.parent, bench_design="CR", variance=True)

        # Adjust axis limits to be the same for all desiderata tradeoff grids
        # adjust_joint_grid_limits(dt_grids, save_dir_res / "igr_checks", dt_grid_save_fnames)
        adjust_joint_grid_limits(dt_grids)
        save_joint_grids(dt_grids, save_dir_res / "igr_checks", dt_grid_save_fnames)
