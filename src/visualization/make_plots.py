import argparse
from pathlib import Path
import pickle

from src.design.fitness_functions import Fitness
from src.sims.run_kenya_trial import SimulatedKenyaTrial
from src.sims.run_multarm_trial import SimulatedMultiArmTrial
from src.sims.run_composition_trial import SimulatedCompositionTrial
from src.sims.run_network_trial import SimulatedNetworkTrial

from src.visualization.scatter.scatter_error_vs_score import (
    make_scatter_error_vs_score_fig,
)
from src.visualization.scatter.scatter_2dscore_disagg import (
    make_scatter_2dscore_disagg_fig,
)
from src.visualization.scatter.scatter_ranks import make_scatter_ranks_fig
from src.visualization.hist.hist_same_z_validity import make_hist_same_z_validity_fig
from src.visualization.hist.hist_z import make_hist_z_fig
from src.visualization.hist.hist_score import make_hist_score_fig
from src.visualization.hist.hist_2dscore_disagg import make_hist_2dscore_disagg_fig

from src.visualization.box.box_covariate_distr_per_arm import (
    make_covariate_distr_per_arm_fig,
)
from src.visualization.box.box_covariate_balance import make_covariate_balance_fig
from src.visualization.box.box_covariate_pairwise_balance import (
    make_covariate_pairwise_balance_fig,
)

from src.visualization.kenya.covar import (
    plot_kenya_school_locs,
    plot_y0_distr_across_sets,
    plot_deg_distr_across_sets,
)
from src.visualization.kenya.adjacency_dist import (
    plot_kenya_adjacency,
    plot_kenya_adj_v_dist,
    plot_kenya_pairwise_dists,
)
from src.visualization.kenya.deg_z_y import make_deg_z_fig, make_deg_y_fig, make_y_z_fig

from src.visualization.utils.gather_trials import (
    multarm_trial_set,
    composition_trial_set,
    kenya_trial_set,
    network_trial_set,
)


def network_randomization_plots(args):
    # check if any randomization plots are requested
    if not any(
        [
            fig_type in args.fig_types
            for fig_type in [
                "scatter_error_vs_score",
                "scatter_2dscore_disagg",
                "scatter_ranks",
                "hist_same_z_validity",
                "hist_z",
                "hist_z_uniqueness",
                "hist_score",
                "hist_2dscore_disagg",
                "heat_adj_z",
                "scatter_z_y",
            ]
        ]
    ):
        return
    output_dir = (
        Path(args.output_dir)
        / args.exp_subdir
        / args.net_mdl_subdir
        / args.n_z_subdir
        / args.po_mdl_subdir
    )
    print(output_dir)
    save_dir = Path(args.fig_dir) / args.exp_subdir
    save_prefix = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}"

    for rand_mdl_subdir in args.rand_mdl_subdirs:
        print(f"\t{rand_mdl_subdir}")
        trial_fname = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}.pkl"
        trial_path = output_dir / rand_mdl_subdir / trial_fname

        trial_sets, fitness_fns = network_trial_set(trial_path)

        save_subdir = (
            save_dir
            / args.net_mdl_subdir
            / args.n_z_subdir
            / args.po_subdir
            / rand_mdl_subdir
        )
        if not save_subdir.exists():
            save_subdir.mkdir(parents=True)

        for fig_type in args.fig_types:
            if fig_type == "scatter_error_vs_score":
                make_scatter_error_vs_score_fig(
                    trial_sets, fitness_fns, save_subdir, save_prefix
                )
            if fig_type == "scatter_2dscore_disagg":
                make_scatter_2dscore_disagg_fig(
                    trial_sets, args.axis_fns, save_subdir, save_prefix
                )
            if fig_type == "hist_same_z_validity":
                make_hist_same_z_validity_fig(trial_sets, save_subdir, save_prefix)
            if fig_type == "hist_z":
                make_hist_z_fig(trial_sets, save_subdir, save_prefix)
            if fig_type == "hist_score":
                make_hist_score_fig(trial_sets, save_subdir, save_prefix)
            if fig_type == "hist_2dscore_disagg":
                save_subdir = (
                    save_dir / args.net_mdl_subdir / args.n_z_subdir / args.po_subdir
                )
                make_hist_2dscore_disagg_fig(trial_sets, save_subdir, save_prefix)


def kenya_randomization_plots(args):
    # check if any randomization plots are requested
    if not any(
        [
            fig_type in args.fig_types
            for fig_type in [
                "scatter_error_vs_score",
                "scatter_2dscore_disagg",
                "scatter_ranks",
                "hist_same_z_validity",
                "hist_z",
                "hist_z_uniqueness",
                "hist_score",
                "hist_2dscore_disagg",
                "heat_adj_z",
                "scatter_deg_z",
                "scatter_y_z",
            ]
        ]
    ):
        return

    output_dir = (
        Path(args.output_dir)
        / args.exp_subdir
        / args.param_subdir
        / args.net_mdl_subdir[0]
        / args.po_mdl_subdir
        / args.n_z_subdir
    )
    print(output_dir)
    save_dir = Path(args.fig_dir) / args.exp_subdir
    save_prefix = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}"

    for rand_mdl_subdir in args.rand_mdl_subdirs:
        print(f"\t{rand_mdl_subdir}")
        trial_fname = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}.pkl"
        trial_path = output_dir / rand_mdl_subdir / trial_fname

        if not trial_path.exists():
            print(f"{trial_path} does not exist. Skipping...")
            continue

        trial_sets, fitness_fns = kenya_trial_set(
            trial_path, args.include_settlement_cluster
        )
        save_subdir = (
            save_dir
            / args.param_subdir
            / args.net_mdl_subdir[0]
            / args.po_mdl_subdir
            / args.n_z_subdir
            / rand_mdl_subdir
        )
        if not save_subdir.exists():
            save_subdir.mkdir(parents=True)

        for fig_type in args.fig_types:
            if fig_type == "scatter_error_vs_score":
                make_scatter_error_vs_score_fig(
                    trial_sets, fitness_fns, save_subdir, save_prefix
                )

            if fig_type == "scatter_2dscore_disagg":
                make_scatter_2dscore_disagg_fig(
                    trial_sets, args.axis_fns, save_subdir, save_prefix
                )

            if fig_type == "scatter_ranks":
                trial_sets_no_bench = {}
                for data_rep, trial_dict in trial_sets.items():
                    trial_sets_no_bench[data_rep] = {
                        k: v
                        for k, v in trial_dict.items()
                        if "CBR" in k and "CBRg" not in k
                    }
                make_scatter_ranks_fig(
                    trial_sets_no_bench, fitness_fns, save_subdir, save_prefix
                )

            if fig_type == "hist_same_z_validity":
                make_hist_same_z_validity_fig(trial_sets, save_subdir, save_prefix)

            if fig_type == "hist_z":
                make_hist_z_fig(trial_sets, save_subdir, save_prefix)

            if fig_type == "hist_score":
                make_hist_score_fig(trial_sets, save_subdir, save_prefix)

            if fig_type == "scatter_deg_z":
                make_deg_z_fig(
                    trial_sets, save_subdir, save_prefix, by_settlement=False
                )
                make_deg_z_fig(trial_sets, save_subdir, save_prefix, by_settlement=True)

            if fig_type == "scatter_y_z":
                make_y_z_fig(trial_sets, save_subdir, save_prefix, by_settlement=False)
                make_y_z_fig(trial_sets, save_subdir, save_prefix, by_settlement=True)


def kenya_data_plots(args):
    # check if any data plots are requested
    if not any(
        [
            fig_type in args.fig_types
            for fig_type in [
                "school_locs",
                "y0_distr_across_sets",
                "deg_distr_across_sets",
                "adjacency",
                "adj_v_dist",
                "pairwise_dists",
                "cos_sim_v_dist",
                "scatter_deg_y",
            ]
        ]
    ):
        return

    data_path = (
        Path(args.data_dir)
        / args.exp_subdir
        / args.param_subdir
        / args.net_mdl_subdir[0]
        / args.po_mdl_subdir
    )
    print(data_path)
    for data_fname in data_path.glob(f"*.pkl"):
        print(f"\t{data_fname}")
        with open(data_fname, "rb") as f:
            data = pickle.load(f)
            y0, _, X, X_school, _, A, sch_coords, _ = data
        save_dir = (
            Path(args.fig_dir)
            / args.exp_subdir
            / args.param_subdir
            / args.net_mdl_subdir
            / args.po_subdir
            / data_fname.stem
        )
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        if "school_locs" in args.fig_types:
            plot_kenya_school_locs(sch_coords, save_dir)

        if "adjacency" in args.fig_types:
            plot_kenya_adjacency(X, X_school, A, save_dir)

        if "adj_v_dist" in args.fig_types:
            plot_kenya_adj_v_dist(X, X_school, A, save_dir)

        if "pairwise_dists" in args.fig_types:
            plot_kenya_pairwise_dists(X_school, save_dir)

        if "scatter_deg_y" in args.fig_types:
            make_deg_y_fig(X, A, y0, save_dir, school_avg=False, by_settlement=False)
            make_deg_y_fig(X, A, y0, save_dir, school_avg=False, by_settlement=True)

        if "y0_distr_across_sets" in args.fig_types:
            plot_y0_distr_across_sets(y0, X, save_dir)

        if "deg_distr_across_sets" in args.fig_types:
            plot_deg_distr_across_sets(X, X_school, A, save_dir)


def multarm_randomization_plots(args):
    # check if any randomization plots are requested
    if not any(
        [
            fig_type in args.fig_types
            for fig_type in [
                "scatter_error_vs_score",
                "cov_pairwise_balance",
                "cov_balance",
                "cov_distr",
                "hist_z_uniqueness",
                "hist_score",
            ]
        ]
    ):
        return
    output_dir = (
        Path(args.output_dir)
        / args.exp_subdir
        / args.n_arm_subdir
        / args.n_per_arm_subdir
        / args.n_z_subdir
    )
    save_dir = Path(args.fig_dir) / args.exp_subdir
    save_prefix = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}"

    for rand_mdl_subdir in args.rand_mdl_subdirs:
        trial_path = output_dir / rand_mdl_subdir / f"{save_prefix}.pkl"
        trial_set, fitness_fn = multarm_trial_set(trial_path)

        if "scatter_error_vs_score" in args.fig_types:
            save_subdir = (
                save_dir
                / args.n_arm_subdir
                / args.n_per_arm_subdir
                / args.n_z_subdir
                / rand_mdl_subdir
            )
            make_scatter_error_vs_score_fig(
                trial_set, fitness_fn, save_subdir, save_prefix
            )
        if "hist_score" in args.fig_types:
            save_subdir = (
                save_dir
                / args.n_arm_subdir
                / args.n_per_arm_subdir
                / args.n_z_subdir
                / rand_mdl_subdir
            )
            make_hist_score_fig(trial_set, save_subdir, save_prefix)

        save_subdir = (
            save_dir / args.n_arm_subdir / args.n_per_arm_subdir / args.n_z_subdir
        )
        if not save_subdir.exists():
            save_subdir.mkdir(parents=True)

        if "cov_pairwise_balance" in args.fig_types:
            make_covariate_pairwise_balance_fig(
                trial_set, args.balance_fn_name, save_subdir, save_prefix
            )
        if "cov_balance" in args.fig_types:
            make_covariate_balance_fig(
                trial_set, args.balance_fn_name, save_subdir, save_prefix
            )
        if "cov_distr" in args.fig_types:
            make_covariate_distr_per_arm_fig(trial_set, save_subdir, save_prefix)


def composition_randomization_plots(args):
    # check if any randomization plots are requested
    if not any(
        [
            fig_type in args.fig_types
            for fig_type in [
                "scatter_error_vs_score",
                "cov_pairwise_balance",
                "cov_balance",
                "cov_distr",
                "hist_score",
                "hist_z_uniqueness",
            ]
        ]
    ):
        return

    output_dir = (
        Path(args.output_dir)
        / args.exp_subdir
        / args.p_ctxts_subdir
        / args.n_per_arm_subdir
        / args.n_z_subdir
    )
    save_dir = Path(args.fig_dir) / args.exp_subdir
    save_prefix = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}"

    for rand_mdl_subdir in args.rand_mdl_subdirs:
        trial_path = output_dir / rand_mdl_subdir / f"{save_prefix}.pkl"
        trial_set, fitness_fn = composition_trial_set(trial_path)

        if "scatter_error_vs_score" in args.fig_types:
            save_subdir = (
                save_dir
                / args.p_ctxts_subdir
                / args.n_per_arm_subdir
                / args.n_z_subdir
                / rand_mdl_subdir
            )
            make_scatter_error_vs_score_fig(
                trial_set, fitness_fn, save_subdir, save_prefix
            )
        if "hist_score" in args.fig_types:
            save_subdir = (
                save_dir
                / args.p_ctxts_subdir
                / args.n_per_arm_subdir
                / args.n_z_subdir
                / rand_mdl_subdir
            )
            make_hist_score_fig(trial_set, save_subdir, save_prefix)

        save_subdir = (
            save_dir / args.p_ctxts_subdir / args.n_per_arm_subdir / args.n_z_subdir
        )

        if "cov_pairwise_balance" in args.fig_types:
            make_covariate_pairwise_balance_fig(
                trial_set, args.balance_fn_name, save_subdir, save_prefix
            )
        if "cov_balance" in args.fig_types:
            make_covariate_balance_fig(
                trial_set, args.balance_fn_name, save_subdir, save_prefix
            )
        if "cov_distr" in args.fig_types:
            make_covariate_distr_per_arm_fig(trial_set, save_subdir, save_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--exp-subdir", type=str, default="kenya")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--data-rep", type=int, default=0)
    parser.add_argument("--run-seed", type=int, default=42)

    parser.add_argument("--n-arm-subdir", type=str, default="arms-4")
    parser.add_argument("--n-per-arm-subdir", type=str, default="n-per-arm-20")
    parser.add_argument("--balance-fn-name", type=str, default="signed-max-abs-smd")
    parser.add_argument("--pairwise-balance-fn-name", type=str, default="smd")

    parser.add_argument("--p-ctxts-subdir", type=str, default="p-comps-[0.5, 0.3, 0.7]")

    parser.add_argument("--include-settlement-cluster", action="store_true")
    parser.add_argument(
        "--param-subdir", type=str, default="ki-mu-pos_ko-da-pos_sd-sm_bal"
    )
    parser.add_argument(
        "--net-mdl-subdir",
        nargs="+",
        type=str,
        default=[
            "net-nested-2lvl-sb_psi-0.20_pdiso-0.02_pdido-0.01_intxn-euclidean-dist-power-decay-gamma-0.5"
        ],
    )
    parser.add_argument(
        "--po-mdl-subdir",
        type=str,
        default="kenya-hierarchical-nbr-sum",
    )
    parser.add_argument("--n-z-subdir", type=str, default="n-z-100000_n-cutoff-500")
    parser.add_argument(
        "--rand-mdl-subdirs",
        nargs="+",
        default=[
            "rand-restricted_min-pairwise-euclidean-dist_cluster-school",
            "rand-restricted_frac-exposed_cluster-school",
            "rand-restricted_max-mahalanobis_cluster-school",
            "rand-restricted_lin-comb_max-mahalanobis-0.50_min-pairwise-euclidean-dist-0.50_cluster-school",
            "rand-restricted_lin-comb_max-mahalanobis-0.50_frac-exposed-0.50_cluster-school",
            "rand-restricted_lin-comb_max-mahalanobis-0.75_min-pairwise-euclidean-dist-0.25_cluster-school",
            "rand-restricted_lin-comb_max-mahalanobis-0.75_frac-exposed-0.25_cluster-school",
            "rand-restricted_lin-comb_max-mahalanobis-0.25_min-pairwise-euclidean-dist-0.75_cluster-school",
            "rand-restricted_lin-comb_max-mahalanobis-0.25_frac-exposed-0.75_cluster-school",
        ],
    )

    parser.add_argument("--fig-types", nargs="+", type=str)
    parser.add_argument(
        "--axis-fns", nargs="+", type=str, default=["max-mahalanobis", "frac-expo"]
    )
    parser.add_argument("--fig-dir", type=str, default="figs")

    args = parser.parse_args()

    if args.exp_subdir == "kenya":
        kenya_randomization_plots(args)
        kenya_data_plots(args)
    if "mult-arm" in args.exp_subdir:
        multarm_randomization_plots(args)
    if args.exp_subdir == "composition":
        composition_randomization_plots(args)
    if args.exp_subdir == "network":
        network_randomization_plots(args)
