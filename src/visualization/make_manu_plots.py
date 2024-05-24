import argparse
from pathlib import Path

from src.sims.run_kenya_trial import SimulatedKenyaTrial
from src.sims.run_multarm_trial import SimulatedMultiArmTrial
from src.sims.run_composition_trial import SimulatedCompositionTrial
from src.sims.run_network_trial import SimulatedNetworkTrial

from src.visualization.box.box_covariate_balance import make_covariate_balance_fig
from src.visualization.box.box_covariate_pairwise_balance import (
    make_covariate_pairwise_balance_fig,
)
from src.visualization.scatter.scatter_2dscore_disagg import (
    make_scatter_2dscore_disagg_mult_fig,
)
from src.visualization.scatter.scatter_error_vs_score import (
    make_scatter_error_vs_score_mult_fig,
)
from src.visualization.hist.hist_score import make_hist_score_mult_fig
from src.visualization.hist.hist_same_z_validity import (
    make_hist_same_z_validity_mult_fig,
)
from src.visualization.hist.hist_2dscore_disagg import (
    make_hist_2dscore_disagg_mult_dgp_fig,
)
from src.visualization.utils.gather_trials import (
    multarm_trial_set,
    composition_trial_set,
    kenya_trial_set,
    kenya_trial_set_for_hist2d,
)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--fig-dir", type=str, default="manu_figs")
    parser.add_argument("--exp-subdir", type=str, default="kenya")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--data-rep", type=int, default=0)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument(
        "--fig-type", 
        nargs="+", 
        type=str, 
        default=["scatter_error_vs_score"]
    )
    parser.add_argument("--plt-suffix", type=str, default=None)

    parser.add_argument("--n-arm-subdir", type=str, default="arms-4")
    parser.add_argument("--n-per-arm-subdir", type=str, default="n-per-arm-20")
    parser.add_argument("--balance-fn-name", type=str, default="signed-max-abs-smd")
    parser.add_argument("--pairwise-balance-fn-name", type=str, default="smd")

    parser.add_argument("--p-ctxts-subdir", type=str, default="p-comps-[0.5, 0.3, 0.7]")

    parser.add_argument("--param-subdir", type=str, default="ki-mu-neg_ko-da-pos_sd-sm")
    parser.add_argument(
        "--net-mdl-subdir",
        nargs="+",
        type=str,
        default=[
            "net-nested-2lvl-sb_psi-0.20_pdiso0.02_p-dido0.01_intxn-euclidean-dist-power-decay-gamma-0.25"
        ],
    )
    parser.add_argument(
        "--po-mdl-subdir",
        type=str,
        default="kenya-hierarchical",
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
    parser.add_argument(
        "--axis-fn-names", 
        nargs="+", 
        default=["max-mahalanobis", "frac-expo"]
    )

    return parser


def multarm(args):
    """
    Make manuscript figures for the mult-arm simulated experiment.
    """
    output_dir = (
        Path(args.output_dir)
        / args.exp_subdir
        / args.n_arm_subdir
        / args.n_per_arm_subdir
        / args.n_z_subdir
    )
    print(output_dir)
    save_dir = Path(args.fig_dir) / args.exp_subdir
    save_prefix = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}"

    all_trial_sets, all_fitness_fns = [], []
    rand_mdl_save_subdirs = []
    for rand_mdl_subdir in args.rand_mdl_subdirs:
        print(f"\t{rand_mdl_subdir}")
        trial_fname = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}.pkl"
        trial_path = output_dir / rand_mdl_subdir / trial_fname

        trial_set, fitness_fn = multarm_trial_set(trial_path)

        save_subdir = (
            save_dir
            / args.n_arm_subdir
            / args.n_per_arm_subdir
            / args.n_z_subdir
            / rand_mdl_subdir
        )
        if not save_subdir.exists():
            save_subdir.mkdir(parents=True)

        rand_mdl_save_subdirs.append(save_subdir)
        all_trial_sets.append(trial_set)
        all_fitness_fns.append(fitness_fn)

    for fig_type in args.fig_type:
        if fig_type == "scatter_error_vs_score":
            make_scatter_error_vs_score_mult_fig(
                all_trial_sets, all_fitness_fns, rand_mdl_save_subdirs, save_prefix
            )
        elif fig_type == "covariate_balance":
            save_subdir = (
                save_dir / args.n_arm_subdir / args.n_per_arm_subdir / args.n_z_subdir
            )
            if not save_subdir.exists():
                save_subdir.mkdir(parents=True)

            for trial_set in all_trial_sets:
                make_covariate_balance_fig(
                    trial_set,
                    args.balance_fn_name,
                    save_subdir,
                    save_prefix,
                    legend=False,
                )
                make_covariate_pairwise_balance_fig(
                    trial_set, args.pairwise_balance_fn_name, save_prefix, save_subdir
                )
        elif fig_type == "hist_same_z_validity":
            make_hist_same_z_validity_mult_fig(
                all_trial_sets, save_subdir, save_prefix=save_prefix
            )
        elif fig_type == "hist_score":
            make_hist_score_mult_fig(
                all_trial_sets, save_subdir, save_prefix=save_prefix
            )
        else:
            print(f"Unrecognized figure type: {fig_type}, skipping")


def composition(args):
    """
    Make manuscript figures for the composition simulated experiment.
    """
    output_dir = (
        Path(args.output_dir)
        / args.exp_subdir
        / args.p_ctxts_subdir
        / args.n_per_arm_subdir
        / args.n_z_subdir
    )
    print(output_dir)
    save_dir = Path(args.fig_dir) / args.exp_subdir
    save_prefix = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}"

    all_trial_sets, rand_mdl_save_subdirs = [], []
    for rand_mdl_subdir in args.rand_mdl_subdirs:
        print(f"\t{rand_mdl_subdir}")
        trial_fname = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}.pkl"
        trial_path = output_dir / rand_mdl_subdir / trial_fname

        trial_set, _ = composition_trial_set(trial_path)
        save_subdir = (
            save_dir
            / args.p_ctxts_subdir
            / args.n_per_arm_subdir
            / args.n_z_subdir
            / rand_mdl_subdir
        )
        if not save_subdir.exists():
            save_subdir.mkdir(parents=True)

        rand_mdl_save_subdirs.append(save_subdir)
        all_trial_sets.append(trial_set)

    for fig_type in args.fig_type:
        if fig_type == "covariate_balance":
            save_subdir = (
                save_dir / args.p_ctxts_subdir / args.n_per_arm_subdir / args.n_z_subdir
            )
            if not save_subdir.exists():
                save_subdir.mkdir(parents=True)
            for trial_set in all_trial_sets:
                make_covariate_balance_fig(
                    trial_set,
                    args.balance_fn_name,
                    save_subdir,
                    save_prefix,
                    legend=False,
                )
                make_covariate_pairwise_balance_fig(
                    trial_set, args.pairwise_balance_fn_name, save_prefix, save_subdir
                )
        elif fig_type == "hist_same_z_validity":
            make_hist_same_z_validity_mult_fig(
                all_trial_sets, save_subdir, save_prefix=save_prefix
            )
        elif fig_type == "hist_score":
            make_hist_score_mult_fig(
                all_trial_sets, save_subdir, save_prefix=save_prefix
            )
        else:
            print(f"Unrecognized figure type: {fig_type}, skipping")


def kenya(args):
    """
    Make manuscript figures for the Kenya simulated experiment.
    """
    if "hist_2dscore_disagg" in args.fig_type:
        save_dir = (
            Path(args.fig_dir)
            / args.exp_subdir
            / args.param_subdir
            / args.po_mdl_subdir
            / args.n_z_subdir
        )
        save_prefix = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}"

        trial_set = kenya_trial_set_for_hist2d(
            args.param_subdir,
            args.output_dir,
            args.exp_subdir,
            args.net_mdl_subdir,
            args.po_mdl_subdir,
            args.n_z_subdir,
            args.rand_mdl_subdirs[0],
        )
        
        make_hist_2dscore_disagg_mult_dgp_fig(trial_set, save_dir, save_prefix)

    else:
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

        all_trial_sets, all_fitness_fns = [], []
        save_subdirs = []
        for rand_mdl_subdir in args.rand_mdl_subdirs:
            print(f"\t{rand_mdl_subdir}")
            trial_fname = f"data-rep-{args.data_rep}_run-seed-{args.run_seed}.pkl"
            trial_path = output_dir / rand_mdl_subdir / trial_fname

            if not trial_path.exists():
                print(f"{trial_path} does not exist. Skipping...")
                continue

            trial_set, fitness_fn = kenya_trial_set(trial_path)

            all_trial_sets.append(trial_set)
            all_fitness_fns.append(fitness_fn)

            save_subdir = (
                save_dir
                / args.param_subdir
                / args.net_mdl_subdir[0]
                / args.n_z_subdir
                / args.po_mdl_subdir
                / rand_mdl_subdir
            )

            if not save_subdir.exists():
                save_subdir.mkdir(parents=True)
            save_subdirs.append(save_subdir)

        for fig_type in args.fig_type:
            if fig_type == "scatter_error_vs_score":
                make_scatter_error_vs_score_mult_fig(
                    all_trial_sets, all_fitness_fns, save_subdirs, save_prefix
                )
            elif fig_type == "scatter_2dscore_disagg":
                make_scatter_2dscore_disagg_mult_fig(
                    all_trial_sets, args.axis_fn_names, save_subdirs, save_prefix
                )
            elif fig_type == "hist_same_z_validity":
                save_subdir = (
                    save_dir
                    / args.param_subdir
                    / args.net_mdl_subdir[0]
                    / args.n_z_subdir
                    / args.po_mdl_subdir
                )
                make_hist_same_z_validity_mult_fig(
                    all_trial_sets,
                    save_subdir,
                    save_prefix=save_prefix,
                    save_suffix=args.plt_suffix,
                )
            elif fig_type == "hist_score":
                save_subdir = (
                    save_dir
                    / args.param_subdir
                    / args.net_mdl_subdir[0]
                    / args.n_z_subdir
                    / args.po_mdl_subdir
                )
                make_hist_score_mult_fig(
                    all_trial_sets,
                    save_subdir,
                    save_prefix=save_prefix,
                    save_suffix=args.plt_suffix,
                )
            else:
                print(f"Unrecognized figure type: {fig_type}, skipping")


if __name__ == "__main__":
    args = parser().parse_args()

    if args.exp_subdir == "mult-arm":
        multarm(args)
    elif args.exp_subdir == "composition":
        composition(args)
    elif args.exp_subdir == "kenya":
        kenya(args)
