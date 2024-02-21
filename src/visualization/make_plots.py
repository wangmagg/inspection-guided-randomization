import argparse
from pathlib import Path
import pickle

from src.design.fitness_functions import *
from src.sims.trial import *
from src.sims.run_kenya_trial import *
from src.sims.run_multarm_trial import *
from src.sims.run_composition_trial import *

import src.visualization.scatter.scatter_error_vs_score as scatter_error_vs_score 
import src.visualization.scatter.scatter_2dscore_disagg as scatter_2dscore_disagg 
import src.visualization.hist.hist_same_z_validity as hist_same_z_validity 
import src.visualization.hist.hist_z as hist_z
import src.visualization.box.box_covariate_distr_per_arm as box_covariate_distr_per_arm 
import src.visualization.box.box_covariate_balance as box_covariate_balance 
import src.visualization.box.box_covariate_pairwise_balance as box_covariate_pairwise_balance 


def _kenya_trial_sets(n_z_subdir, rand_mdl_subdir, include_settlement_cluster=True):
    trial_sets = {}
    fitness_fns = {}
    for trial_fname in rand_mdl_subdir.glob("*.pkl"):
        print(trial_fname)
        trial_set = {}

        # Load complete randomization results for school-level clustering
        cr_trial_fname = n_z_subdir / "rand-complete_cluster-school" / trial_fname.name
        with open(cr_trial_fname, "rb") as f:
            cr_trial = pickle.load(f)
        cr_trial.loader = SimulatedTrialLoader(cr_trial.config)
        cr_trial.set_data_from_config()
        cr_trial.set_design_from_config()
        trial_set["complete_cluster-school"] = cr_trial

        # Load complete randomization results for settlement-level clustering
        if include_settlement_cluster:
            cr_set_trial_fname = (
                n_z_subdir / "rand-complete_cluster-settlement" / trial_fname.name
            )
            with open(cr_set_trial_fname, "rb") as f:
                cr_set_trial = pickle.load(f)
            cr_set_trial.loader = SimulatedTrialLoader(cr_set_trial.config)
            cr_set_trial.set_data_from_config()
            cr_set_trial.set_design_from_config()
            trial_set["complete_cluster-settlement"] = cr_set_trial

        # Load restricted randomization results
        with open(trial_fname, "rb") as f:
            trial = pickle.load(f)
        trial.loader = SimulatedTrialLoader(trial.config)
        trial.set_data_from_config()
        trial.set_design_from_config()
        trial_set[trial.rand_mdl.name] = trial
        fitness_fns[int(trial_fname.stem)] = trial.rand_mdl.fitness_fn

        # Load restricted randomization with genetic search results
        genetic_trial_fname = (
            n_z_subdir
            / f"rand-restricted-genetic_{trial.rand_mdl.fitness_fn.name}_cluster-school"
            / trial_fname.name
        )

        if genetic_trial_fname.exists():
            with open(genetic_trial_fname, "rb") as f:
                genetic_trial = pickle.load(f)
            genetic_trial.loader = SimulatedTrialLoader(genetic_trial.config)
            genetic_trial.set_data_from_config()
            genetic_trial.set_design_from_config()
            trial_set[genetic_trial.rand_mdl.name] = genetic_trial

        trial_sets[int(trial_fname.stem)] = trial_set

    return trial_sets, fitness_fns


def kenya_randomization_plots(args):
    output_dir = Path(args.output_dir) / args.exp_subdir / args.param_subdir
    save_dir = Path(args.fig_dir) / args.exp_subdir / args.param_subdir

    for outcome_mdl_subdir in output_dir.glob(f"outcome-*"):
        for net_mdl_subdir in outcome_mdl_subdir.glob(f"net-*"):
            for n_z_subdir in net_mdl_subdir.glob(f"n-z-*"):
                for rand_mdl_subdir in n_z_subdir.glob(f"rand-restricted_*"):
                    print(rand_mdl_subdir.name)
                    for include_settlement_cluster in [True, False]:
                        trial_sets, fitness_fns = _kenya_trial_sets(
                            n_z_subdir, rand_mdl_subdir, include_settlement_cluster
                        )

                        save_subdir = (
                            save_dir
                            / net_mdl_subdir.name
                            / n_z_subdir.name
                            / rand_mdl_subdir.name
                            / f"settlement-cluster-{include_settlement_cluster}"
                        )
                        if not save_subdir.exists():
                            save_subdir.mkdir(parents=True)

                        if "scatter_error_vs_score" in args.fig_types:
                            scatter_error_vs_score.make_scatter_error_vs_score_fig(
                                trial_sets, fitness_fns, save_subdir
                            )
                        if "scatter_2dscore_disagg" in args.fig_types:
                            scatter_2dscore_disagg.make_scatter_2dscore_disagg_fig(
                                trial_sets, args.axis_fns, save_subdir
                            )
                        if "hist_same_z_validity" in args.fig_types:
                            hist_same_z_validity.make_hist_same_z_validity_fig(trial_sets, save_subdir)

                        if "hist_z" in args.fig_types:
                            hist_z.make_hist_z_fig(trial_sets, save_subdir)


def _multarm_trial_sets(n_z_subdir, rand_mdl_subdir):
    trial_sets = {}
    fitness_fns = {}
    for trial_fname in rand_mdl_subdir.glob("*.pkl"):
        print(trial_fname)
        trial_set = {}

        # Load complete randomization results
        cr_trial_fname = n_z_subdir / "rand-complete" / trial_fname.name
        if cr_trial_fname.exists():
            with open(cr_trial_fname, "rb") as f:
                cr_trial = pickle.load(f)
            cr_trial.loader = SimulatedTrialLoader(cr_trial.config)
            cr_trial.mapping, cr_trial.use_cols = None, None
            cr_trial.set_data_from_config()
            cr_trial.set_design_from_config()
            trial_set["complete"] = cr_trial

        # Load quickblock randomization results
        qb_trial_fname = n_z_subdir / "rand-quick-block" / trial_fname.name

        if qb_trial_fname.exists():
            with open(qb_trial_fname, "rb") as f:
                qb_trial = pickle.load(f)
            qb_trial.loader = SimulatedTrialLoader(qb_trial.config)
            qb_trial.mapping, qb_trial.use_cols = None, None
            qb_trial.set_data_from_config()
            qb_trial.set_design_from_config()
            trial_set["quick-block"] = qb_trial

        # Load restricted randomization results
        with open(trial_fname, "rb") as f:
            trial = pickle.load(f)
        trial.loader = SimulatedTrialLoader(trial.config)
        trial.mapping, trial.use_cols = None, None
        trial.set_data_from_config()
        trial.set_design_from_config()
        trial_set[trial.rand_mdl.name] = trial
        fitness_fns[int(trial_fname.stem)] = trial.rand_mdl.fitness_fn

        # Load restricted randomization with genetic search results
        genetic_trial_fname = (
            n_z_subdir
            / f"rand-restricted-genetic_{trial.rand_mdl.fitness_fn.name}"
            / trial_fname.name
        )

        if genetic_trial_fname.exists():
            with open(genetic_trial_fname, "rb") as f:
                genetic_trial = pickle.load(f)
            genetic_trial.loader = SimulatedTrialLoader(genetic_trial.config)
            genetic_trial.mapping, genetic_trial.use_cols = None, None
            genetic_trial.set_data_from_config()
            genetic_trial.set_design_from_config()
            trial_set[genetic_trial.rand_mdl.name] = genetic_trial

        trial_sets[int(trial_fname.stem)] = trial_set

    return trial_sets, fitness_fns


def multarm_randomization_plots(args):
    output_dir = Path(args.output_dir) / args.exp_subdir
    save_dir = Path(args.fig_dir) / args.exp_subdir

    for n_arm_subdir in output_dir.glob("arms-*"):
        for n_per_arm_subdir in n_arm_subdir.glob("n-per-arm-*"):
            for n_z_subdir in n_per_arm_subdir.glob("n-z-*"):
                data_rep_to_trials = {}
                for rand_mdl_subdir in n_z_subdir.glob(f"rand-restricted_*"):
                    trial_sets, fitness_fns = _multarm_trial_sets(
                        n_z_subdir, rand_mdl_subdir
                    )

                    for data_rep in trial_sets.keys():
                        if data_rep not in data_rep_to_trials:
                            data_rep_to_trials[data_rep] = trial_sets[data_rep]
                        else:
                            data_rep_to_trials[data_rep].update(trial_sets[data_rep])

                    if "scatter_error_vs_score" in args.fig_types:
                        save_subdir = (
                            save_dir
                            / n_arm_subdir.name
                            / n_per_arm_subdir.name
                            / n_z_subdir.name
                            / rand_mdl_subdir.name
                        )
                        scatter_error_vs_score.make_scatter_error_vs_score_fig(
                            trial_sets, fitness_fns, save_subdir
                        )

                save_subdir = (
                    save_dir
                    / n_arm_subdir.name
                    / n_per_arm_subdir.name
                    / n_z_subdir.name
                )
                if not save_subdir.exists():
                    save_subdir.mkdir(parents=True)

                if "cov_pairwise_balance" in args.fig_types:
                    box_covariate_pairwise_balance.make_covariate_pairwise_balance_fig(
                        data_rep_to_trials, args.balance_fn_name, save_subdir
                    )
                if "cov_balance" in args.fig_types:
                    box_covariate_balance.make_covariate_balance_fig(
                        data_rep_to_trials, args.balance_fn_name, save_subdir
                    )
                if "cov_distr" in args.fig_types:
                    box_covariate_distr_per_arm.make_covariate_distr_per_arm_fig(data_rep_to_trials, save_subdir)

def _composition_trial_sets(n_z_subdir, rand_mdl_subdir):
    trial_sets = {}
    fitness_fns = {}
    for trial_fname in rand_mdl_subdir.glob("*.pkl"):
        print(trial_fname)
        trial_set = {}

        # Load complete randomization results
        cr_trial_fname = n_z_subdir / "rand-group-formation" / trial_fname.name

        if cr_trial_fname.exists():
            with open(cr_trial_fname, "rb") as f:
                cr_trial = pickle.load(f)
            cr_trial.loader = SimulatedTrialLoader(cr_trial.config)
            cr_trial.set_data_from_config()
            cr_trial.mapping = None 
            cr_trial.set_design_from_config()
            trial_set["group-formation"] = cr_trial

        # Load restricted randomization results
        with open(trial_fname, "rb") as f:
            trial = pickle.load(f)
        trial.loader = SimulatedTrialLoader(trial.config)
        trial.set_data_from_config()
        trial.mapping = None
        trial.set_design_from_config()
        trial_set[trial.rand_mdl.name] = trial
        fitness_fns[int(trial_fname.stem)] = trial.rand_mdl.fitness_fn

        # Load restricted randomization with genetic search results
        genetic_trial_fname = (
            n_z_subdir
            / f"rand-restricted-genetic_{trial.rand_mdl.fitness_fn.name}"
            / trial_fname.name
        )

        if genetic_trial_fname.exists():
            with open(genetic_trial_fname, "rb") as f:
                genetic_trial = pickle.load(f)
            genetic_trial.loader = SimulatedTrialLoader(genetic_trial.config)
            genetic_trial.set_data_from_config()
            genetic_trial.mapping = None
            genetic_trial.set_design_from_config()
            trial_set[genetic_trial.rand_mdl.name] = genetic_trial

        trial_sets[int(trial_fname.stem)] = trial_set

    return trial_sets, fitness_fns

def composition_randomization_plots(args):
    output_dir = Path(args.output_dir) / args.exp_subdir
    save_dir = Path(args.fig_dir) / args.exp_subdir

    for p_ctxts_subdir in output_dir.glob('p-comps-*'):
        for n_per_arm_subdir in p_ctxts_subdir.glob('n-per-arm-*'):
            for n_z_subdir in n_per_arm_subdir.glob('n-z-*'):
                data_rep_to_trials = {}
                for rand_mdl_subdir in n_z_subdir.glob(f"rand-*restricted_*"):
                    trial_sets, fitness_fns = _composition_trial_sets(
                        n_z_subdir, rand_mdl_subdir
                    )

                    for data_rep in trial_sets.keys():
                        if data_rep not in data_rep_to_trials:
                            data_rep_to_trials[data_rep] = trial_sets[data_rep]
                        else:
                            data_rep_to_trials[data_rep].update(trial_sets[data_rep])

                    if "scatter_error_vs_score" in args.fig_types:
                        save_subdir = (
                            save_dir
                            / p_ctxts_subdir.name
                            / n_per_arm_subdir.name
                            / n_z_subdir.name
                            / rand_mdl_subdir.name
                        )
                        scatter_error_vs_score.make_scatter_error_vs_score_fig(
                            trial_sets, fitness_fns, save_subdir
                        )

                save_subdir = save_dir / p_ctxts_subdir.name / n_per_arm_subdir.name / n_z_subdir.name
                if not save_subdir.exists():
                    save_subdir.mkdir(parents=True)
                    
                if "cov_pairwise_balance" in args.fig_types:
                    box_covariate_pairwise_balance.make_covariate_pairwise_balance_fig(data_rep_to_trials, args.balance_fn_name, save_subdir)
                if "cov_balance" in args.fig_types:
                    box_covariate_balance.make_covariate_balance_fig(data_rep_to_trials,  args.balance_fn_name, save_subdir)
                if "cov_distr" in args.fig_types:
                    box_covariate_distr_per_arm.make_covariate_distr_per_arm_fig(data_rep_to_trials, save_subdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--exp-subdir", type=str, default="kenya")
    parser.add_argument("--param-subdir", type=str, default="params_sigma_scale-1.0")

    parser.add_argument("--fig-types", nargs="+", type=str)
    parser.add_argument("--axis-fns", nargs="+", type=str)
    parser.add_argument("--balance-fn-name", type=str)

    parser.add_argument("--fig-dir", type=str, default="figs")
    

    args = parser.parse_args()

    if args.exp_subdir == "kenya":
        kenya_randomization_plots(args)
    if "mult-arm" in args.exp_subdir:
        multarm_randomization_plots(args)
    if args.exp_subdir == 'composition':
        composition_randomization_plots(args)
