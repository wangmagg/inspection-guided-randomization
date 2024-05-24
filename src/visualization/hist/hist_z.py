from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from src.sims.trial import SimulatedTrial
from src.visualization.utils.aes import color_mapping


def _plot_hist_z(df_frac_z: pd.DataFrame, 
                 df_exp_z: pd.DataFrame,
                 save_dir: Path, 
                 save_prefix: str):
    """
    Helper function to make histogram of fraction of units
    assigned to treated arm (assuming binary treatment), for each randomization design

    Args:
        - df: dataframe containing fraction of units assigned to treatment arm for each randomization design
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """

    n_rand_mdls = len(df_frac_z["rand_mdl_name"].unique())
    fig_frac, ax_frac = plt.subplots(1, n_rand_mdls, figsize=(10*n_rand_mdls, 6))
    fig_exp, ax_exp = plt.subplots(1, n_rand_mdls, figsize=(10*n_rand_mdls, 6))

    max_y_frac_z_1 = df_frac_z.groupby("rand_mdl_name").size().max()
    max_y_exp_z = df_exp_z.groupby("rand_mdl_name").size().max()

    for i, rand_mdl_name in enumerate(df_frac_z["rand_mdl_name"].unique()):
        df_frac_z_rand_mdl = df_frac_z[df_frac_z["rand_mdl_name"] == rand_mdl_name]
        df_exp_z_rand_mdl = df_exp_z[df_exp_z["rand_mdl_name"] == rand_mdl_name]

        if "-" not in rand_mdl_name:
            rand_mdl_name_wrapped = rand_mdl_name
        else:
            rand_mdl_split = rand_mdl_name.split(" - ")
            rand_mdl_prefix = rand_mdl_split[0]
            rand_mdl_suffix = rand_mdl_split[-1]
            rand_mdl_name_wrapped = f"{rand_mdl_prefix}\n{rand_mdl_suffix}"

        # Make histogram for fraction of units assigned to treatment arm
        ax_frac[i].hist(
            df_frac_z_rand_mdl["frac_z_1"],
            color = color_mapping(rand_mdl_name),
            edgecolor="black",
            linewidth=1,
            bins = np.arange(-0.01, 1.01, 0.02))
        ax_frac[i].axvline(0.5, color="black", linestyle="--")
        ax_frac[i].set_xlim(0, 1)
        ax_frac[i].tick_params(axis='x', labelsize=16)
        ax_frac[i].set_ylim(0, max_y_frac_z_1)
        ax_frac[i].tick_params(axis='y', labelsize=16)
        ax_frac[i].set_xlabel(r"$\hat{P}(z_i = 1)$", fontsize=20)
        ax_frac[i].set_ylabel(r"Number of Allocations", fontsize=20)
        ax_frac[i].set_title(rand_mdl_name_wrapped, fontsize=20)
        
        # Make histogram for expected z_i across all treatment allocations in pool
        ax_exp[i].hist(
            df_exp_z_rand_mdl["expected_z"],
            color=color_mapping(rand_mdl_name),
            edgecolor="black",
            linewidth=1,
            bins = np.arange(-0.01, 1.01, 0.02))
        ax_exp[i].axvline(0.5, color="black", linestyle="--")
        ax_exp[i].set_xlim(0, 1)
        ax_exp[i].tick_params(axis='x', labelsize=16)
        ax_exp[i].set_ylim(0, max_y_exp_z)
        ax_exp[i].tick_params(axis='y', labelsize=16)
        ax_exp[i].set_xlabel(r"$E[z_i]$", fontsize=20)
        ax_exp[i].set_ylabel(r"Number of Units", fontsize=20)
        ax_exp[i].set_title(rand_mdl_name_wrapped, fontsize=20)

    # Save figure
    save_frac_fname = f"{save_prefix}_frac_z.svg"
    save_exp_fname = f"{save_prefix}_exp_z.svg"
    save_frac_path = save_dir / save_frac_fname
    save_exp_path = save_dir / save_exp_fname
    print(save_frac_path)
    print(save_exp_path)
    fig_frac.savefig(save_frac_path, bbox_inches="tight", dpi=200)
    fig_exp.savefig(save_exp_path, bbox_inches="tight", dpi=200)
    plt.close()

def _get_frac_z_1_df(trial_dict: Dict[str, SimulatedTrial]) -> pd.DataFrame:
    """
    Helper function to get dataframe of fraction of units assigned to treatment arm for each randomization design

    Args:
        - trial_dict: dictionary of trial objects for each randomization design
    Returns:
        dataframe containing fraction of units assigned to treatment arm for each randomization design
    """
    all_trial_frac_z_1 = []
    all_trial_exp_z = []

    # Iterate over randomization designs
    for trial_type, trial in trial_dict.items():
        # Get fraction of units assigned to treatment arm across 
        # all treatment allocations in pool
        z_pool = trial.z_pool
        frac_z_1 = np.mean(z_pool, axis=1)
        expected_z = np.mean(z_pool, axis=0)
        trial_frac_z_1_dict = {
            "rand_mdl_name": trial_type,
            "frac_z_1": frac_z_1,
        }
        trial_exp_z_dict = {
            "rand_mdl_name": trial_type,
            "expected_z": expected_z
        }
        trial_frac_z_1_df = pd.DataFrame.from_dict(trial_frac_z_1_dict)
        trial_exp_z_df = pd.DataFrame.from_dict(trial_exp_z_dict)

        all_trial_frac_z_1.append(trial_frac_z_1_df)
        all_trial_exp_z.append(trial_exp_z_df)

    all_trial_frac_z_1_df = pd.concat(all_trial_frac_z_1).reset_index(drop=True)
    all_trial_exp_z_df = pd.concat(all_trial_exp_z).reset_index(drop=True)

    return all_trial_frac_z_1_df, all_trial_exp_z_df

def make_hist_z_fig(
    trial_set: Dict[str, SimulatedTrial], 
    save_dir: Path,
    save_prefix: str
):
    """
    Make histogram of fraction of units assigned to treated arm (assuming binary treatment), 
    for each randomization design

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - save_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True
    # all_data_rep_frac_z_1_df_list = []
    # all_data_rep_exp_z_df_list = []

    # Iterate over data replicates
    # for data_rep, trial_dict in trials.items():

    # Get fraction of units assigned to treatment arm for each randomization design
    all_trial_frac_z_1_df, all_trial_exp_z_df = _get_frac_z_1_df(trial_set)

    # Make histogram
    _plot_hist_z(all_trial_frac_z_1_df, all_trial_exp_z_df, save_dir, save_prefix)

    # # Make histogram combining all data replicates
    # if len(trials) > 1:
    #     all_data_rep_frac_z_1_df = pd.concat(all_data_rep_frac_z_1_df_list, ignore_index=True)
    #     all_data_rep_exp_z_df = pd.concat(all_data_rep_exp_z_df_list, ignore_index=True)
    #     data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    #     _plot_hist_z(all_data_rep_frac_z_1_df, all_data_rep_exp_z_df, save_dir, data_reps_str)