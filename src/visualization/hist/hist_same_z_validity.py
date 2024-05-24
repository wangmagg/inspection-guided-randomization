from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple

from src.sims.trial import SimulatedTrial
from src.visualization.utils.aes import get_hue_order, get_palette
from typing import Dict


def _rand_mdl_name_no_ff(rand_mdl_name: str) -> str:
    return rand_mdl_name.split("-")[0].strip()


def _subplot_hist_same_z_validity(df_ff: pd.DataFrame, ax: plt.Axes, ax_title: str):
    uniq_rand_mdl_names = df_ff["rand_mdl_name"].unique()

    df_ff = df_ff.copy()
    df_ff["rand_mdl_name"] = df_ff["rand_mdl_name"].apply(
        lambda x: _rand_mdl_name_no_ff(x)
    )
    hue_order = get_hue_order(uniq_rand_mdl_names, ff_in_name=False)
    palette = get_palette(uniq_rand_mdl_names, ff_in_name=False)

    # Make histogram
    sns.histplot(
        df_ff,
        x="same_z",
        hue="rand_mdl_name",
        hue_order=hue_order,
        palette=palette,
        binwidth=0.01,
        element="step",
        fill=False,
        alpha=0.6,
        kde=True,
        line_kws={"linewidth": 2},
        stat="probability",
        common_norm=False,
        ax=ax,
    )
    ax.axvline(
        1 / df_ff["n_arms"].values[0], color="black", linestyle="--", linewidth=1
    )
    ax.set_title(ax_title, fontsize=22)
    if "IGR (GFR)" in df_ff["rand_mdl_name"].unique():
        ax.set_xlabel(r"$\hat{P}(z_i = z_j, g_i = g_j)$", fontsize=22)
    else:
        ax.set_xlabel(r"$\hat{P}(z_i = z_j)$", fontsize=22)
    ax.set_ylabel(r"Fraction of Pairs $(i, j)$", fontsize=22)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16, labelleft=True)

    legend = ax.legend_
    handles = legend.legend_handles
    for h in handles:
        h.set_linewidth(2)
        h.set_alpha(1)
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=16,
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
    )


def _plot_hist_same_z_validity(
    df: pd.DataFrame, save_dir: Path, save_prefix: str, save_suffix: str = None
) -> None:
    """
    Helper function to make histogram of how frequently two units are assigned
    to the same treatment arm, for each randomization designs

    Args:
        - df: dataframe containing "same-z" frequencies for each randomization design
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """

    ff_names = df["ff_name"].unique()
    fig, axs = plt.subplots(
        ncols=len(ff_names), figsize=(6 * len(ff_names), 5), sharex=True, sharey=True
    )
    plt.subplots_adjust(wspace=0.4)

    if len(ff_names) == 1:
        _subplot_hist_same_z_validity(df, axs, ff_names[0])
    else:
        for ff_name, ax in zip(ff_names, axs):
            df_ff = df[df["ff_name"] == ff_name].copy()
            _subplot_hist_same_z_validity(df_ff, ax, ff_name)

    # Save figure
    if save_suffix is not None:
        save_fname = f"{save_prefix}_{save_suffix}_same_z_validity.svg"
    else:
        save_fname = f"{save_prefix}_same_z_validity.svg"
    save_path = save_dir / save_fname
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()


def _get_same_z_df(trial_dict: Dict[str, SimulatedTrial]) -> pd.DataFrame:
    """
    Calculates the frequency with which two units are assigned to the same treatment arm
    for each randomization design

    Args:
        - trial_dict: dictionary mapping randomization design name to trial object
        - save_dir: directory to save figure
    Returns
        dataframe containing pairwise "same-z" frequencies for each randomization design
    """
    all_trial_same_z = []

    # Iterate through each randomization design
    ff_name = None
    for trial_type, trial in trial_dict.items():
        # Map settlement to schools for simulated Kenya trial
        # if randomization is on settlement-level
        if "cluster-settlement" in trial_type:
            mapping = (
                trial.X[["settlement_id", "school_id"]]
                .groupby("school_id")
                .agg(max)["settlement_id"]
            )
            z_pool = np.vstack([z[mapping] for z in trial.z_pool])
        else:
            z_pool = trial.z_pool

        # Get frequency with which two units are assigned to the same treatment arm
        # across treatment allocations in the pool
        same_z_mat = np.zeros((z_pool.shape[1], z_pool.shape[1]))
        for i, j in combinations(range(z_pool.shape[1]), 2):
            p_same = np.mean(z_pool[:, i] == z_pool[:, j])
            same_z_mat[i, j] = p_same

        same_z_upper = same_z_mat[np.triu_indices(same_z_mat.shape[0], k=1)]
        same_z_upper_flat_norm = same_z_upper.flatten()
        trial_same_z_dict = {
            "rand_mdl_name": trial_type,
            "same_z": same_z_upper_flat_norm,
            "n_arms": np.max(trial.z_pool) + 1,
        }
        trial_same_z_df = pd.DataFrame.from_dict(trial_same_z_dict)
        all_trial_same_z.append(trial_same_z_df)

        if ff_name is None and "IGR" in trial_type:
            ff_name = trial_type.split("-")[1].strip()

    all_trial_same_z_df = pd.concat(all_trial_same_z).reset_index(drop=True)
    all_trial_same_z_df["ff_name"] = ff_name

    return all_trial_same_z_df


def make_hist_same_z_validity_fig(
    trial_set: Dict[str, SimulatedTrial], save_dir: Path, save_prefix: str
) -> None:
    """
    Make histogram of how frequently two units are assigned to the same treatment arm,
    for each randomization design

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - save_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True
    all_data_rep_df_list = []

    same_z_df = _get_same_z_df(trial_set)

    # Make histogram
    _plot_hist_same_z_validity(same_z_df, save_dir, save_prefix)

    all_data_rep_df_list.append(same_z_df)

    # # Make histogram combining all data replicates
    # if len(trials) > 1:
    #     all_data_rep_df = pd.concat(all_data_rep_df_list, ignore_index=True)
    #     data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    #     _plot_hist_same_z_validity(all_data_rep_df, save_dir, data_reps_str)


def make_hist_same_z_validity_mult_fig(
    trial_sets: List[Dict[str, SimulatedTrial]],
    save_dir: Path,
    save_suffix: str = None,
    save_prefix: str = None,
):

    same_z_df_list = []
    for trial_set in trial_sets:
        same_z_df = _get_same_z_df(trial_set)
        same_z_df_list.append(same_z_df)

    same_z_df = pd.concat(same_z_df_list)
    _plot_hist_same_z_validity(same_z_df, save_dir, save_prefix, save_suffix)
