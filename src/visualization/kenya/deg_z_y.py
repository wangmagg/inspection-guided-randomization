from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy.stats import pearsonr

from src.sims.trial import SimulatedTrial
from src.visualization.utils.aes import color_mapping, get_hue_order, get_palette
from src.visualization.hist.hist_z import _get_frac_z_1_df


def _get_degs(X, A):
    n = X["school_id"].nunique()
    mapping = X["school_id"].values

    degs = np.zeros(n)
    for schl in range(n):
        indivs_in_schl_mask = mapping == schl
        indivs_in_schl_adj = np.sum(A[indivs_in_schl_mask,], axis=0)
        degs[schl] = np.sum(indivs_in_schl_adj)

    return degs


def _get_y_means(X, y_0):
    n = X["school_id"].nunique()
    mapping = X["school_id"].values

    y_mean_in_schl = np.zeros(n)
    for schl in range(n):
        indivs_in_schl_mask = mapping == schl
        y_mean_in_schl[schl] = np.mean(y_0[indivs_in_schl_mask])

    return y_mean_in_schl


def _subplot_deg_z(ax, degs, exp_z, clr, ax_title=None, leg_lbl=None):
    ax.scatter(degs, exp_z, color=clr, s=20, label=leg_lbl)
    ax.set_xlabel(r"Total Degree", fontsize=20)
    ax.set_ylabel(r"$E[Z_i]$", fontsize=22)

    ax.set_ylim(0, 1)

    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    if ax_title is not None:
        ax.set_title(ax_title, fontsize=22)

    return ax


def _subplot_deg_y(
    ax,
    degs,
    y,
    clr,
    school_avg=True,
    degree_suffix="Total",
    ax_title=None,
    leg_lbl=None,
):
    ax.scatter(degs, y, color=clr, s=10, alpha=0.5, label=leg_lbl)
    if school_avg:
        ax.set_xlabel(f"Sch Summed Degree - {degree_suffix}", fontsize=20)
        ax.set_ylabel(r"$\bar{Y}$", fontsize=22)
    else:
        ax.set_xlabel(f"Degree - {degree_suffix}", fontsize=20)
        ax.set_ylabel(r"$Y_i$", fontsize=22)

    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    if ax_title is not None:
        ax.set_title(ax_title, fontsize=22)

    return ax


def _subplot_z_y(ax, exp_z, y_mean_in_schl, clr, ax_title=None, leg_lbl=None):
    ax.scatter(exp_z, y_mean_in_schl, color=clr, s=20, label=leg_lbl)
    ax.set_xlabel(r"$E[Z_i]$", fontsize=20)
    ax.set_ylabel(r"$\bar{Y}$", fontsize=22)

    ax.set_xlim(0, 1)

    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    if ax_title is not None:
        ax.set_title(ax_title, fontsize=22)

    return ax


def _plot_deg_z(
    df_exp_z: pd.DataFrame,
    X: pd.DataFrame,
    A: np.ndarray,
    save_dir: Path,
    save_prefix: str,
    by_settlement: bool = False,
):
    degs = _get_degs(X, A)

    if by_settlement:
        n_sets = len(X["settlement_id"].unique())
        fig, ax = plt.subplots(
            1, n_sets, figsize=(8 * n_sets, 8), sharex=True, sharey=True
        )

        for i, set_name in enumerate(X["settlement"].unique()):
            set_mask = X.groupby("school_id")["settlement"].first() == set_name

            for rand_mdl_name in df_exp_z["rand_mdl_name"].unique():
                df_exp_z_rand_mdl = df_exp_z[df_exp_z["rand_mdl_name"] == rand_mdl_name]
                exp_z_set = df_exp_z_rand_mdl["expected_z"].values[set_mask]
                deg_set = degs[set_mask]

                ax[i] = _subplot_deg_z(
                    ax[i],
                    deg_set,
                    exp_z_set,
                    color_mapping(rand_mdl_name),
                    set_name.capitalize(),
                    leg_lbl=rand_mdl_name,
                )

            if i == 0:
                ax[i].legend(
                    title=None,
                    fontsize=20,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.4),
                    edgecolor="black",
                )
            else:
                ax[i].legend().remove()

        save_path = save_dir / f"{save_prefix}_deg_z_by_set.svg"

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        for i, rand_mdl_name in enumerate(df_exp_z["rand_mdl_name"].unique()):
            df_exp_z_rand_mdl = df_exp_z[df_exp_z["rand_mdl_name"] == rand_mdl_name]
            exp_z = df_exp_z_rand_mdl["expected_z"].values

            ax = _subplot_deg_z(
                ax,
                degs,
                exp_z,
                color_mapping(rand_mdl_name),
                ax_title=None,
                leg_lbl=rand_mdl_name,
            )

        ax.legend(
            title=None,
            fontsize=20,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.4),
            edgecolor="black",
        )
        save_path = save_dir / f"{save_prefix}_deg_z.svg"

    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _plot_z_y(
    df_exp_z: pd.DataFrame,
    X: pd.DataFrame,
    y_0: np.ndarray,
    save_dir: Path,
    save_prefix: str,
    by_settlement: bool = False,
):
    y_mean_in_schl = _get_y_means(X, y_0)

    if by_settlement:
        n_sets = len(X["settlement_id"].unique())
        fig, ax = plt.subplots(
            1, n_sets, figsize=(8 * n_sets, 8), sharex=True, sharey=True
        )

        for i, set_name in enumerate(X["settlement"].unique()):
            set_mask = X.groupby("school_id")["settlement"].first() == set_name

            for rand_mdl_name in df_exp_z["rand_mdl_name"].unique():
                df_exp_z_rand_mdl = df_exp_z[df_exp_z["rand_mdl_name"] == rand_mdl_name]
                exp_z_set = df_exp_z_rand_mdl["expected_z"].values[set_mask]
                y_mean_set = y_mean_in_schl[set_mask]

                ax[i] = _subplot_z_y(
                    ax[i],
                    exp_z_set,
                    y_mean_set,
                    color_mapping(rand_mdl_name),
                    set_name.capitalize(),
                    leg_lbl=rand_mdl_name,
                )

            if i == 0:
                ax[i].legend(
                    title=None,
                    fontsize=20,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.4),
                    edgecolor="black",
                )
            else:
                ax[i].legend().remove()

        save_path = save_dir / f"{save_prefix}_y_z_by_set.svg"

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        for i, rand_mdl_name in enumerate(df_exp_z["rand_mdl_name"].unique()):
            df_exp_z_rand_mdl = df_exp_z[df_exp_z["rand_mdl_name"] == rand_mdl_name]
            exp_z = df_exp_z_rand_mdl["expected_z"].values

            ax = _subplot_z_y(
                ax,
                exp_z,
                y_mean_in_schl,
                color_mapping(rand_mdl_name),
                ax_title=None,
                leg_lbl=rand_mdl_name,
            )

        ax.legend(
            title=None,
            fontsize=20,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.4),
            edgecolor="black",
        )

        save_path = save_dir / f"{save_prefix}_y_z.svg"

    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)


def make_deg_z_fig(
    trial_set: Dict[str, SimulatedTrial],
    save_dir: Path,
    save_prefix: str,
    by_settlement: bool = False,
):
    """
    Make scatterplot of expected Z_i vs. total degree (sum of degrees of individuals in each school) for each randomization design
    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - save_dir: directory to save figure
    """
    # Iterate over data replicates
    # for data_rep, trial_dict in trials.items():

    X = list(trial_set.values())[0].X
    A = list(trial_set.values())[0].A

    # Get fraction of units assigned to treatment arm for each randomization design
    _, all_trial_exp_z_df = _get_frac_z_1_df(trial_set)

    # Make histogram
    _plot_deg_z(
        all_trial_exp_z_df,
        X,
        A,
        save_dir,
        save_prefix,
        by_settlement=by_settlement,
    )


def make_deg_y_fig(
    X: pd.DataFrame,
    A: np.ndarray,
    y_0: np.ndarray,
    save_dir: Path,
    school_avg: bool = True,
    by_settlement: bool = False,
):

    if school_avg:
        degs = _get_degs(X, A)
        y = _get_y_means(X, y_0)
    else:
        degs_tot = A.sum(axis=1)
        degs_same_sch = np.zeros(X.shape[0])
        degs_diff_sch = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sch_id = X.loc[i, "school_id"]
            sch_id_mask = X["school_id"] == sch_id
            degs_same_sch[i] = A[i, sch_id_mask].sum()
            degs_diff_sch[i] = A[i, :].sum() - A[i, sch_id_mask].sum()
        degs_diff_set = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            set_id = X.loc[i, "settlement_id"]
            set_id_mask = X["settlement_id"] == set_id
            degs_diff_set[i] = A[i, :].sum() - A[i, set_id_mask].sum()
        deg_types = ["Tot", "Same Sch", "Diff Sch", "Diff Set"]
        y = y_0
    all_degs = [degs_tot, degs_same_sch, degs_diff_sch, degs_diff_set]

    if by_settlement:
        n_sets = len(X["settlement_id"].unique())
        fig, ax = plt.subplots(
            len(all_degs),
            n_sets,
            figsize=(8 * n_sets, 8 * len(all_degs)),
            sharex=True,
            sharey=True,
        )

        for i, (degs, deg_type) in enumerate(zip(all_degs, deg_types)):
            for j, set_name in enumerate(X["settlement"].unique()):
                if school_avg:
                    set_mask = X.groupby("school_id")["settlement"].first() == set_name
                else:
                    set_mask = X["settlement"] == set_name
                degs_set = degs[set_mask]
                y_set = y[set_mask]

                stat, pval = pearsonr(degs_set, y_set)
                ax_title = f"{set_name.capitalize()} - (r={stat:.3f}, p={pval:.3f})"

                ax[i][j] = _subplot_deg_y(
                    ax[i][j],
                    degs_set,
                    y_set,
                    clr="black",
                    school_avg=school_avg,
                    ax_title=ax_title,
                    degree_suffix=deg_type,
                )

        save_path = save_dir / f"deg_y_by_set-{school_avg}.svg"

    else:
        fig, ax = plt.subplots(
            len(all_degs), 1, figsize=(8, 8 * len(all_degs)), sharex=True, sharey=True
        )

        for i, (degs, deg_type) in enumerate(zip(all_degs, deg_types)):
            stat, pval = pearsonr(degs, y)
            ax_title = f"r={stat:.3f}, p={pval:.3f}"
            ax[i] = _subplot_deg_y(
                ax[i],
                degs,
                y,
                clr="black",
                school_avg=school_avg,
                ax_title=ax_title,
                degree_suffix=deg_type,
            )

        save_path = save_dir / f"deg_y_sch-avg-{school_avg}.svg"

    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def make_y_z_fig(
    trial_set: Dict[str, SimulatedTrial],
    save_dir: Path,
    save_prefix: str,
    by_settlement: bool = False,
):
    """
    Make scatterplot of Ybar versus expected Z_i for each randomization design
    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - save_dir: directory to save figure
    """
    # Iterate over data replicates
    # for data_rep, trial_dict in trials.items():

    X = list(trial_set.values())[0].X
    y_0 = list(trial_set.values())[0].y_0

    # Get fraction of units assigned to treatment arm for each randomization design
    _, all_trial_exp_z_df = _get_frac_z_1_df(trial_set)

    # Make histogram
    _plot_z_y(
        all_trial_exp_z_df,
        X,
        y_0,
        save_dir,
        save_prefix,
        by_settlement=by_settlement,
    )
