import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, Tuple, List

from src.sims.trial import SimulatedTrial
from src.design.randomization_designs import GraphRandomization
from src.visualization.utils.aes import get_hue_order, get_palette
from src.design.fitness_functions import Fitness


def _plot_scatter_error_vs_score(
    df: pd.DataFrame,
    save_dir: Path,
    save_prefix: str,
    ylim: Tuple[float, float] = None,
    legend: bool = True,
) -> None:
    """
    Helper function to make scatterplot of residuals versus fitness scores, for each randomization design

    Args:
        - df: dataframe containing scores and residuals for each randomization design
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """
    uniq_rand_mdl_names = df["rand_mdl_name"].unique()
    hue_order = get_hue_order(uniq_rand_mdl_names, ff_in_name=False)
    palette = get_palette(uniq_rand_mdl_names, ff_in_name=False)

    df["rand_mdl_name"] = df["rand_mdl_name"].apply(lambda x: x.split("-")[0].strip())
    igr_in_name_mask = ["IGR" in name for name in uniq_rand_mdl_names]
    ff_name = uniq_rand_mdl_names[igr_in_name_mask][0].split("-")[1].strip()

    jnt_grid = sns.JointGrid(
        data=df,
        x="score",
        y="error",
        hue="rand_mdl_name",
        height=5,
        hue_order=hue_order,
        palette=palette,
    )
    # Make scatterplot with marginal histograms
    jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.5)
    jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
    jnt_grid.ax_joint.axhline(y=0, color="black", linestyle="--")
    jnt_grid.set_axis_labels(ff_name, r"$\hat{\tau} - \tau$", fontsize=22)

    jnt_grid.ax_joint.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    jnt_grid.ax_joint.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    jnt_grid.ax_joint.tick_params(axis='both', which='major', labelsize=18)

    if legend:
        jnt_grid.ax_joint.legend(
            title=None,
            markerscale=0.75,
            fontsize=14,
            handlelength=1,
            labelspacing=0.2,
            handletextpad=0.2,
            borderpad=0.2,
            borderaxespad=0.2
        )
    else:
        jnt_grid.ax_joint.legend_.remove()
    if ylim is None:
        max_abs_residual = df["error"].abs().max()
        jnt_grid.ax_joint.set_ylim(max_abs_residual * -1.1, max_abs_residual * 1.1)
    else:
        jnt_grid.ax_joint.set_ylim(ylim)

    # Save figure
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_fname = f"{save_prefix}_scatter_error_vs_score.svg"
    save_path = save_dir / save_fname
    print(save_path)
    jnt_grid.savefig(save_path, bbox_inches="tight", dpi=200, transparent=True)
    plt.close()


def _plot_scatter_error_se_vs_score_bucket(
    df: pd.DataFrame,
    save_dir: Path,
    save_prefix: str,
    ymax: float = None,
    legend: bool = True,
) -> None:
    """
    Helper function to make scatterplot of standard errors of residuals versus fitness scores,
    for each randomization design

    Args:
        - df: dataframe containing scores and residuals for each randomization design
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """
    uniq_rand_mdl_names = df["rand_mdl_name"].unique()
    hue_order = get_hue_order(uniq_rand_mdl_names, ff_in_name=False)
    palette = get_palette(uniq_rand_mdl_names, ff_in_name=False)

    df["rand_mdl_name"] = df["rand_mdl_name"].apply(lambda x: x.split("-")[0].strip())
    igr_in_name_mask = ["IGR" in name for name in uniq_rand_mdl_names]
    ff_name = uniq_rand_mdl_names[igr_in_name_mask][0].split("-")[1].strip()

    jnt_grid = sns.JointGrid(
        data=df,
        x="score_bin_mid",
        y="se",
        hue="rand_mdl_name",
        height=5,
        hue_order=hue_order,
        palette=palette,
    )
    # Make scatterplot with marginal histograms
    jnt_grid.plot_joint(
        sns.scatterplot, s=14, linewidth=0.1, edgecolor="black", alpha=0.8
    )
    jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
    jnt_grid.set_axis_labels(ff_name, r"$SE_{\hat{\tau} - \tau}$", fontsize=22)

    xmin = np.min((0, df["score_bin_mid"].min()))
    jnt_grid.ax_joint.set_xlim(xmin, jnt_grid.ax_joint.get_xlim()[1])

    if ymax is not None:
        jnt_grid.ax_joint.set_ylim(0, ymax)
    else:
        jnt_grid.ax_joint.set_ylim(0, jnt_grid.ax_joint.get_ylim()[1])

    jnt_grid.ax_joint.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    jnt_grid.ax_joint.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    jnt_grid.ax_joint.tick_params(axis='both', which='major', labelsize=18)

    if legend:
        jnt_grid.ax_joint.legend(
            title=None,
            markerscale=0.75,
            fontsize=14,
            handlelength=1,
            labelspacing=0.2,
            handletextpad=0.2,
            borderpad=0.2,
            borderaxespad=0.2
        )
    else:
        jnt_grid.ax_joint.legend_.remove()

    # Save figure
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_fname = f"{save_prefix}_scatter_se_vs_score.svg"
    save_path = save_dir / save_fname
    print(save_path)
    jnt_grid.savefig(save_path, bbox_inches="tight", dpi=200, transparent=True)
    plt.close()


def _get_rand_score_error_df(
    trial_dict: Dict[str, SimulatedTrial], fitness_fn: Fitness
) -> pd.DataFrame:
    """
    Helper function to get dataframe of fitness scores and residuals for each randomization design

    Args:
        - trial_dict: dictionary of randomization designs
        - fitness_fn: fitness function
    Returns
        dataframe containing scores and residuals for each randomization design
    """
    rand_mdl_names = list(trial_dict.keys())
    z_pool_sizes = [trial.z_pool.shape[0] for trial in trial_dict.values()]
    all_rand_mdl_names = np.repeat(rand_mdl_names, z_pool_sizes)

    all_z_pool = []

    # Iterate over randomization designs and
    # get treatment allocation pool for each
    for rand_mdl_name, trial in trial_dict.items():
        # Map settlement to schools for simulated Kenya trial
        # if randomization is on settlement-level
        if "Set" in rand_mdl_name:
            mapping = (
                trial.X[["settlement_id", "school_id"]]
                .groupby("school_id")
                .agg(max)["settlement_id"]
            )
            all_z_pool.append(np.vstack([z[mapping] for z in trial.z_pool]))
        # Map cluster to individuals if used graph randomization
        elif isinstance(trial.rand_mdl, GraphRandomization):
            all_z_pool.append(np.vstack([z[trial.mapping] for z in trial.z_pool]))
        else:
            all_z_pool.append(trial.z_pool)

    # Get fitness scores for each treatment allocation
    all_scores = fitness_fn(np.vstack(all_z_pool))

    # Get residuals for effect estimates under each treatment allocation
    all_errors = []
    for trial in trial_dict.values():
        if np.isscalar(trial.tau_true):
            error = trial.tau_hat_pool - trial.tau_true
        else:
            # If multi-arm trial, calculate mean residual across arms
            error = np.mean(trial.tau_hat_pool - trial.tau_true, axis=1)
        all_errors.append(error)

    all_errors = np.hstack(all_errors)

    all_trial_df = pd.DataFrame.from_dict(
        {"rand_mdl_name": all_rand_mdl_names, "score": all_scores, "error": all_errors}
    )

    return all_trial_df


def _get_rand_score_error_se_df(rand_score_error_df: pd.DataFrame) -> pd.DataFrame:
    df_rand_mdl_grouped = rand_score_error_df.groupby("rand_mdl_name")
    df_rand_mdl_list = []

    # Iterate over randomization designs and calculate standard errors of
    # the residuals in treatment effect estimates for each score bin
    for _, df_rand_mdl in df_rand_mdl_grouped:
        # If there's only one unique score, set the bin to be the score itself
        # and the standard error to be 0
        if df_rand_mdl["score"].nunique() == 1:
            df_rand_mdl["score_bin_mid"] = df_rand_mdl["score"].iloc[0]
            df_rand_mdl["se"] = 0
        else:
            # Otherwise, split the scores into 50 bins (or fewer if there are <50 unique scores)
            # and calculate the standard error of the residuals for each bin
            labels, bins = pd.qcut(
                df_rand_mdl["score"],
                q=50,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            bins_mid = (bins[1:] + bins[:-1]) / 2
            df_rand_mdl["score_bin_mid"] = bins_mid[labels]
            df_rand_mdl = df_rand_mdl.groupby(["rand_mdl_name", "score_bin_mid"]).agg(
                se=("error", "sem")
            )
            df_rand_mdl = df_rand_mdl.reset_index()
        df_rand_mdl_list.append(df_rand_mdl)

    df = pd.concat(df_rand_mdl_list)

    return df


def make_scatter_error_vs_score_fig(
    trial_set: Dict[str, SimulatedTrial],
    fitness_fn: Fitness,
    save_dir: Path,
    save_prefix: str,
    legend: bool = False,
) -> None:
    """
    Make scatterplot of residuals versus fitness scores, for each randomization design

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - fitness_fns: dictionary mapping data replicate to fitness function
        - save_dir: directory to save figure
    """
    # all_data_rep_df_list = []
    # all_data_rep_df_se_list = []

    # Get dataframe of fitness scores and residuals for each randomization design
    all_trial_df = _get_rand_score_error_df(trial_set, fitness_fn)
    all_trial_se_df = _get_rand_score_error_se_df(all_trial_df)

    # Make scatterplot of residuals versus fitness scores
    _plot_scatter_error_vs_score(all_trial_df, save_dir, save_prefix)

    # Make scatterplot of standard errors of residuals versus fitness scores
    _plot_scatter_error_se_vs_score_bucket(all_trial_se_df, save_dir, save_prefix)

    # all_data_rep_df_list.append(all_trial_df)
    # all_data_rep_df_se_list.append(all_trial_se_df)

    # Make scatterplots combining all data replicates
    # if len(trials) > 1:
    #     all_data_rep_df = pd.concat(all_data_rep_df_list, ignore_index=True)
    #     all_data_rep_df_se = pd.concat(all_data_rep_df_se_list, ignore_index=True)
    #     data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    #     _plot_scatter_error_vs_score(all_data_rep_df, save_dir, data_reps_str, legend)
    #     _plot_scatter_error_se_vs_score_bucket(
    #         all_data_rep_df_se, save_dir, data_reps_str
    #     )


def make_scatter_error_vs_score_mult_fig(
    trial_sets: List[Dict[str, SimulatedTrial]],
    fitness_fns: List[Fitness],
    save_dirs: List[Path],
    save_prefix: str,
    legend: bool = True,
):

    # Get dataframe of fitness scores and residuals for each randomization design
    all_trial_df_list = [
        _get_rand_score_error_df(trial_set, fitness_fn)
        for trial_set, fitness_fn in zip(trial_sets, fitness_fns)
    ]
    all_trial_df_se_list = [
        _get_rand_score_error_se_df(df) for df in all_trial_df_list
    ]

    max_abs_residual = np.max([df["error"].abs().max() for df in all_trial_df_list])
    ylim = (-1.1 * max_abs_residual, 1.1 * max_abs_residual)
    ymax_se = np.max([df["se"].max() for df in all_trial_df_se_list])

    for all_trial_df, all_trial_df_se, save_dir in zip(
        all_trial_df_list, all_trial_df_se_list, save_dirs
    ):
        _plot_scatter_error_vs_score(all_trial_df, save_dir, save_prefix, ylim, legend)
        _plot_scatter_error_se_vs_score_bucket(
            all_trial_df_se, save_dir, save_prefix, ymax_se, legend
        )
