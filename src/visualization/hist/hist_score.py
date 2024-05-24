from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from src.design.fitness_functions import LinearCombinationFitness
from src.sims.trial import SimulatedTrial
from src.visualization.utils.aes import get_palette, get_two_tone_palette


def _rand_mdl_name_to_enum_type(rand_mdl_name: str):
    if "IGRg" in rand_mdl_name:
        return r"$\mathcal{Z}^{*}_{pool}$"
    else:
        return r"$\mathcal{Z}_{pool}$"


def _subplot_hist_score(
    score_df_ff: pd.DataFrame,
    score_cutoff_df_ff: pd.DataFrame,
    ax: plt.Axes,
    ax_title: str,
    hist_palette: Dict[str, str],
    vline_palette: Dict[str, str],
):

    # Make histogram
    sns.histplot(
        score_df_ff,
        x="score",
        hue="enum_type",
        palette=hist_palette,
        bins=80,
        element="step",
        fill=False,
        kde=True,
        line_kws={"linewidth": 1.5},
        stat="frequency",
        common_norm=False,
        ax=ax,
        log_scale=True,
    )
    for _, rand_mdl_row in score_cutoff_df_ff.iterrows():
        ax.axvline(
            rand_mdl_row["cutoff"],
            color=vline_palette[rand_mdl_row["rand_mdl_name"]],
            linewidth=1.5,
        )
        ax.text(
            x=rand_mdl_row["cutoff"] * rand_mdl_row["cutoff_pad"],
            y=0.99,
            s=rand_mdl_row["annot_txt"],
            color=vline_palette[rand_mdl_row["rand_mdl_name"]],
            ha=rand_mdl_row["annot_side"],
            va="top",
            rotation=90,
            transform=ax.get_xaxis_transform(),
            fontsize=18,
        )

    ax.set_title(ax_title, fontsize=22, pad=20)
    ax.set_xlabel(r"$f(\mathbf{z})$", fontsize=20, loc="right")
    ax.set_ylabel("Number of Allocations", fontsize=20)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="x", which="minor", rotation=30, labelsize=14)
    ax.tick_params(axis="y", labelsize=16, labelleft=True)
    ax.yaxis.get_offset_text().set_fontsize(14)

    legend = ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=18,
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
    )


def _plot_hist_score(
    score_df: pd.DataFrame,
    score_cutoff_df: pd.DataFrame,
    save_dir: Path,
    save_prefix: str,
    save_suffix: str = None,
):
    score_df["enum_type"] = score_df["rand_mdl_name"].apply(
        lambda x: _rand_mdl_name_to_enum_type(x)
    )
    score_cutoff_df["annot_txt"] = (
        score_cutoff_df["rand_mdl_name"].apply(lambda x: x.split("-")[0].strip())
        + r" $s*$"
    )
    if score_cutoff_df.shape[0] > 1:
        score_cutoff_df["annot_side"] = np.tile(
            ["right", "left"], score_cutoff_df.shape[0] // 2
        )
        score_cutoff_df["cutoff_pad"] = np.tile([0.96, 1.04], score_cutoff_df.shape[0] // 2)
    else:
        score_cutoff_df["annot_side"] = ["right"]
        score_cutoff_df["cutoff_pad"] = [0.96]

    ff_names = score_df["ff_name"].unique()
    fig, axs = plt.subplots(
        ncols=len(ff_names), figsize=(6 * len(ff_names), 5), sharey=True
    )
    plt.subplots_adjust(wspace=0.4)
    hist_palette = get_two_tone_palette(score_df["enum_type"].unique())
    vline_palette = get_palette(score_cutoff_df["rand_mdl_name"].unique())

    if len(ff_names) == 1:
        _subplot_hist_score(
            score_df, score_cutoff_df, axs, ff_names[0], hist_palette, vline_palette
        )
    else:
        for ax, ff_name in zip(axs, ff_names):
            score_df_ff = score_df[score_df["ff_name"] == ff_name]
            score_cutoff_df_ff = score_cutoff_df[score_cutoff_df["ff_name"] == ff_name]
            _subplot_hist_score(
                score_df_ff,
                score_cutoff_df_ff,
                ax,
                ff_name,
                hist_palette,
                vline_palette,
            )

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Save figure
    if save_suffix is not None:
        save_fname = f"{save_prefix}_{save_suffix}_hist_score.svg"
    else:
        save_fname = f"{save_prefix}_hist_score.svg"
    save_path = save_dir / save_fname
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()


def _get_score_df(trial_dict: Dict[str, SimulatedTrial]):
    """
    Helper function to get scores for each IGR randomization design

    Args:
        - trial_dict: dictionary mapping randomization design to trial

    Returns:
        - df: dataframe containing scores for each randomization design
    """

    all_trial_scores = []
    all_trial_score_dfs = []

    for rand_mdl_name, trial in trial_dict.items():
        if "IGR" not in rand_mdl_name:
            continue

        if isinstance(trial.rand_mdl.fitness_fn, LinearCombinationFitness):
            score_df = pd.DataFrame(
                {
                    "score_fn_0": trial.fn_scores[0],
                    "score_fn_1": trial.fn_scores[1],
                    "rand_mdl_name": rand_mdl_name,
                    "ff_name": trial.rand_mdl.fitness_fn.plotting_name,
                    "enum_type": _rand_mdl_name_to_enum_type(rand_mdl_name),
                    "cutoff_idx": np.argsort(trial.scores)[trial.config.n_cutoff],
                }
            )
        else:
            score_df = pd.DataFrame(
                {
                    "score": trial.scores,
                    "rand_mdl_name": rand_mdl_name,
                    "ff_name": trial.rand_mdl.fitness_fn.plotting_name,
                    "enum_type": _rand_mdl_name_to_enum_type(rand_mdl_name),
                    "cutoff_idx": np.argsort(trial.scores)[trial.config.n_cutoff],
                }
            )
        all_trial_score_dfs.append(score_df)

    all_trial_score_df = pd.concat(all_trial_score_dfs)

    if "score_fn_0" in all_trial_score_df.columns:
        all_trial_scores_fn_0 = all_trial_score_df["score_fn_0"]
        all_trial_scores_fn_1 = all_trial_score_df["score_fn_1"]

        if np.max(all_trial_scores_fn_0) - np.min(all_trial_scores_fn_0) > 0:
            all_trial_scores_fn_0 = (
                all_trial_scores_fn_0 - np.min(all_trial_scores_fn_0)
            ) / (np.max(all_trial_scores_fn_0) - np.min(all_trial_scores_fn_0))
        else:
            all_trial_scores_fn_0 = np.zeros_like(all_trial_scores_fn_0)
        if np.max(all_trial_scores_fn_1) - np.min(all_trial_scores_fn_1) > 0:
            all_trial_scores_fn_1 = (
                all_trial_scores_fn_1 - np.min(all_trial_scores_fn_1)
            ) / (np.max(all_trial_scores_fn_1) - np.min(all_trial_scores_fn_1))
        else:
            all_trial_scores_fn_1 = np.zeros_like(all_trial_scores_fn_1)

        all_trial_scores = (
            trial.config.fitness_fn_weights[0] * all_trial_scores_fn_0
            + trial.config.fitness_fn_weights[1] * all_trial_scores_fn_1
        )
        all_trial_score_df["score"] = all_trial_scores

    all_trial_score_cutoff_df = (
        all_trial_score_df[["cutoff_idx", "score", "rand_mdl_name", "ff_name"]]
        .groupby(["rand_mdl_name", "ff_name"])
        .apply(lambda x: x["score"][x["cutoff_idx"].iloc[0]])
        .reset_index(name="cutoff")
        .sort_values(by="cutoff", ascending=True)
    )

    return all_trial_score_df, all_trial_score_cutoff_df


def make_hist_score_fig(trial_set: Dict[str, SimulatedTrial], save_dir: Path, save_prefix: str):
    """
    Make histogram of scores for each randomization design

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - save_dir: directory to save figure
    """
    score_df, score_cutoff_df = _get_score_df(trial_set)
    _plot_hist_score(score_df, score_cutoff_df, save_dir, save_prefix)


def make_hist_score_mult_fig(
    trial_sets: List[Dict[str, SimulatedTrial]],
    save_dir: Path,
    save_prefix: str,
    save_suffix: str = None,
):

    score_df_list = []
    score_cutoff_df_list = []
    for trial_set in trial_sets:
        score_df, score_cutoff_df = _get_score_df(trial_set)
        score_df_list.append(score_df)
        score_cutoff_df_list.append(score_cutoff_df)

    score_df = pd.concat(score_df_list)
    score_cutoff_df = pd.concat(score_cutoff_df_list)

    _plot_hist_score(score_df, score_cutoff_df, save_dir, save_prefix, save_suffix)
