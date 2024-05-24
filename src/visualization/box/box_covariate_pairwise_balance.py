from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict

from src.sims import trial_loader
from src.sims.trial import SimulatedTrial
from src.visualization.utils.aes import get_hue_order, get_palette

def _plot_covariate_pairwise_balance(
    df: pd.DataFrame,
    balance_fn_name: str,
    save_dir: Path,
    save_prefix: str,
    legend: bool = True,
) -> None:
    """
    Helper function to make boxplots of covariate balance across all pairwise arm comparisons
    for each covariate and randomization design, with separate subplot for each covariate

    Args:
        - df: balance scores across randomization designs
        - balance_fn_name: name of balance function
        - save_dir: directory to save figure
    """
    cov_names = df.columns[~df.columns.isin(["group_i", "group_j", "rand_mdl"])]
    n_cov = len(cov_names)
    fig, axs = plt.subplots(1, n_cov, figsize=(5 * n_cov, 5), sharey=True, sharex=True)

    # Combine group_i and group_j comparison into single column
    df["arm_comparison"] = (
        df["group_i"].astype(str) + " vs " + df["group_j"].astype(str)
    )
    df = df.drop(columns=["group_i", "group_j"])
    hue_order = get_hue_order(df["rand_mdl"].unique())
    palette = get_palette(df["rand_mdl"].unique())

    # Make subplot for each covariate
    for ax, cov in zip(axs, cov_names):
        df_cov = df[["arm_comparison", "rand_mdl", cov]]
        sns.boxplot(
            data=df_cov,
            y="arm_comparison",
            x=cov,
            hue="rand_mdl",
            hue_order=hue_order,
            palette=palette,
            fliersize=1,
            ax=ax,
        )
        ax.set_title(f"{cov}".capitalize(), fontsize=24)
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"{balance_fn_name}", fontsize=22)
        ax.tick_params(axis="x", labelsize=18)
        ax.set_ylabel(f"Group Comparison", fontsize=22)
        ax.tick_params(axis="y", labelsize=18)
        ax.get_legend().set_visible(False)

    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        ncol = 1
        loc = "center right"
        bbox_to_anchor = (1.2, 0.5)
        fig.legend(
            handles,
            labels,
            title=None,
            fontsize=18,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            edgecolor="black",
        )

    # Save figure
    save_fname = f"{save_prefix}_{balance_fn_name}_covariate_pairwise_balance.svg"
    save_path = save_dir / save_fname

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    print(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


def make_covariate_pairwise_balance_fig(
    trial_set: Dict[str, SimulatedTrial],
    balance_fn_name: str,
    fig_dir: Path,
    save_prefix: str,
    legend: bool = True,
) -> None:
    """
    Make boxplots of covariate balance across all pairwise arm comparisons
    for each covariate and randomization design, with separate subplots for each covariate
    and separate figure for each data replicate

    Args:
        - data_rep_to_trials_dict: maps data replicate to trials under each randomization design
        - balance_fn_name: name of balance function
        - fig_dir: directory to save figure
    """

    balance_scores_df_list = []
    for rand_mdl, trial in trial_set.items():

        # Calculate balance scores
        trial.config.fitness_fn_name = balance_fn_name
        trial.use_cols = None
        balance_fn = trial_loader.get_fitness_fn(trial)
        balance_scores = balance_fn(trial.z_pool)

        # Store balance scores for pairwise arm comparisons
        for i, pair in enumerate(trial.arm_compare_pairs):
            balance_scores_df = pd.DataFrame(
                balance_scores[:, i, :], columns=trial.X_fit.columns
            )

            balance_scores_df["rand_mdl"] = rand_mdl
            balance_scores_df["group_i"] = pair[0]
            balance_scores_df["group_j"] = pair[1]
            balance_scores_df_list.append(balance_scores_df)

    # Make boxplot
    balance_scores_df = pd.concat(balance_scores_df_list)
    _plot_covariate_pairwise_balance(
        balance_scores_df, balance_fn.plotting_name, fig_dir, save_prefix, legend
    )
