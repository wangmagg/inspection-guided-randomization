from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict

from src.sims import trial_loader
from src.sims.trial import SimulatedTrial
from src.visualization.utils.aes import get_hue_order, get_palette


def _plot_covariate_balance_fig(
    data_rep_balance_df: pd.DataFrame, 
    balance_fn_name: str, 
    save_dir: Path, 
    save_prefix: str,
    legend: bool = True
) -> None:
    """
    Helper function to make boxplot of global covariate balance across all arms
    for each covariate and randomization design

    Args:
    - data_rep_balance_df: balance scores across randomization designs
    - balance_fn_name: name of balance function
    - save_dir: directory to save figure
    - save_prefix: prefix for figure name
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Melt data into long form
    data_rep_balance_df_long = data_rep_balance_df.melt(
        id_vars=["rand_mdl"],
        var_name="covariate",
        value_name=balance_fn_name,
    )

    # Capitalize covariate names
    data_rep_balance_df_long["covariate"] = data_rep_balance_df_long["covariate"].str.capitalize()

    # Make box plot
    hue_order = get_hue_order(data_rep_balance_df_long["rand_mdl"].unique())
    palette = get_palette(data_rep_balance_df_long["rand_mdl"].unique())
    sns.boxplot(
        data=data_rep_balance_df_long,
        x="covariate",
        y=balance_fn_name,
        hue="rand_mdl",
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )
    if legend:
        legend = ax.legend_
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
        ax.legend(
            handles=handles,
            labels=labels,
            title=None,
            fontsize=18,
            bbox_to_anchor=(0.5, -0.5),
            loc="lower center",
            ncol=len(hue_order) // 2,
            edgecolor='black'
        )
    else:
        ax.get_legend().remove()
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Covariate", fontsize=22)
    ax.set_ylabel(f"{balance_fn_name}", fontsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    # Save figure
    save_fname = f"{save_prefix}_{balance_fn_name}_covariate_balance.svg"
    save_path = save_dir / save_fname
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


def make_covariate_balance_fig(trial_set: Dict[str, SimulatedTrial],
                               balance_fn_name: str, 
                               fig_dir: Path,
                               save_prefix: str,
                               legend: bool = True) -> None:
    """
    Make boxplot of global covariate balance across all arms for each covariate,
    with a separate figure for each data replicate 

    Args:
        - data_rep_to_trials_dict: maps data replicate to trials under each randomization design
        - balance_fn_name: name of balance function
        - fig_dir: directory to save figure
    """

    data_rep_balance_df_list = []
    for rand_mdl, trial in trial_set.items():
        trial.config.fitness_fn_name = balance_fn_name
        balance_fn = trial_loader.get_fitness_fn(trial)
        balance_scores = balance_fn(trial.z_pool)
        balance_df = pd.DataFrame(balance_scores, columns=trial.X_fit.columns)
        balance_df["rand_mdl"] = rand_mdl
        data_rep_balance_df_list.append(balance_df)

    # Make boxplot
    data_rep_balance_df = pd.concat(data_rep_balance_df_list)
    _plot_covariate_balance_fig(
        data_rep_balance_df, balance_fn.plotting_name, fig_dir, save_prefix, legend
    )

    return