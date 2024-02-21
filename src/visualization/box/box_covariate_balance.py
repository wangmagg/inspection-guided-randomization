from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from pathlib import Path

from src.sims.trial import SimulatedTrial
from typing import Dict

def _plot_covariate_balance_fig(
    data_rep_balance_df: pd.DataFrame, 
    balance_fn_name: str, 
    save_dir: Path, 
    save_prefix: str
):
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
        id_vars=["data_rep", "rand_mdl"],
        var_name="covariate",
        value_name=balance_fn_name,
    )

    # Make box plot
    hue_order = np.sort(data_rep_balance_df["rand_mdl"].unique())
    sns.boxplot(
        data=data_rep_balance_df_long,
        x="covariate",
        y=balance_fn_name,
        hue="rand_mdl",
        hue_order=hue_order,
        ax=ax,
    )

    legend = ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        title="Design",
        fontsize=18,
        title_fontsize=20,
        bbox_to_anchor=(0.5, -0.5),
        loc="lower center",
        ncol=len(hue_order) // 2
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Covariate", fontsize=20)
    ax.set_ylabel(f"{balance_fn_name}", fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Save figure
    save_fname = f"{save_prefix}_{balance_fn_name}_covariate_balance.png"
    save_path = save_dir / save_fname
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_covariate_balance_fig(data_rep_to_trials_dict: Dict[str, Dict[str, SimulatedTrial]],
                               balance_fn_name: str, 
                               fig_dir: Path):
    """
    Make boxplot of global covariate balance across all arms for each covariate,
    with a separate figure for each data replicate 

    Args:
        - data_rep_to_trials_dict: maps data replicate to trials under each randomization design
        - balance_fn_name: name of balance function
        - fig_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True
    all_balance_df_list = []

    # Iterate through data replicates 
    for data_rep, rand_mdl_to_trial_dict in data_rep_to_trials_dict.items():
        data_rep_balance_df_list = []
        for rand_mdl, trial in rand_mdl_to_trial_dict.items():
            trial.loader.config.fitness_fn_name = balance_fn_name
            balance_fn = trial.loader.get_fitness_fn(trial)
            balance_scores = balance_fn(trial.z_pool)
            balance_df = pd.DataFrame(balance_scores, columns=trial.X_fit.columns)
            balance_df["data_rep"] = data_rep
            balance_df["rand_mdl"] = rand_mdl
            data_rep_balance_df_list.append(balance_df)

        # Make boxplot
        data_rep_balance_df = pd.concat(data_rep_balance_df_list)
        _plot_covariate_balance_fig(
            data_rep_balance_df, balance_fn_name, fig_dir, data_rep
        )

        all_balance_df_list.extend(data_rep_balance_df_list)

    # Make boxplot combining all data replicates
    all_balance_df = pd.concat(all_balance_df_list)
    data_rep_str = "-".join(
        [str(data_rep) for data_rep in range(len(data_rep_to_trials_dict))]
    )
    _plot_covariate_balance_fig(all_balance_df, balance_fn_name, fig_dir, data_rep_str)

    return