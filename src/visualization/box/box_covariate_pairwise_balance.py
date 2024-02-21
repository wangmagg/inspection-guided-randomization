from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from src.sims.trial import SimulatedTrial
from typing import Dict

def _plot_covariate_pairwise_balance(
        df: pd.DataFrame, 
        balance_fn_name: str, 
        save_dir: Path, 
        save_prefix: str):
    """
    Helper function to make boxplots of covariate balance across all pairwise arm comparisons
    for each covariate and randomization design, with separate subplot for each covariate

    Args:
        - df: balance scores across randomization designs
        - balance_fn_name: name of balance function
        - save_dir: directory to save figure
    """
    
    cov_names = df.columns[~df.columns.isin(['group_i', 'group_j', 'rand_mdl'])]
    n_cov = len(cov_names)
    fig, axs = plt.subplots(1, n_cov, figsize=(5*n_cov, 5), sharey=True, sharex=True)

    # Combine group_i and group_j comparison into single column 
    df['arm_comparison'] = df['group_i'].astype(str) + ' vs ' + df['group_j'].astype(str)
    df = df.drop(columns=['group_i', 'group_j'])
    hue_order = np.sort(df["rand_mdl"].unique())

    # Make subplot for each covariate 
    for ax, cov in zip(axs, cov_names):
        df_cov = df[['arm_comparison', 'rand_mdl', cov]]
        sns.boxplot(data=df_cov, 
                    y='arm_comparison', 
                    x=cov, 
                    hue='rand_mdl', 
                    hue_order=hue_order,
                    fliersize=1,
                    ax=ax)
        ax.set_title(f"{cov}", fontsize=20)
        ax.axvline(0, color='black', linestyle='--')
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"{balance_fn_name}", fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.set_ylabel(f"Group Comparison", fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
        ax.get_legend().set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, 
               labels, 
               title="Design",
               fontsize=18,
               title_fontsize=20,
               loc = 'lower center',
               bbox_to_anchor=(0.5, -0.4),
               ncol=len(labels)//2)

    # Save figure
    save_fname = f"{save_prefix}_{balance_fn_name}_covariate_pairwise_balance.png"
    save_path = save_dir / save_fname

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    print(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_covariate_pairwise_balance_fig(
    data_rep_to_trials_dict: Dict[str, Dict[str, SimulatedTrial]], 
    balance_fn_name: str, 
    fig_dir: Path
):
    """
    Make boxplots of covariate balance across all pairwise arm comparisons
    for each covariate and randomization design, with separate subplots for each covariate
    and separate figure for each data replicate

    Args:
        - data_rep_to_trials_dict: maps data replicate to trials under each randomization design
        - balance_fn_name: name of balance function
        - fig_dir: directory to save figure
    """
    
    plt.rcParams["text.usetex"] = True
    all_balance_df_list = []

    # Iterate through data replicates and randomization designs
    for data_rep, rand_mdl_to_trial_dict in data_rep_to_trials_dict.items():
        balance_scores_df_list = []
        for rand_mdl, trial in rand_mdl_to_trial_dict.items():

            # Calculate balance scores
            trial.use_cols = None
            trial.loader.config.fitness_fn_name = balance_fn_name
            balance_fn = trial.loader.get_fitness_fn(trial)
            balance_scores = balance_fn(trial.z_pool)

            # Store balance scores for pairwise arm comparisons
            n_groups = balance_scores.shape[1]
            for group_i in range(n_groups):
                for group_j in range(group_i + 1, n_groups):
                    balance_scores_df = pd.DataFrame(
                        balance_scores[:, group_i, group_j, :], columns=trial.X_fit.columns
                    )

                    balance_scores_df["rand_mdl"] = rand_mdl
                    balance_scores_df["group_i"] = group_i
                    balance_scores_df["group_j"] = group_j
                    balance_scores_df_list.append(balance_scores_df)
        
        # Make boxplot
        balance_scores_df = pd.concat(balance_scores_df_list)
        _plot_covariate_pairwise_balance(
            balance_scores_df, balance_fn_name, fig_dir, data_rep
        )
        all_balance_df_list.extend(balance_scores_df_list)

    # Make boxplot combining all data replicates
    all_balance_df = pd.concat(all_balance_df_list)
    data_rep_str = "-".join(
        [str(data_rep) for data_rep in range(len(data_rep_to_trials_dict))]
    )
    _plot_covariate_pairwise_balance(all_balance_df, balance_fn_name, fig_dir, data_rep_str)