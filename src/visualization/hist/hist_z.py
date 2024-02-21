from matplotlib import pyplot as plt
import seaborn as sns

from src.sims.trial import *
from typing import Dict

def _plot_hist_z(df: pd.DataFrame, 
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

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Make histogram
    n_bins = len(df["rand_mdl_name"].unique()) * 20
    sns.histplot(
        df,
        x="frac_z_1",
        hue="rand_mdl_name",
        element="step",
        bins=n_bins,
        alpha=0.3,
        ax=ax,
    )
    ax.set_xlabel(r"$\hat{P}(z_i = 1)$", fontsize=14)
    ax.set_ylabel(r"Allocations", fontsize=14)

    legend = ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        title="Design",
        fontsize=12,
        title_fontsize=12,
        bbox_to_anchor=(0.5, -0.5),
        loc="lower center",
    )

    # Save figure
    save_fname = f"{save_prefix}_z.png"
    save_path = save_dir / save_fname
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

    return ax

def _get_frac_z_1_df(trial_dict: Dict[str, SimulatedTrial]) -> pd.DataFrame:
    """
    Helper function to get dataframe of fraction of units assigned to treatment arm for each randomization design

    Args:
        - trial_dict: dictionary of trial objects for each randomization design
    Returns:
        dataframe containing fraction of units assigned to treatment arm for each randomization design
    """
    all_trial_frac_z_1 = []

    # Iterate over randomization designs
    for trial_type, trial in trial_dict.items():
        # Get fraction of units assigned to treatment arm across 
        # all treatment allocations in pool
        z_pool = trial.z_pool
        frac_z_1 = np.mean(z_pool, axis=1)
        trial_frac_z_1_dict = {
            "rand_mdl_name": trial_type,
            "frac_z_1": frac_z_1,
        }
        trial_frac_z_1_df = pd.DataFrame.from_dict(trial_frac_z_1_dict)

        all_trial_frac_z_1.append(trial_frac_z_1_df)
    all_trial_frac_z_1_df = pd.concat(all_trial_frac_z_1).reset_index(drop=True)

    return all_trial_frac_z_1_df

def make_hist_z_fig(
    trials: Dict[str, Dict[str, SimulatedTrial]], 
    save_dir: Path
):
    """
    Make histogram of fraction of units assigned to treated arm (assuming binary treatment), 
    for each randomization design

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - save_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True
    all_data_rep_df_list = []

    # Iterate over data replicates
    for data_rep, trial_dict in trials.items():

        # Get fraction of units assigned to treatment arm for each randomization design
        all_trial_frac_z_1_df = _get_frac_z_1_df(trial_dict)

        # Make histogram
        _plot_hist_z(all_trial_frac_z_1_df, save_dir, data_rep)

        all_data_rep_df_list.append(all_trial_frac_z_1_df)

    # Make histogram combining all data replicates
    all_data_rep_df = pd.concat(all_data_rep_df_list, ignore_index=True)
    data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    _plot_hist_z(all_data_rep_df, save_dir, data_reps_str)