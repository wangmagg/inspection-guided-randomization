from matplotlib import pyplot as plt
import seaborn as sns

from src.sims.trial import *
from typing import Dict

def _plot_hist_same_z_validity(df: pd.DataFrame, 
                               save_dir: Path, 
                               save_prefix: str):
    """
    Helper function to make histogram of how frequently two units are assigned 
    to the same treatment arm, for each randomization designs

    Args:
        - df: dataframe containing "same-z" frequencies for each randomization design
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Make histogram
    n_bins = len(df["rand_mdl_name"].unique()) * 20
    sns.histplot(
        df,
        x="same_z",
        hue="rand_mdl_name",
        element="step",
        bins=n_bins,
        alpha=0.3,
        kde=True,
        line_kws={"linewidth": 1},
        stat="probability",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlabel(r"$\hat{P}(z_i = z_j)$", fontsize=14)
    ax.set_ylabel(r"Fraction of Pairs $(i, j)$", fontsize=14)

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
    save_fname = f"{save_prefix}_same_z_validity.png"
    save_path = save_dir / save_fname
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
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
        same_z_mat = np.matmul(z_pool.T, z_pool) + np.matmul(1 - z_pool.T, 1 - z_pool)
        same_z_upper = same_z_mat[np.triu_indices(same_z_mat.shape[0], k=1)]
        same_z_upper_flat_norm = same_z_upper.flatten() / trial.z_pool.shape[0]
        trial_same_z_dict = {
            "rand_mdl_name": trial_type,
            "same_z": same_z_upper_flat_norm,
        }
        trial_same_z_df = pd.DataFrame.from_dict(trial_same_z_dict)

        all_trial_same_z.append(trial_same_z_df)
    all_trial_same_z_df = pd.concat(all_trial_same_z).reset_index(drop=True)

    return all_trial_same_z_df


def make_hist_same_z_validity_fig(
    trials: Dict[str, Dict[str, SimulatedTrial]], 
    save_dir: Path
):
    """
    Make histogram of how frequently two units are assigned to the same treatment arm,
    for each randomization design

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - save_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True
    all_data_rep_df_list = []

    # Iterate through data replicates
    for data_rep, trial_dict in trials.items():
        all_trial_same_z_df = _get_same_z_df(trial_dict)

        # Make histogram
        _plot_hist_same_z_validity(all_trial_same_z_df, save_dir, data_rep)

        all_data_rep_df_list.append(all_trial_same_z_df)

    # Make histogram combining all data replicates
    all_data_rep_df = pd.concat(all_data_rep_df_list, ignore_index=True)
    data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    _plot_hist_same_z_validity(all_data_rep_df, save_dir, data_reps_str)