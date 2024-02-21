from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

from src.sims.trial import *
from typing import Dict

def _plot_scatter_error_vs_score(df: pd.DataFrame, 
                                 save_dir: Path, 
                                 save_prefix: str):
    """
    Helper function to make scatterplot of residuals versus fitness scores, for each randomization design

    Args:
        - df: dataframe containing scores and residuals for each randomization design
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """
    hue_order = np.sort(df["rand_mdl_name"].unique())
    jnt_grid = sns.JointGrid(
        data=df, x="score", y="error", hue="rand_mdl_name", height=5, hue_order=hue_order
    )
    # Make scatterplot with marginal histograms
    jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.5)
    jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
    jnt_grid.ax_joint.axhline(y=0, color="black", linestyle="--")
    jnt_grid.set_axis_labels("Score", "Residual", fontsize=20)
    jnt_grid.ax_joint.legend(
        title="Design",
        fontsize=20,
        title_fontsize=20,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.5),
    )

    # Save figure
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_fname = f"{save_prefix}_scatter_error_vs_score.png"
    save_path = save_dir / save_fname
    print(save_path)
    jnt_grid.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

def _plot_scatter_error_se_vs_score_bucket(df: pd.DataFrame, 
                                           save_dir: Path, 
                                           save_prefix: str):
    """
    Helper function to make scatterplot of standard errors of residuals versus fitness scores, 
    for each randomization design

    Args:
        - df: dataframe containing scores and residuals for each randomization design
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """
    df_rand_mdl_grouped = df.groupby("rand_mdl_name")
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
            labels, bins = pd.qcut(df_rand_mdl["score"], q=50, labels=False, retbins=True, duplicates="drop")
            bins_mid = (bins[1:] + bins[:-1]) / 2
            df_rand_mdl["score_bin_mid"] = bins_mid[labels] 
            df_rand_mdl = df_rand_mdl.groupby(["rand_mdl_name", "score_bin_mid"]).agg(se = ("error", "sem"))
            df_rand_mdl = df_rand_mdl.reset_index()
        df_rand_mdl_list.append(df_rand_mdl)

    df = pd.concat(df_rand_mdl_list)

    hue_order = np.sort(df["rand_mdl_name"].unique())
    jnt_grid = sns.JointGrid(
        data=df, x="score_bin_mid", y="se", hue="rand_mdl_name", height=5, hue_order=hue_order
    )
    # Make scatterplot with marginal histograms
    jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.8)
    jnt_grid.set_axis_labels("Score", "SE of the Residual", fontsize=20)
    jnt_grid.ax_joint.legend(
        title="Design",
        fontsize=18,
        title_fontsize=20,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.8),
    )
    
    # Save figure
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_fname = f"{save_prefix}_scatter_se_vs_score.png"
    save_path = save_dir / save_fname
    print(save_path)
    jnt_grid.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def _get_rand_score_error_df(trial_dict: Dict[str, SimulatedTrial],
                             fitness_fn: Fitness) -> pd.DataFrame:
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
        if "cluster-settlement" in rand_mdl_name:
            mapping = (
                trial.X[["settlement_id", "school_id"]]
                .groupby("school_id")
                .agg(max)["settlement_id"]
            )
            all_z_pool.append(np.vstack([z[mapping] for z in trial.z_pool]))
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
        {"rand_mdl_name": all_rand_mdl_names, 
         "score": all_scores,
         "error": all_errors}
    )

    return all_trial_df


def make_scatter_error_vs_score_fig(
    trials: Dict[str, Dict[str, SimulatedTrial]],
    fitness_fns: Dict[str, Fitness],
    save_dir: Path,
):
    """
    Make scatterplot of residuals versus fitness scores, for each randomization design

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - fitness_fns: dictionary mapping data replicate to fitness function
        - save_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True
    all_data_rep_df_list = []

    # Iterate over data replicates
    for data_rep, trial_dict in trials.items():
        # Get dataframe of fitness scores and residuals for each randomization design
        all_trial_df = _get_rand_score_error_df(trial_dict, fitness_fns[data_rep])

        # Make scatterplot of residuals versus fitness scores
        _plot_scatter_error_vs_score(all_trial_df, save_dir, f"{data_rep}")

        # Make scatterplot of standard errors of residuals versus fitness scores
        _plot_scatter_error_se_vs_score_bucket(all_trial_df, save_dir, f"{data_rep}")

        all_data_rep_df_list.append(all_trial_df)

    # Make scatterplots combining all data replicates
    all_data_rep_df = pd.concat(all_data_rep_df_list, ignore_index=True)
    data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    _plot_scatter_error_vs_score(all_data_rep_df, save_dir, data_reps_str)
    _plot_scatter_error_se_vs_score_bucket(all_data_rep_df, save_dir, data_reps_str)
