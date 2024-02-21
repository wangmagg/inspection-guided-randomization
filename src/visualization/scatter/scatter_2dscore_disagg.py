from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

from src.sims.trial import *
from typing import List, Dict

def _plot_scatter_2dscore_disagg(
    df: pd.DataFrame, 
    xaxis_fn_name: str, 
    yaxis_fn_name: str, 
    save_dir: Path, 
    save_prefix: str
):
    """
    Helper function to make scatterplot of fitness scores corresponding to two 
    (possibly competing) design desiderata, for each randomization design

    Args:
        - df: dataframe containing scores for each randomization design
        - xaxis_fn_name: name of x-axis fitness function
        - yaxis_fn_name: name of y-axis fitness function
        - save_dir: directory to save figure
        - save_prefix: prefix for figure name
    """
    hue_order = np.sort(df["rand_mdl_name"].unique())
    jnt_grid = sns.JointGrid(
        data=df, 
        x=xaxis_fn_name, 
        y=yaxis_fn_name, 
        hue="rand_mdl_name", 
        height=5,
        hue_order=hue_order
    )

    # Make scatterplot with marginal histograms
    jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.5)
    jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
    jnt_grid.set_axis_labels(xaxis_fn_name, yaxis_fn_name, fontsize=20)
    jnt_grid.ax_joint.legend(
        title="Design",
        fontsize=20,
        title_fontsize=20,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.5),
    )

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Save figure
    save_fname = f"{save_prefix}_scatter2d_score_disagg.png"
    save_path = save_dir / save_fname
    print(save_path)
    jnt_grid.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

    return jnt_grid

def _get_rand_xval_yval_df(trial_dict: Dict[str, SimulatedTrial],
                           xaxis_fn_name: str, 
                           yaxis_fn_name: str) -> pd.DataFrame:
    """
    Helper function to get dataframe of fitness scores corresponding to two
    (possibly competing) design desiderata, for each randomization design

    Args:
        - trial_dict: dictionary of randomization designs
        - xaxis_fn_name: name of x-axis fitness function
        - yaxis_fn_name: name of y-axis fitness function
    Returns
        dataframe containing scores for each randomization design
    """
    x_axis_fns = []
    y_axis_fns = []
    # Iterate over randomization designs
    for trial in trial_dict.values():
        # Calculate x-axis and y-axis fitness scores
        trial.config.fitness_fn_name = xaxis_fn_name
        x_axis_fn = trial.loader.get_fitness_fn(trial)
        x_axis_fns.append(x_axis_fn)

        trial.config.fitness_fn_name = yaxis_fn_name
        y_axis_fn = trial.loader.get_fitness_fn(trial)
        y_axis_fns.append(y_axis_fn)

    # Combine x- and y-axis scores across randomization designs
    rand_mdl_names = list(trial_dict.keys())
    z_pool_sizes = [trial.z_pool.shape[0] for trial in trial_dict.values()]
    all_rand_mdl_names = np.repeat(rand_mdl_names, z_pool_sizes)

    all_z_pool = [trial.z_pool for trial in trial_dict.values()]
    xvals = np.hstack(
        [x_axis_fn(z_pool) for x_axis_fn, z_pool in zip(x_axis_fns, all_z_pool)]
    )
    yvals = np.hstack(
        [y_axis_fn(z_pool) for y_axis_fn, z_pool in zip(y_axis_fns, all_z_pool)]
    )

    all_trial_df = pd.DataFrame.from_dict(
        {
            "rand_mdl_name": all_rand_mdl_names,
            xaxis_fn_name: xvals,
            yaxis_fn_name: yvals,
        }
    )

    return all_trial_df

def make_scatter_2dscore_disagg_fig(
    trials: Dict[str, Dict[str, SimulatedTrial]],
    axis_fn_names: List[str],
    save_dir: Path,
):
    """
    Make scatterplot of fitness scores corresponding to two (possibly competing) design desiderata

    Each point in the scatterplot corresponds to a treatment allocation
    Allocations accepted by a restricted randomization design are overlaid on the same plot as 
    allocations from a benchmark randomization design (e.g. complete randomization)

    Args:
        - trials: dictionary mapping data replicate to trials under each randomization design
        - axis_fn_names: list of names of x-axis and y-axis fitness functions
        - save_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True
    all_data_rep_df_list = []

    xaxis_fn_name, yaxis_fn_name = axis_fn_names

    # Iterate over data replicates
    for data_rep, trial_dict in trials.items():
        all_trial_df = _get_rand_xval_yval_df(trial_dict, xaxis_fn_name, yaxis_fn_name)

        # Make scatterplot
        _plot_scatter_2dscore_disagg(
            all_trial_df, xaxis_fn_name, yaxis_fn_name, save_dir, data_rep
        )
        all_data_rep_df_list.append(all_trial_df)

    # Make scatterplot combining all data replicates
    all_data_rep_df = pd.concat(all_data_rep_df_list, ignore_index=True)
    data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    _plot_scatter_2dscore_disagg(
        all_data_rep_df, xaxis_fn_name, yaxis_fn_name, save_dir, data_reps_str
    )