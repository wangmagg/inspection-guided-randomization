from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from pathlib import Path
from typing import List, Dict, Tuple

from src.sims import trial_loader
from src.sims.trial import SimulatedTrial
from src.visualization.utils.aes import get_hue_order, get_palette, axis_lim, format_ax

def _plot_scatter_2dscore_disagg(
    df: pd.DataFrame,
    xaxis_fn_name: str,
    yaxis_fn_name: str,
    save_dir: Path,
    save_prefix: str,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    legend: bool = True
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
    df = df.copy()
    uniq_rand_mdl_names = df["rand_mdl_name"].unique()
    hue_order = get_hue_order(uniq_rand_mdl_names, ff_in_name=False)
    palette = get_palette(uniq_rand_mdl_names, ff_in_name=False)

    df["rand_mdl_name"] = df["rand_mdl_name"].apply(lambda x: x.split("-")[0].strip())
    igr_in_name_mask = ["IGR" in name for name in uniq_rand_mdl_names]
    ff_name = uniq_rand_mdl_names[igr_in_name_mask][0].split("-")[1].strip()

    jnt_grid = sns.JointGrid(
        data=df,
        x=xaxis_fn_name,
        y=yaxis_fn_name,
        hue="rand_mdl_name",
        height=5,
        palette=palette,
        hue_order=hue_order,
    )

    # Make scatterplot with marginal histograms
    jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.5)
    jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
    jnt_grid.set_axis_labels(xaxis_fn_name, yaxis_fn_name, fontsize=18)

    jnt_grid.fig.suptitle(f"{ff_name}", fontsize=22)
    jnt_grid.fig.subplots_adjust(top=0.85)

    if xlim is None:
        xmax = df[xaxis_fn_name].max()
        xmin = df[xaxis_fn_name].min()
        xrange = df[xaxis_fn_name].max() - df[xaxis_fn_name].min()
        xmax = 0.01 * xrange + xmax
        xmin = -0.01 * xrange + xmin
    if ylim is None:
        ymax = df[yaxis_fn_name].max()
        ymin = df[yaxis_fn_name].min()
        yrange = df[yaxis_fn_name].max() - df[yaxis_fn_name].min()
        ymax = 0.01 * yrange + ymax
        ymin = -0.01 * yrange + ymin

    jnt_grid.ax_joint.set_xlim(xlim)
    jnt_grid.ax_joint.set_ylim(ylim)

    jnt_grid.ax_joint = format_ax(jnt_grid.ax_joint)

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

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Save figure
    save_fname = (
        f"{save_prefix}_scatter2d_score_disagg_x-{xaxis_fn_name}_y-{yaxis_fn_name}.svg"
    )
    save_path = save_dir / save_fname
    print(save_path)
    jnt_grid.savefig(save_path, bbox_inches="tight", dpi=200, transparent=True)
    plt.close()

    return jnt_grid


def _get_rand_xval_yval_df(
    trial_dict: Dict[str, SimulatedTrial], xaxis_fn_name: str, yaxis_fn_name: str
) -> pd.DataFrame:
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
        x_axis_fn = trial_loader.get_fitness_fn(trial)
        x_axis_fns.append(x_axis_fn)

        trial.config.fitness_fn_name = yaxis_fn_name
        y_axis_fn = trial_loader.get_fitness_fn(trial)
        y_axis_fns.append(y_axis_fn)

    x_axis_fn_name_plt = x_axis_fns[0].plotting_name
    y_axis_fn_name_plt = y_axis_fns[0].plotting_name

    # Combine x- and y-axis scores across randomization designs
    rand_mdl_names = np.array(list(trial_dict.keys()))
    z_pool_sizes = [trial.z_pool.shape[0] for trial in trial_dict.values()]
    all_rand_mdl_names = np.repeat(rand_mdl_names, z_pool_sizes)

    all_z_pool = [trial.z_pool for trial in trial_dict.values()]
    xvals = np.hstack(
        [
            x_axis_fn(z_pool).squeeze()
            for x_axis_fn, z_pool in zip(x_axis_fns, all_z_pool)
        ]
    )
    yvals = np.hstack(
        [
            y_axis_fn(z_pool).squeeze()
            for y_axis_fn, z_pool in zip(y_axis_fns, all_z_pool)
        ]
    )

    all_trial_df = pd.DataFrame.from_dict(
        {
            "rand_mdl_name": all_rand_mdl_names,
            x_axis_fn_name_plt: xvals,
            y_axis_fn_name_plt: yvals,
        }
    )

    return all_trial_df, x_axis_fn_name_plt, y_axis_fn_name_plt


def make_scatter_2dscore_disagg_fig(
    trial_set: Dict[str, SimulatedTrial],
    axis_fn_names: List[str],
    save_dir: Path,
    save_prefix: str
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
    all_trial_df, xaxis_fn_name_plt, yaxis_fn_name_plt = (
        _get_rand_xval_yval_df(trial_set, xaxis_fn_name, yaxis_fn_name)
    )

    # Make scatterplot
    _plot_scatter_2dscore_disagg(
        all_trial_df,
        xaxis_fn_name_plt,
        yaxis_fn_name_plt,
        save_dir,
        save_prefix,
    )
    all_data_rep_df_list.append(all_trial_df)

    # # Make scatterplot combining all data replicates
    # if len(trials) > 1:
    #     all_data_rep_df = pd.concat(all_data_rep_df_list, ignore_index=True)
    #     data_reps_str = "-".join([str(data_rep) for data_rep in range(len(trials))])
    #     _plot_scatter_2dscore_disagg(
    #         all_data_rep_df,
    #         xaxis_fn_name_plt,
    #         yaxis_fn_name_plt,
    #         save_dir,
    #         data_reps_str,
    #     )
    
def make_scatter_2dscore_disagg_mult_fig(
    trials: List[Dict[str, SimulatedTrial]],
    axis_fn_names: List[str],
    save_dirs: List[Path],
    save_prefix: str,
    legend: bool = True,
):
    xaxis_fn_name, yaxis_fn_name = axis_fn_names

    all_trial_df_list = []
    xaxis_fn_names = []
    yaxis_fn_names = []
    for trial_dict in trials:
        all_trial_df, xaxis_fn_name_plt, yaxis_fn_name_plt = (
            _get_rand_xval_yval_df(trial_dict, xaxis_fn_name, yaxis_fn_name)
        )
        xaxis_fn_names.append(xaxis_fn_name_plt)
        yaxis_fn_names.append(yaxis_fn_name_plt)
        all_trial_df_list.append(all_trial_df)

    min_xval, max_xval = axis_lim(all_trial_df_list, xaxis_fn_names)
    min_yval, max_yval = axis_lim(all_trial_df_list, yaxis_fn_names)

    for (
        all_trial_df,
        xaxis_fn_name_plt,
        yaxis_fn_name_plt,
        save_dir,
    ) in zip(
        all_trial_df_list, xaxis_fn_names, yaxis_fn_names, save_dirs
    ):
        _plot_scatter_2dscore_disagg(
            all_trial_df,
            xaxis_fn_name_plt,
            yaxis_fn_name_plt,
            save_dir,
            save_prefix,
            (min_xval, max_xval),
            (min_yval, max_yval),
            legend,
        )
