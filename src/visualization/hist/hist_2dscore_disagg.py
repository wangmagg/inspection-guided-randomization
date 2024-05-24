from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns
from pathlib import Path

from src.sims.trial import *
from typing import Dict


def format_ax(ax):
    max_x = ax.get_xlim()[1]
    max_y = ax.get_ylim()[1]

    if max_x < 0.1:
        x_fmt = "%.3f"
    else:
        x_fmt = "%.2f"
    if max_y < 0.1:
        y_fmt = "%.3f"
    else:
        y_fmt = "%.2f"

    ax.xaxis.set_major_formatter(FormatStrFormatter(x_fmt))
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))
    
    return ax

def _plot_hist_2dscore_disagg(
    score_df, score_fn_0_name, score_fn_1_name, save_dir, save_prefix, by_dgp=False
):
    if not by_dgp:
        grid_spec = {"width_ratios": (0.9, 0.07)}
        fig, (ax, cbar_ax) = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw=grid_spec)

        sns.histplot(
            data=score_df,
            x=score_fn_0_name,
            y=score_fn_1_name,
            bins=60,
            ax=ax,
            cbar=True,
            cbar_kws=dict(shrink=0.75),
            cbar_ax=cbar_ax, 
            cmap=sns.color_palette("flare", as_cmap=True)
        )

        ax.set_xlabel(score_fn_0_name, fontsize=24)
        ax.set_ylabel(score_fn_1_name, fontsize=24)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        cbar_ax.tick_params(axis="y", labelsize=20)
        
        ax = format_ax(ax)
    else:
        dgp_names = score_df["dgp_name"].unique()
        fig, axs = plt.subplots(
            1,
            len(dgp_names),
            figsize=(8 * len(dgp_names), 6),
            sharex=True,
            sharey=False
        )
        plt.subplots_adjust(wspace=0.5)
        
        if len(dgp_names) == 1:
            cur_plt = sns.histplot(
                data=score_df,
                x=score_fn_0_name,
                y=score_fn_1_name,
                bins=60,
                ax=axs,
                cbar=True,
                cmap=sns.color_palette("flare", as_cmap=True)
            )

            axs.set_xlabel(score_fn_0_name, fontsize=24)
            axs.set_ylabel(score_fn_1_name, fontsize=24)
            axs.tick_params(axis="x", labelsize=20)
            axs.tick_params(axis="y", labelsize=20)
            cbar = cur_plt.collections[0].colorbar
            cbar.ax.tick_params(axis="y", labelsize=20)
            axs.set_title(dgp_names, fontsize=24)

            axs = format_ax(axs)
        else:
            dgp_names_sorted = sorted(dgp_names)
            for dgp, ax in zip(dgp_names_sorted, axs):
                score_df_dgp = score_df[score_df["dgp_name"] == dgp]

                cur_plt = sns.histplot(
                    data=score_df_dgp,
                    x=score_fn_0_name,
                    y=score_fn_1_name,
                    bins=60,
                    ax=ax,
                    cbar=True,
                    cmap=sns.color_palette("flare", as_cmap=True)
                )

                ax.set_xlabel(score_fn_0_name, fontsize=24)
                ax.set_ylabel(score_fn_1_name, fontsize=24)
                ax.tick_params(axis="x", labelsize=20)
                ax.tick_params(axis="y", labelsize=20)
                ax.set_title(dgp, fontsize=24)
               
                cbar = cur_plt.collections[0].colorbar
                cbar.ax.tick_params(axis="y", labelsize=20)

                ax = format_ax(ax)

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_fname = f"{save_prefix}_hist2d_score_disagg.svg"
    save_path = save_dir / save_fname
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()


def make_hist_2dscore_disagg_fig(
    trial_set: Dict[str, SimulatedTrial],
    save_dir: Path,
    save_prefix: str
):
    trial_set_keys = list(trial_set.keys())
    cbr_keys = [key for key in trial_set_keys if "IGR - " in key]
    cbr_trial = trial_set[cbr_keys[0]]

    if not isinstance(cbr_trial.rand_mdl.fitness_fn, LinearCombinationFitness):
        raise ValueError("Fitness function must be a linear combination of metrics")
    
    ff_0 = cbr_trial.rand_mdl.fitness_fn.fitness_fns[0]
    ff_1 = cbr_trial.rand_mdl.fitness_fn.fitness_fns[1]

    score_fn_0_name = ff_0.plotting_name
    score_fn_1_name = ff_1.plotting_name

    score_df = pd.DataFrame(
        {
            score_fn_0_name: cbr_trial.fn_scores[0],
            score_fn_1_name: cbr_trial.fn_scores[1],
        }
    )

    _plot_hist_2dscore_disagg(
        score_df,
        score_fn_0_name,
        score_fn_1_name,
        save_dir,
        save_prefix
    )


def make_hist_2dscore_disagg_mult_dgp_fig(
    trial_set: Dict[str, SimulatedTrial],
    save_dir: Path,
    save_prefix: str,
):
    all_score_dfs = []
    for dgp_name, trial in trial_set.items():
        ff_0 = trial.rand_mdl.fitness_fn.fitness_fns[0]
        ff_1 = trial.rand_mdl.fitness_fn.fitness_fns[1]

        score_fn_0_name = ff_0.plotting_name
        score_fn_1_name = ff_1.plotting_name

        score_df = pd.DataFrame(
            {
                score_fn_0_name: trial.fn_scores[0],
                score_fn_1_name: trial.fn_scores[1],
                "dgp_name": dgp_name,
            }
        )

        all_score_dfs.append(score_df)
    
    score_df = pd.concat(all_score_dfs)

    _plot_hist_2dscore_disagg(
        score_df, score_fn_0_name, score_fn_1_name, save_dir, save_prefix, by_dgp=True
    )
