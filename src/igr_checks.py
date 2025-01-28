import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from typing import List, Dict

from src.aesthetics import (
    get_design_palette,
    design_color_mapping,
    get_design_hue_order,
    format_ax
)

def discriminatory_power(
    fitness_lbl: str,
    scores_1: np.ndarray,
    n_accept: int,
    ax: plt.Axes,
    scores_2:np.ndarray=None,
    agg_fn:callable=None,
    agg_kwargs:dict=None,
) -> None:
    """
    Plot histograms of fitness function scores (under IGR and IGRg) to
    check for adequate discriminatory power
    Args:
        - fitness_lbl: Label for fitness function
        - scores_1: Insepction metric scores for candidate pool
        - n_accept: Number of allocations accpeted
        - ax: Axes object for plot
        - scores_2: Scores from second inspection metric under IGR, if applicable
        - scores_2_g: Scores from second inspection metric under IGRg, if applicable
        - agg_fn: Aggregation function to apply to scores_1 and scores_2, if applicable
        - agg_kwargs: Keyword arguments for aggregation function, if applicable
    """
    scores_dict = {r"$\mathcal{Z}_{pool}$": scores_1}

    # Get cutoff scores for the accepted allocations
    score_cutoff = np.sort(scores_1)[n_accept]

    # If scores_2 is not None, stack scores from IGR and IGRg, then apply aggregation function
    # so that they have the same standardization
    if scores_2 is not None:
        scores_stacked = agg_fn(scores_1, scores_2, **agg_kwargs)
        scores = scores_stacked[: scores_1.shape[0]]
        score_cutoff = np.sort(scores)[n_accept]
        scores_dict = {r"$\mathcal{Z}_{pool}$": scores}

    scores_df = pd.DataFrame(scores_dict).melt(
        var_name="design", value_name="score"
    )

    # Make histogram of scores with vertical lines at cutoffs
    palette = {r"$\mathcal{Z}_{pool}$": "grey"}
    sns.histplot(
        scores_df,
        x="score",
        hue="design",
        palette=palette,
        bins=80,
        element="step",
        fill=False,
        line_kws={"linewidth": 1.5},
        stat="count",
        common_norm=False,
        ax=ax
    )
    ax.set_xscale("log")
    vline_clr = design_color_mapping("IGR", fitness_lbl)

    ax.axvline(score_cutoff, color=vline_clr, linewidth=1.5)
    ax.text(
        x=score_cutoff * 1.04,
        y=0.98,
        s="IGR s*",
        color=vline_clr,
        ha="left",
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        fontsize=26,
    )
    
    # Format plot
    # ax.set_title(fitness_lbl, fontsize=24, pad=5)
    ax.set_xlabel(r"$f(\mathbf{z})$", fontsize=26, loc="right")
    ax.set_ylabel("Number of Allocations", fontsize=26)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(axis="x", which="minor", rotation=30, labelsize=18)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.yaxis.get_offset_text().set_fontsize(24)

    legend = ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=26,
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
    )

def desiderata_tradeoffs_jntgrid(
    metric_lbls: List[str],
    fitness_lbl: str,
    design_to_scores: Dict[str, np.ndarray]
) -> sns.JointGrid:
    """
    Plot desiderata tradeoffs between inspection metrics in the accepted allocations
    Args:
        - metric_lbls: Labels for inspection metrics
        - fitness_lbl: Label for fitness function
        - design_to_scores: Dictionary mapping design name to inspection metric scores
    """
    # Combine scores from different designs into a single dataframe
    df_list = []
    for design, scores in design_to_scores.items():
        df = pd.DataFrame(
            {
                "design": design,
                metric_lbls[0]: scores[0],
                metric_lbls[1]: scores[1],
            }
        )
        df_list.append(df)
    df = pd.concat(df_list)

    # Plot inspection metric scores against each other, 
    # and include marginal histograms
    palette = get_design_palette(design_to_scores.keys(), fitness_lbl)
    hue_order = get_design_hue_order(design_to_scores.keys())

    jnt_grid = sns.JointGrid(
        data=df,
        x=metric_lbls[0],
        y=metric_lbls[1],
        hue="design",
        height=5,
        palette=palette,
        hue_order=hue_order
    )

    jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.5)
    jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)

    # Format plot
    jnt_grid.set_axis_labels(metric_lbls[0], metric_lbls[1], fontsize=22)
    jnt_grid.figure.suptitle(" +\n".join(fitness_lbl.split(" + ")), fontsize=26)
    jnt_grid.figure.subplots_adjust(top=0.85)

    jnt_grid.ax_joint = format_ax(jnt_grid.ax_joint, lbl_size=20)
    jnt_grid.ax_joint.legend(
        title=None,
        markerscale=1.5,
        fontsize=20,
        handlelength=1,
        labelspacing=0.2,
        handletextpad=0.2,
        borderpad=0.2,
        borderaxespad=0.2
    )

    return jnt_grid

def desiderata_tradeoffs(
    metric_lbls: List[str],
    fitness_lbl: str,
    design_to_scores: Dict[str, np.ndarray],
    ax: plt.Axes
):
     # Combine scores from different designs into a single dataframe
    df_list = []
    for design, scores in design_to_scores.items():
        df = pd.DataFrame(
            {
                "design": design,
                metric_lbls[0]: scores[0],
                metric_lbls[1]: scores[1],
            }
        )
        df_list.append(df)
    df = pd.concat(df_list)

    # Plot inspection metric scores against each other, 
    # and include marginal histograms
    palette = get_design_palette(design_to_scores.keys(), fitness_lbl)
    hue_order = get_design_hue_order(design_to_scores.keys())

    # Make scatterplot of inspection metric scores
    sns.scatterplot(
        data=df,
        x=metric_lbls[0],
        y=metric_lbls[1],
        hue="design",
        palette=palette,
        hue_order=hue_order,
        s=8,
        linewidth=0,
        alpha=0.6,
        ax=ax
    )

    # Format plot
    ax.set_xlabel(metric_lbls[0], fontsize=24)
    ax.set_ylabel(metric_lbls[1], fontsize=24)
    ax.set_title(f"Accepted Allocations", fontsize=24)
    ax = format_ax(ax, lbl_size=22)
    ax.legend(
        title=None,
        markerscale=1.5,
        fontsize=20,
        handlelength=1,
        labelspacing=0.2,
        handletextpad=0.2,
        borderpad=0.2,
        borderaxespad=0.2
    )

def desiderata_tradeoffs_pool(
    metric_lbls: List[str],
    scores_pool: List[np.ndarray],
    ax: plt.Axes
):
    """
    Plot desiderata tradeoffs between inspection metrics in 
    the candidate pool of allocations as a heatmap
    Args:
        - metric_lbls: Labels for inspection metrics
        - scores_pool: Inspection metric scores for candidate pool
        - ax: Axes object for plot
        - title: Title for plot
    """
    dt_fig = sns.histplot(
        x=scores_pool[0],
        y=scores_pool[1],
        bins=60,
        ax=ax,
        cbar=True,
        cmap=sns.color_palette("flare", as_cmap=True)
    )
    ax = format_ax(ax, lbl_size=20)
    ax.set_xlabel(metric_lbls[0], fontsize=24)
    ax.set_ylabel(metric_lbls[1], fontsize=24)

    cbar = dt_fig.collections[0].colorbar
    cbar.ax.tick_params(axis="y", labelsize=22)
    ax.set_title("Candidate Allocations", fontsize=24)

def overrestriction(
        fitness_lbl: str, 
        design_to_z_accepted: Dict[str, np.ndarray],
        ax: plt.Axes, 
        gfr:bool=False):
    """
    Plot histogram of the frequency with which each pair of units is 
    assigned the same treatment across accepted allocations
    Args:
        - fitness_lbl: Label for fitness function used in IGR and IGRg
        - design_to_z_accepted: Dictionary mapping design name to accepted allocations
        - ax: Axes object for plot
        - gfr: Flag indicating whether group formation randomization was used,
                which affects the axis label
    """
    same_z_dfs = []
    # For each pairs of units, calculate the fraction of allocations for which
    # they are assigned to the same arm
    for design, z_pool in design_to_z_accepted.items():
        comps = z_pool[:, :, np.newaxis] == z_pool[:, np.newaxis, :]
        p_same = np.mean(comps, axis=0)
        upper_tri_p_same = p_same[np.triu_indices(p_same.shape[0], k=1)]
        same_z_df = pd.DataFrame(
            {
                "design": design,
                "same_z": upper_tri_p_same,
            }
        )
        same_z_dfs.append(same_z_df)
    same_z_df = pd.concat(same_z_dfs)

    # Make histogram of the frequencies of same treatment assignments
    palette = get_design_palette(design_to_z_accepted.keys(), fitness_lbl)
    hue_order = get_design_hue_order(design_to_z_accepted.keys())

    sns.histplot(
        same_z_df,
        x="same_z",
        hue="design",
        hue_order=hue_order,
        palette=palette,
        binwidth=0.01,
        element="step",
        fill=False,
        alpha=0.6,
        kde=True,
        line_kws={"linewidth": 2},
        stat="probability",
        common_norm=False,
        ax=ax,
    )

    n_arms = z_pool.max() + 1

    # Add vertical line at 1/n_arms
    ax.axvline(1 / n_arms, color="black", linestyle="--", linewidth=1)

    # Format plot
    #(fitness_lbl, fontsize=24, pad=10)
    if gfr:
        ax.set_xlabel(r"$\hat{P}(z_i = z_j, g_i = g_j)$", fontsize=26)
    else:
        ax.set_xlabel(r"$\hat{P}(z_i = z_j)$", fontsize=26)
    ax.set_ylabel(r"Fraction of Pairs $(i, j)$", fontsize=26)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.tick_params(axis="both", which="major", labelsize=24)

    legend = ax.legend_
    handles = legend.legend_handles
    for h in handles:
        h.set_linewidth(2)
        h.set_alpha(1)
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=20,
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
    )

    # get overrestriction summary stats
    summary_df = same_z_df.groupby("design")["same_z"].describe().reset_index()

    return summary_df

def overrestriction_heatmap(
    fitness_lbl: str, 
    design_to_z_accepted: Dict[str, np.ndarray],
    design_to_ax: Dict[str, plt.Axes]):

    """
    Plot histogram of the frequency with which each pair of units is 
    assigned the same treatment across accepted allocations
    Args:
        - fitness_lbl: Label for fitness function used in IGR and IGRg
        - design_to_z_accepted: Dictionary mapping design name to accepted allocations
        - ax: Axes object for plot
        - gfr: Flag indicating whether group formation randomization was used,
                which affects the axis label
    """
    # For each pairs of units, calculate the fraction of allocations for which
    # they are assigned to the same arm
    fitness_lbl_terms = fitness_lbl.split(" + ")
    if len(fitness_lbl_terms) > 1:
        fitness_lbl = " +\n".join(fitness_lbl_terms)

    for design, z_pool in design_to_z_accepted.items():
        comps = z_pool[:, :, np.newaxis] == z_pool[:, np.newaxis, :]
        p_same = np.mean(comps, axis=0)
        upper_tri = np.triu_indices(p_same.shape[0], k=1)
        upper_tri_mask = np.zeros_like(p_same, dtype=bool)
        upper_tri_mask[upper_tri] = True
        p_same[~upper_tri_mask] = None
        
        design_to_ax[design].imshow(p_same, cmap='viridis', vmin=0, vmax=1)
        if 'IGR' in design:
            design_to_ax[design].set_title(f'{design} - {fitness_lbl}')
        else:
            design_to_ax[design].set_title(f'{design}')
        design_to_ax[design].set_xticks([])
        design_to_ax[design].set_yticks([])
        design_to_ax[design].set_aspect('equal')
    




