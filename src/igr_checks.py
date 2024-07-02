import numpy as np
import pandas as pd
import seaborn as sns

from src.aesthetics import (
    get_palette,
    get_two_tone_palette,
    color_mapping,
    get_hue_order,
    format_ax
)

def discriminatory_power(
    fitness_lbl,
    scores_1,
    scores_1_g,
    n_accept,
    ax,
    scores_2=None,
    scores_2_g=None,
    agg_fn=None,
    agg_kwargs=None,
):
    scores_dict = {
        r"IGR $\mathcal{Z}_{pool}$": scores_1,
        r"IGRg $\mathcal{Z}^{*}_{pool}$": scores_1_g,
    }
    score_cutoff = np.sort(scores_1)[n_accept]
    score_g_cutoff = np.sort(scores_1_g)[n_accept]

    if scores_2 is not None:
        scores_1_stacked = np.hstack([scores_1, scores_1_g])
        scores_2_stacked = np.hstack([scores_2, scores_2_g])
        scores_stacked = agg_fn(scores_1_stacked, scores_2_stacked, **agg_kwargs)
        scores = scores_stacked[: scores_1.shape[0]]
        score_cutoff = np.sort(scores)[n_accept]
        scores_g = scores_stacked[scores_1.shape[0]:]
        score_g_cutoff = np.sort(scores_g)[n_accept]
        scores_dict = {
            r"IGR $\mathcal{Z}_{pool}$": scores,
            r"IGRg $\mathcal{Z}^{*}_{pool}$": scores_g,
        }

    scores_df = pd.DataFrame(scores_dict).melt(
        var_name="design", value_name="score"
    )

    # Make histogram
    sns.histplot(
        scores_df,
        x="score",
        hue="design",
        palette=get_two_tone_palette(scores_dict.keys()),
        bins=80,
        element="step",
        fill=False,
        line_kws={"linewidth": 1.5},
        stat="count",
        common_norm=False,
        ax=ax
    )
    ax.set_xscale("log")
    vline_clr = color_mapping("IGR", fitness_lbl)
    ha_opts = ["left", "right"]
    pad_opts = [1.04, 0.96]
    igr_less = score_cutoff < score_g_cutoff
    igr_ha = ha_opts[igr_less]
    igr_pad = pad_opts[igr_less]
    igr_g_ha = ha_opts[~igr_less]
    igr_g_pad = pad_opts[~igr_less]

    ax.axvline(score_cutoff, color=vline_clr, linewidth=1.5)
    ax.text(
        x=score_cutoff * igr_pad,
        y=0.99,
        s="IGR s*",
        color=vline_clr,
        ha=igr_ha,
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        fontsize=20,
    )
    vline_g_clr = color_mapping("IGRg", fitness_lbl)
    ax.axvline(score_g_cutoff, color=vline_g_clr, linewidth=1.5)
    ax.text(
        x=score_g_cutoff * igr_g_pad,
        y=0.99,
        s="IGRg s*",
        color=vline_g_clr,
        ha=igr_g_ha,
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        fontsize=20,
    )

    fitness_lbl_terms = fitness_lbl.split(" + ")
    if len(fitness_lbl_terms) > 1:
        fitness_lbl = " +\n".join(fitness_lbl_terms)
    ax.set_title(fitness_lbl, fontsize=22, pad=20)
    ax.set_xlabel(r"$f(\mathbf{z})$", fontsize=20, loc="right")
    ax.set_ylabel("Number of Allocations", fontsize=20)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(axis="x", which="minor", rotation=30, labelsize=12)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.yaxis.get_offset_text().set_fontsize(18)

    legend = ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=20,
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
    )

def desiderata_tradeoffs(
    metric_lbls,
    fitness_lbl,
    design_to_scores
):
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

    palette = get_palette(design_to_scores.keys(), fitness_lbl)
    hue_order = get_hue_order(design_to_scores.keys())

    jnt_grid = sns.JointGrid(
        data=df,
        x=metric_lbls[0],
        y=metric_lbls[1],
        hue="design",
        height=5,
        palette=palette,
        hue_order=hue_order
    )

    # Make scatterplot with marginal histograms
    jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.5)
    jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
    jnt_grid.set_axis_labels(metric_lbls[0], metric_lbls[1], fontsize=18)

    jnt_grid.figure.suptitle(" +\n".join(fitness_lbl.split(" + ")), fontsize=22)
    jnt_grid.figure.subplots_adjust(top=0.85)

    jnt_grid.ax_joint = format_ax(jnt_grid.ax_joint, lbl_size=14)
    jnt_grid.ax_joint.legend(
        title=None,
        markerscale=1.5,
        fontsize=14,
        handlelength=1,
        labelspacing=0.2,
        handletextpad=0.2,
        borderpad=0.2,
        borderaxespad=0.2
    )

    return jnt_grid

def desiderata_tradeoffs_pool(
    metric_lbls,
    scores_pool,
    ax, 
    title=None
):
    dt_fig = sns.histplot(
        x=scores_pool[0],
        y=scores_pool[1],
        bins=60,
        ax=ax,
        cbar=True,
        cmap=sns.color_palette("flare", as_cmap=True)
    )
    ax = format_ax(ax, lbl_size=20)
    ax.set_xlabel(metric_lbls[0], fontsize=22)
    ax.set_ylabel(metric_lbls[1], fontsize=22)

    cbar = dt_fig.collections[0].colorbar
    cbar.ax.tick_params(axis="y", labelsize=20)

    if title:
        ax.set_title(title, fontsize=24)

def overrestriction(
        fitness_lbl, 
        design_to_z_accepted, 
        ax, 
        gfr=False):
    
    same_z_dfs = []
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

    palette = get_palette(design_to_z_accepted.keys(), fitness_lbl)
    hue_order = get_hue_order(design_to_z_accepted.keys())

    # Make histogram
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
    ax.axvline(1 / n_arms, color="black", linestyle="--", linewidth=1)

    fitness_lbl_terms = fitness_lbl.split(" + ")
    if len(fitness_lbl_terms) > 1:
        fitness_lbl = " +\n".join(fitness_lbl_terms)
    ax.set_title(fitness_lbl, fontsize=22)
    if gfr:
        ax.set_xlabel(r"$\hat{P}(z_i = z_j, g_i = g_j)$", fontsize=20)
    else:
        ax.set_xlabel(r"$\hat{P}(z_i = z_j)$", fontsize=20)
    ax.set_ylabel(r"Fraction of Pairs $(i, j)$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)

    legend = ax.legend_
    handles = legend.legend_handles
    for h in handles:
        h.set_linewidth(2)
        h.set_alpha(1)
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=16,
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
    )
