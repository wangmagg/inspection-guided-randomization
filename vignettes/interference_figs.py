from argparse import ArgumentParser
from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns

from src.aesthetics import (
    design_color_mapping,
    get_design_palette,
    get_design_hue_order,
    format_ax,
    adjust_joint_grid_limits,
    save_joint_grids,
    setup_fig,
)
from src.igr_checks import discriminatory_power, overrestriction
from src.aggregators import LinComb


def _parse_design(design: str) -> tuple[str, str, str, str, float, float]:
    """
    Parse design string to extract the design prefix, the fitness label, the balance metric name,
    the interference metric name, and the weights applied to the balance and interference metrics.

    Ex: 
    design = "IGR - 0.25*MaxMahalanobis + 0.75*FracExpo"
    design_prefix = "IGR"
    fitness_lbl = "0.25*MaxMahalanobis + 0.75*FracExpo"
    b_metric_name = "MaxMahalanobis"
    i_metric_name = "FracExpo"
    w_b = 0.25
    w_i = 0.75
    """
    design_prefix = design.split(" - ")[0]

    if len(design.split(" - ")) == 1:
        return design_prefix, None, None, None, None, None
    else:
        fitness_lbl = design.split(" - ")[1]
        b_metric_term = fitness_lbl.split(" + ")[0]
        i_metric_term = fitness_lbl.split(" + ")[1]
        b_metric_name = b_metric_term.split("*")[1]
        i_metric_name = i_metric_term.split("*")[1]
        w_b = float(b_metric_term.split("*")[0])
        w_i = float(i_metric_term.split("*")[0])

        return design_prefix, fitness_lbl, b_metric_name, i_metric_name, w_b, w_i

def _strip_weights_from_design(design: str) -> str:
    """
    Remove the weights applied to the balance and interference metrics
    from the design string 

    Ex:
    design = "IGR - 0.25*MaxMahalanobis + 0.75*FracExpo"
    design_no_weights = "IGR - MaxMahalanobis + FracExpo"
    """
    design_prefix, _, b_metric_name, i_metric_name, _, _ = _parse_design(design)
    return f"{design_prefix} - {b_metric_name} + {i_metric_name}"


def interference_bias_var_rr_vs_weight(
    n_enum: int, n_accept:int, tau_size: float, mirror_type: str, res_dir: Path, fig_dir: Path
) -> None:
    """
    Make lineplot of bias, variance, and rejection rate versus the weight applied to the balance metric
    across data iterations for a given number of enumerated allocations, a given number of accepted allocations,
    a given effect size, and a given type of mirror allocation inclusion.

    Args:
        - n_enum: number of enumerated candidate allocations
        - n_accept: number of accepted allocations
        - tau_size: effect size
        - mirror_type: type of mirror allocation inclusion (all or good)
        - res_dir: directory containing the results
        - fig_dir: directory to save the figure to 
    """

    # Load results
    res = pd.read_csv(res_dir / "res_collated.csv")
    res = res[
        (res["n_enum"] == n_enum)
        & (res["n_accept"] == n_accept)
        & (res["tau_size"] == tau_size)
        & (res["mirror_type"] == mirror_type)
    ]
    res_igr = res[res["design"].str.contains("IGR")]
    res_igr = res_igr.assign(
        igr_type=res_igr["design"].apply(lambda x: _parse_design(x)[0]),
        fitness_lbl=res_igr["design"].apply(lambda x: _parse_design(x)[1]),
        w_balance=res_igr["design"].apply(lambda x: _parse_design(x)[4]),
        design_no_weights=res_igr["design"].apply(
            lambda x: _strip_weights_from_design(x)
        ),
    )

    # Set up figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    palette = get_design_palette(res_igr["design_no_weights"].unique())
    hue_order = get_design_hue_order(res_igr["design_no_weights"].unique())

    # Plot bias
    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="perc_CR_bias",
        hue="design_no_weights",
        linewidth=1,
        alpha=0.5,
        units="data_iter",
        estimator=None,
        palette=palette,
        hue_order=hue_order,
        ax=ax[0],
        legend=False,
    )
    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="perc_CR_bias",
        hue="design_no_weights",
        marker="o",
        markersize=4,
        markeredgecolor="black",
        linewidth=3,
        palette=palette,
        hue_order=hue_order,
        ax=ax[0],
        errorbar=None,
    )
    ax[0].set_xlabel(r"$w_{MaxMahalanobis}$", fontsize=22)
    ax[0].set_ylabel("% CR Bias", fontsize=22)
    ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax[0].tick_params(axis="x", labelsize=20)
    ax[0].tick_params(axis="y", labelsize=20)
    ax[0].get_legend().set_visible(False)

    # Plot variance
    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="perc_CR_var",
        hue="design_no_weights",
        linewidth=1,
        alpha=0.5,
        units="data_iter",
        estimator=None,
        palette=palette,
        hue_order=hue_order,
        ax=ax[1],
        legend=False,
    )
    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="perc_CR_var",
        hue="design_no_weights",
        marker="o",
        markersize=4,
        markeredgecolor="black",
        linewidth=3,
        palette=palette,
        hue_order=hue_order,
        ax=ax[1],
        errorbar=None,
    )
    ax[1].set_xlabel(r"$w_{MaxMahalanobis}$", fontsize=22)
    ax[1].set_ylabel("% CR Variance", fontsize=22)
    ax[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax[1].tick_params(axis="x", labelsize=20)
    ax[1].tick_params(axis="y", labelsize=20)
    ax[1].get_legend().set_visible(False)

    # Plot rejection rate
    cr_rr = res[res["design"] == "CR"]["rr"]
    ax[2].axhline(
        cr_rr.mean(), color=design_color_mapping("CR"), linewidth=3, label="CR"
    )
    for rr in cr_rr:
        ax[2].axhline(rr, color=design_color_mapping("CR"), linewidth=1, alpha=0.5)

    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="rr",
        hue="design_no_weights",
        linewidth=1,
        alpha=0.5,
        units="data_iter",
        estimator=None,
        palette=palette,
        hue_order=hue_order,
        ax=ax[2],
        legend=False,
    )

    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="rr",
        hue="design_no_weights",
        marker="o",
        markersize=4,
        markeredgecolor="black",
        linewidth=3,
        palette=palette,
        hue_order=hue_order,
        ax=ax[2],
        errorbar=None,
    )
    ax[2].set_xlabel(r"$w_{MaxMahalanobis}$", fontsize=22)
    ax[2].set_ylabel("Rejection Rate", fontsize=22)
    ax[2].tick_params(axis="x", labelsize=20)
    ax[2].tick_params(axis="y", labelsize=20)
    ax[2].get_legend().set_visible(False)

    # Add legend
    handles, labels = ax[2].get_legend_handles_labels()
    ncol = 1
    bbox_to_anchor = (1.2, 0.5)
    fig.legend(
        handles,
        labels,
        title=None,
        fontsize=20,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        edgecolor="black",
    )

    # Save figure
    save_fname = "bias_rmse_rr_vs_weight.svg"
    save_path = fig_dir / save_fname
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)


def interference_bias_var_rr_vs_enum(
    b_weights: list[float],
    i_weights: list[float],
    tau_size: float,
    mirror_type: str,
    res_dir: Path,
    fig_dir: Path,
) -> None:
    """
    Make lineplot of bias, variance, and rejection rate versus the number of enumerated allocations
    across data iterations for a different weight combinations applied to the balance and interference metrics,
    a given effect size, and a given type of mirror allocation inclusion. 
    Makes subplot for unique values of the number of accepted allocations.

    Args:
        - b_weights: list of weights applied to the balance metric
        - i_weights: list of weights applied to the interference metric
        - tau_size: effect size
        - mirror_type: type of mirror allocation inclusion (all or good)
        - res_dir: directory containing the results
        - fig_dir: directory to save the figure to
    """

    # Load results
    res_df = pd.read_csv(res_dir / "res_collated.csv")
    res_df["b_w"] = res_df["design"].apply(lambda x: _parse_design(x)[-2])
    res_df["i_w"] = res_df["design"].apply(lambda x: _parse_design(x)[-1])

    # Make plot for each weight combination
    for b_w, i_w in zip(b_weights, i_weights):

        # Filter results for the given weight combination, effect size, and mirror type
        res_subdf = res_df[
            (((res_df["b_w"] == b_w) & (res_df["i_w"] == i_w)) 
             | (res_df["design"] == "CR"))
            & (res_df["tau_size"] == tau_size)
            & (res_df["mirror_type"] == mirror_type)]

        # Set up figure
        n_accepts = np.sort(res_df["n_accept"].unique())
        fig, axs = plt.subplots(
            3,
            len(n_accepts),
            figsize=(4 * len(n_accepts), 12),
            sharey="row",
            sharex="all",
        )
        palette = get_design_palette(res_subdf["design"].unique())
        hue_order = get_design_hue_order(res_subdf["design"].unique())

        # Plot bias, variance, and rejection rate against number of enumerated allocations
        for i, n_accept in enumerate(n_accepts):

            # Plot bias
            sns.lineplot(
                data=res_subdf[
                    (res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")
                ],
                x="n_enum",
                y="perc_CR_bias",
                hue="design",
                linewidth=1,
                alpha=0.5,
                palette=palette,
                hue_order=hue_order,
                units="data_iter",
                estimator=None,
                legend=False,
                ax=axs[0][i],
            )
            sns.lineplot(
                data=res_subdf[
                    (res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")
                ],
                x="n_enum",
                y="perc_CR_bias",
                hue="design",
                marker="o",
                markersize=4,
                markeredgecolor="black",
                linewidth=3,
                ax=axs[0][i],
                palette=palette,
                hue_order=hue_order,
                errorbar=None,
            )
            axs[0][i].set_title(f"m = {n_accept}", fontsize=20)
            axs[0][i].set_xlabel("")
            axs[0][i].set_ylabel("% CR Bias", fontsize=20)
            axs[0][i].tick_params(axis="y", which="major", labelsize=20)
            axs[0][i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            axs[0][i].get_legend().set_visible(False)

            # Plot variance
            sns.lineplot(
                data=res_subdf[
                    (res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")
                ],
                x="n_enum",
                y="perc_CR_var",
                hue="design",
                linewidth=1,
                ax=axs[1][i],
                palette=palette,
                hue_order=hue_order,
                units="data_iter",
                estimator=None,
                alpha=0.5,
                legend=False,
            )
            sns.lineplot(
                data=res_subdf[
                    (res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")
                ],
                x="n_enum",
                y="perc_CR_var",
                hue="design",
                marker="o",
                markersize=4,
                markeredgecolor="black",
                linewidth=3,
                ax=axs[1][i],
                palette=palette,
                hue_order=hue_order,
                errorbar=None,
            )
            axs[1][i].set_xlabel("")
            axs[1][i].set_ylabel("% CR Variance", fontsize=20)
            axs[1][i].tick_params(axis="y", which="major", labelsize=20)
            axs[1][i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            axs[1][i].get_legend().set_visible(False)

            # Plot rejection rate
            sns.lineplot(
                data=res_subdf[res_subdf["n_accept"] == n_accept],
                x="n_enum",
                y="rr",
                hue="design",
                linewidth=1,
                ax=axs[2][i],
                palette=palette,
                hue_order=hue_order,
                units="data_iter",
                estimator=None,
                alpha=0.5,
                legend=False,
            )
            sns.lineplot(
                data=res_subdf[res_subdf["n_accept"] == n_accept],
                x="n_enum",
                y="rr",
                hue="design",
                linewidth=3,
                marker="o",
                markersize=4,
                markeredgecolor="black",
                palette=palette,
                hue_order=hue_order,
                errorbar=None,
                ax=axs[2][i],
            )
            axs[2][i].set_xlabel("M", fontsize=20)
            axs[2][i].set_ylabel("Rejection Rate", fontsize=20)
            axs[2][i].tick_params(axis="both", which="major", labelsize=20)
            axs[2][i].ticklabel_format(
                axis="x", style="sci", scilimits=(0, 3), useMathText=True
            )
            axs[2][i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
            axs[2][i].get_legend().set_visible(False)

        # Add legend
        handles, labels = axs[2][i].get_legend_handles_labels()
        ncol = 1
        bbox_to_anchor = (0.5, -0.1)
        fig.legend(
            handles,
            labels,
            title=None,
            fontsize=20,
            bbox_to_anchor=bbox_to_anchor,
            loc="lower center",
            ncol=ncol,
            edgecolor="black",
        )

        # Save figure
        save_fname = f"bias_rmse_rr_vs_enum_tau_size-{tau_size}_b-weight-{b_w}_i-weight-{i_w}.svg"
        save_path = fig_dir / save_fname
        fig.savefig(save_path, bbox_inches="tight", transparent=True)


def interference_err_scatter(
    b_metrics: list[str],
    i_metrics: list[str],
    b_weights: list[float],
    i_weights: list[float],
    agg_fn: callable,
    data_iter: int,
    tau_true: float,
    res_dir: Path,
    fig_dir: Path,
):
    """
    Make scatterplot of error in estimate versus fitness score for different weight combinations
    applied to the balance and interference metrics.

    Args:
        - b_metrics: list of balance metric names
        - i_metrics: list of interference metric names
        - b_weights: list of weights applied to the balance metric
        - i_weights: list of weights applied to the interference metric
        - agg_fn: aggregation function to combine the balance and interference metrics into a fitness score
        - data_iter: data iteration
        - tau_true: true treatment effect
        - res_dir: directory containing the results
        - fig_dir: directory to save the figure to
    """

    # Make separate figures for each metric combination
    for b_metric in b_metrics:
        for i_metric in i_metrics:
            all_jnt_grids = []
            all_fitness_lbls = []

            # Make subplots for each weight combination
            for b_weight, i_weight in zip(b_weights, i_weights):
                all_designs = []
                all_tau_hats = []
                all_score_accepted_b = []
                all_score_accepted_i = []

                # Load estimates and scores for CR design
                tau_hat_fname = f"tau_hat.pkl"
                score_fname = f"{b_metric} + {i_metric}_scores.pkl"
                with open(res_dir / "CR" / tau_hat_fname, "rb") as f:
                    tau_hat = pickle.load(f)
                    all_tau_hats.append(tau_hat)
                    all_designs.append(np.repeat("CR", len(tau_hat)))
                with open(res_dir / "CR" / score_fname, "rb") as f:
                    score_accepted_b, score_accepted_i = pickle.load(f)
                    all_score_accepted_b.append(score_accepted_b)
                    all_score_accepted_i.append(score_accepted_i)

                # Load estimates and scores for IGR and IGRg designs
                fitness_lbl = f"{b_weight:.2f}*{b_metric} + {i_weight:.2f}*{i_metric}"
                all_fitness_lbls.append(fitness_lbl)

                for design in ["IGR", "IGRg"]:
                    tau_hat_fname = f"{fitness_lbl}_tau_hat.pkl"
                    score_fname = f"{fitness_lbl}_scores.pkl"

                    with open(res_dir / design / tau_hat_fname, "rb") as f:
                        tau_hat = pickle.load(f)
                        all_tau_hats.append(tau_hat)
                        all_designs.append(np.repeat(design, len(tau_hat)))
                    with open(res_dir / design / score_fname, "rb") as f:
                        score_accepted_b, score_accepted_i = pickle.load(f)
                        all_score_accepted_b.append(score_accepted_b)
                        all_score_accepted_i.append(score_accepted_i)

                # Combine all estimates and scores into a single dataframe
                all_designs = np.concatenate(all_designs)
                all_tau_hats = np.concatenate(all_tau_hats)
                all_score_accepted_b = np.concatenate(all_score_accepted_b)
                all_score_accepted_i = np.concatenate(all_score_accepted_i)
                all_score_accepted = agg_fn(
                    all_score_accepted_b, all_score_accepted_i, w1=b_weight, w2=i_weight
                )
                df = pd.DataFrame(
                    {
                        "design": all_designs,
                        "score": all_score_accepted,
                        "err": all_tau_hats - tau_true,
                    }
                )

                # Set up joint grid
                hue_order = get_design_hue_order(df["design"].unique())
                palette = get_design_palette(df["design"].unique(), fitness_lbl)

                jnt_grid = sns.JointGrid(
                    data=df,
                    x="score",
                    y="err",
                    hue="design",
                    height=5,
                    hue_order=hue_order,
                    palette=palette,
                )

                # Make scatterplot of estimate error versus score with marginal histograms
                jnt_grid.plot_joint(sns.scatterplot, s=8, linewidth=0, alpha=0.5)
                jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)

                # Format joint grid
                jnt_grid.figure.suptitle(f"{fitness_lbl}", fontsize=24)
                jnt_grid.figure.subplots_adjust(top=0.85)
                jnt_grid.ax_joint.set_xlabel(
                    r"$f(\mathbf{z})$", fontsize=24, loc="right"
                )
                jnt_grid.ax_joint.set_ylabel(r"$\hat{\tau} - \tau$", fontsize=24)
                jnt_grid.ax_joint.set_xlim(0, jnt_grid.ax_joint.get_xlim()[1])
                format_ax(jnt_grid.ax_joint)

                # Add legend
                jnt_grid.ax_joint.legend(
                    title=None,
                    markerscale=1.5,
                    fontsize=18,
                    handlelength=1,
                    labelspacing=0.2,
                    handletextpad=0.2,
                    borderpad=0.2,
                    borderaxespad=0.2,
                )
                all_jnt_grids.append(jnt_grid)

            # Set y-axis limits to be the same for all joint grids
            adjust_joint_grid_limits(all_jnt_grids)

            # Save joint grids
            save_fnames = [
                f"{data_iter}_{fitness_lbl}_tau_hat_scatter.svg"
                for fitness_lbl in all_fitness_lbls
            ]
            save_joint_grids(all_jnt_grids, fig_dir, save_fnames)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="res/vig2_interference")
    parser.add_argument("--n-enum", type=int, default=int(1e5))
    parser.add_argument("--n-accept", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau-size", type=float, default=0.3)
    parser.add_argument("--mirror-type", type=str, default="all")
    parser.add_argument("--data-iter", type=int, default=0)

    parser.add_argument(
        "--balance-metric", type=str, nargs="+", default=["MaxMahalanobis"]
    )
    parser.add_argument(
        "--interference-metric",
        type=str,
        nargs="+",
        default=["FracExpo", "InvMinEuclidDist"],
    )
    parser.add_argument("--w1", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--w2", type=float, nargs="+", default=[0.75, 0.5, 0.25])

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tau_subdir = f"tau_size-{args.tau_size:.1f}"
    dgp_subdir = f"gamma-{args.gamma}"
    enum_subdir = f"n_enum-{args.n_enum}"
    accept_subdir = f"n_accept-{args.n_accept}"
    mirror_subdir = f"mirror_type-{args.mirror_type}"

    # Plot bias, variance, and rejection rate versus weight applied to balance metric
    interference_bias_var_rr_vs_weight(
        n_enum=args.n_enum,
        n_accept=args.n_accept,
        tau_size=args.tau_size,
        mirror_type=args.mirror_type,
        res_dir=out_dir / dgp_subdir,
        fig_dir=out_dir / dgp_subdir,
    )

    # Plot bias, RMSE, and rejection rate versus number of enumerated allocations
    interference_bias_var_rr_vs_enum(
        b_weights=args.w1,
        i_weights=args.w2,
        tau_size=args.tau_size,
        mirror_type=args.mirror_type,
        res_dir=out_dir / dgp_subdir,
        fig_dir=out_dir / dgp_subdir,
    )

    res_dir = (
        out_dir
        / dgp_subdir
        / f"{args.data_iter}"
        / tau_subdir
        / enum_subdir
        / accept_subdir
        / mirror_subdir
    )
    fig_dir = res_dir / "res_figs"
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    with open(
        out_dir / dgp_subdir / f"{args.data_iter}" / tau_subdir / "tau_true.pkl", "rb"
    ) as f:
        tau_true = pickle.load(f)

    # Scatter plot of error in estimate versus fitness score
    interference_err_scatter(
        b_metrics=args.balance_metric,
        i_metrics=args.interference_metric,
        b_weights=args.w1,
        i_weights=args.w2,
        agg_fn=LinComb,
        data_iter=args.data_iter,
        tau_true=tau_true,
        res_dir=res_dir,
        fig_dir=fig_dir,
    )

    # Replot IGR checks with only the specified weights
    for b_metric in args.balance_metric:
        for i_metric in args.interference_metric:
            dp_fig, dp_axs = setup_fig(ncols=len(args.w1), sharex=False, sharey=True)
            or_fig, or_axs = setup_fig(ncols=len(args.w1), sharex=False, sharey=True)

            for i, (w1, w2) in enumerate(zip(args.w1, args.w2)):
                fitness_lbl = f"{w1:.2f}*{b_metric} + {w2:.2f}*{i_metric}"

                # IGR check 1: Discriminatory power
                with open(
                    res_dir / "IGR" / f"{b_metric} + {i_metric}_scores_pool.pkl", "rb"
                ) as f:
                    scores_pool_igr = pickle.load(f)
                with open(
                    res_dir / "IGRg" / f"{b_metric} + {i_metric}_scores_pool.pkl", "rb"
                ) as f:
                    scores_pool_igrg = pickle.load(f)
                discriminatory_power(
                    fitness_lbl=fitness_lbl,
                    scores_1=scores_pool_igr[0],
                    scores_1_g=scores_pool_igrg[0],
                    scores_2=scores_pool_igr[1],
                    scores_2_g=scores_pool_igrg[1],
                    n_accept=args.n_accept,
                    agg_fn=LinComb,
                    agg_kwargs={"w1": w1, "w2": w2},
                    ax=dp_axs[i],
                )

                # IGR check 2: Overrestriction
                with open(res_dir / "CR" / "z.pkl", "rb") as f:
                    z_accepted_cr = pickle.load(f)
                with open(res_dir / "IGR" / f"{fitness_lbl}_z.pkl", "rb") as f:
                    z_accepted_igr = pickle.load(f)
                with open(res_dir / "IGRg" / f"{fitness_lbl}_z.pkl", "rb") as f:
                    z_accepted_igrg = pickle.load(f)
                overrestriction(
                    fitness_lbl=fitness_lbl,
                    design_to_z_accepted={
                        "CR": z_accepted_cr,
                        "IGR": z_accepted_igr,
                        "IGRg": z_accepted_igrg,
                    },
                    ax=or_axs[i],
                )
            dp_fig.savefig(
                fig_dir / f"{b_metric} + {i_metric}_discriminatory_power.svg",
                bbox_inches="tight",
                transparent=True,
            )
            or_fig.savefig(
                fig_dir / f"{b_metric} + {i_metric}_overrestriction.svg",
                bbox_inches="tight",
                transparent=True,
            )
