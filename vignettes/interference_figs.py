from argparse import ArgumentParser
from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns

from src.utils.aesthetics import color_mapping, get_palette, get_hue_order, format_ax, adjust_joint_grid_limits
from src.aggregators import LinComb
from src.utils.collate import perc_of_benchmark

def _parse_design(design):
    design_prefix = design.split(" - ")[0]
    fitness_lbl = design.split(" - ")[1]
    b_metric_term = fitness_lbl.split(" + ")[0]
    i_metric_term = fitness_lbl.split(" + ")[1]
    b_metric_name = b_metric_term.split("*")[1]
    i_metric_name = i_metric_term.split("*")[1]
    w_b = float(b_metric_term.split("*")[0])
    w_i = float(i_metric_term.split("*")[0])

    return design_prefix, fitness_lbl, b_metric_name, i_metric_name, w_b, w_i

def _strip_weights_from_design(design):
    design_prefix, _, b_metric_name, i_metric_name, _, _ = _parse_design(design)
    return f"{design_prefix} - {b_metric_name} + {i_metric_name}"

def interference_bias_rmse_rr_vs_weight(
    n_enum,
    n_accept,
    tau_size,
    mirror_type,
    res_dir,
    fig_dir
):
    res = pd.read_csv(res_dir / "res_collated.csv")
    res = res[(res["n_enum"] == n_enum) & (res["n_accept"] == n_accept) & (res["tau_size"] == tau_size) & (res["mirror_type"] == mirror_type)]
    res_igr = res[res["design"].str.contains("IGR")]
    res_igr = res_igr.assign(igr_type = res_igr["design"].apply(lambda x: _parse_design(x)[0]),
                             fitness_lbl = res_igr["design"].apply(lambda x: _parse_design(x)[1]),
                             w_balance = res_igr["design"].apply(lambda x: _parse_design(x)[4]),
                             design_no_weights = res_igr["design"].apply(lambda x: _strip_weights_from_design(x)))

    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=True)
    palette = get_palette(res_igr["design_no_weights"].unique())
    hue_order = get_hue_order(res_igr["design_no_weights"].unique())

    # Bias 
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
        ax = ax[0],
        legend=False
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
        ax = ax[0],
        errorbar=None
    )
    # ax[0].axhline(100, color='black', linestyle='--', linewidth=1, label="CR")
    ax[0].set_xlabel(r"$w_{MaxMahalanobis}$", fontsize=18)
    ax[0].set_ylabel("% CR Bias", fontsize=18)
    ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax[0].tick_params(axis="x", labelsize=16)
    ax[0].tick_params(axis="y", labelsize=16)
    ax[0].get_legend().set_visible(False)
 

    # RMSE
    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="perc_CR_rmse",
        hue="design_no_weights",
        linewidth=1,
        alpha=0.5,
        units="data_iter",
        estimator=None,
        palette=palette,
        hue_order=hue_order,
        ax = ax[1],
        legend=False
    )
    sns.lineplot(
        data=res_igr,
        x="w_balance",
        y="perc_CR_rmse",
        hue="design_no_weights",
        marker="o",
        markersize=4,
        markeredgecolor="black",
        linewidth=3,
        palette=palette,
        hue_order=hue_order,
        ax = ax[1],
        errorbar=None
    )
    # ax[1].axhline(100, color='black', linestyle='--', linewidth=1, label="CR")
    ax[1].set_xlabel(r"$w_{MaxMahalanobis}$", fontsize=18)
    ax[1].set_ylabel("% CR RMSE", fontsize=18)
    ax[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax[1].tick_params(axis="x", labelsize=16)
    ax[1].tick_params(axis="y", labelsize=16)
    ax[1].get_legend().set_visible(False)

    # Rejection rate
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
        ax = ax[2],
        legend=False
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
    cr_rr = res[res['design'] == 'CR']['rr']
    ax[2].axhline(cr_rr.mean(), color=color_mapping("CR"), linewidth=3, label="CR")
    for rr in cr_rr:
        ax[2].axhline(rr, color=color_mapping("CR"), linewidth=1, alpha=0.5)
    ax[2].set_xlabel(r"$w_{MaxMahalanobis}$", fontsize=18)
    ax[2].set_ylabel("Rejection Rate", fontsize=18)
    ax[2].tick_params(axis="x", labelsize=16)
    ax[2].tick_params(axis="y", labelsize=16)
    ax[2].get_legend().set_visible(False)
    
    handles, labels = ax[0].get_legend_handles_labels()
    ncol = 1
    bbox_to_anchor = (1.2, 0.5)
    fig.legend(
        handles,
        labels,
        title=None,
        fontsize=16,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        edgecolor="black",
    )

    save_fname = "bias_rmse_rr_vs_weight.svg"
    save_path = fig_dir / save_fname
    fig.savefig(save_path, bbox_inches="tight", transparent=True)

def interference_bias_rmse_rr_vs_enum(
    b_weights,
    i_weights,
    tau_size,
    mirror_type,
    res_dir,
    fig_dir
):
    res_df = pd.read_csv(res_dir / "res_collated.csv")

    # all_ymins = []
    # all_ymaxs = []
    # all_figs = []
    # all_axs = []
    for (b_w, i_w) in zip(b_weights, i_weights):
        res_subdf = res_df[((res_df["design"] == f"IGR - {b_w:.2f}*MaxMahalanobis + {i_w:.2f}*FracExpo")|
                                    (res_df["design"] == f"IGRg - {b_w:.2f}*MaxMahalanobis + {i_w:.2f}*FracExpo") |
                                    (res_df["design"] == f"IGR - {b_w:.2f}*MaxMahalanobis + {i_w:.2f}*InvEuclidDist") |
                                    (res_df["design"] == f"IGRg - {b_w:.2f}*MaxMahalanobis + {i_w:.2f}*InvEuclidDist") |
                                    (res_df["design"] == "CR")) & 
                                    (res_df["tau_size"] == tau_size) &
                                    (res_df["mirror_type"] == mirror_type)].copy()
        res_subdf["var"] = res_subdf["rmse"] - res_subdf["bias"]**2

        iter_group_cols = res_subdf.columns[~res_subdf.columns.str.contains("^.*?bias.*?$|^.*?rmse.*?$|^rr.*?$|^var|^design$")].tolist()
        res_subdf.sort_values(by=iter_group_cols, inplace=True)
        res_subdf["perc_CR_var"] = (
            res_subdf.groupby(iter_group_cols, dropna=False)
            .apply(lambda x: perc_of_benchmark("CR", x, "var"))
            .reset_index(drop=True)
            .sort_index()
            .values
        )

        n_accepts = np.sort(res_df["n_accept"].unique())
        fig, axs = plt.subplots(3, len(n_accepts), figsize=(6*len(n_accepts), 18), sharey='row', sharex='all')
        palette = get_palette(res_subdf['design'].unique())
        hue_order = get_hue_order(res_subdf['design'].unique())

        for i, n_accept in enumerate(n_accepts):
            sns.lineplot(
                data=res_subdf[(res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")],
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
                data=res_subdf[(res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")],
                x="n_enum",
                y="perc_CR_bias",
                hue="design",
                marker='o', 
                markersize=4,
                markeredgecolor="black",
                linewidth=3,
                ax=axs[0][i],
                palette=palette,
                hue_order=hue_order,
                errorbar=None
            )
            axs[0][i].set_title(f"m = {n_accept}", fontsize=16)
            axs[0][i].set_xlabel("")
            axs[0][i].set_ylabel("% CR Bias", fontsize=16)
            axs[0][i].tick_params(axis='y', which='major', labelsize=16)
            axs[0][i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            axs[0][i].get_legend().set_visible(False)

            sns.lineplot(
                data=res_subdf[(res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")],
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
                legend=False
            )
            sns.lineplot(
                data=res_subdf[(res_subdf["n_accept"] == n_accept) & (res_subdf["design"] != "CR")],
                x="n_enum",
                y="perc_CR_var",
                hue="design",
                marker='o', 
                markersize=4,
                markeredgecolor="black",
                linewidth=3,
                ax=axs[1][i],
                palette=palette,
                hue_order=hue_order,
                errorbar=None
            )
            axs[1][i].set_xlabel("")
            axs[1][i].set_ylabel("% CR Variance", fontsize=16)
            axs[1][i].tick_params(axis='y', which='major', labelsize=16)
            axs[1][i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            axs[1][i].get_legend().set_visible(False)

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
                legend=False
            )
            sns.lineplot(
                data=res_subdf[res_subdf["n_accept"] == n_accept],
                x="n_enum",
                y="rr",
                hue="design",
                linewidth=3,
                marker='o', 
                markersize=4,
                markeredgecolor="black",
                palette=palette,
                hue_order=hue_order,
                errorbar=None,
                ax=axs[2][i]
            )
            axs[2][i].set_xlabel("M", fontsize=16)
            axs[2][i].set_ylabel("Rejection Rate", fontsize=16)
            axs[2][i].tick_params(axis='both', which='major', labelsize=16)
            axs[2][i].ticklabel_format(axis="x", style="sci", scilimits=(0, 3), useMathText=True)
            axs[2][i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
            axs[2][i].get_legend().set_visible(False)
        
        handles, labels = axs[2][i].get_legend_handles_labels()
        ncol = 1
        bbox_to_anchor = (0.5, -0.1)
        fig.legend(
            handles,
            labels,
            title=None,
            fontsize=16,
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
    b_metrics,
    i_metrics,
    b_weights,
    i_weights,
    agg_fn,
    data_iter,
    tau_true,
    res_dir,
    fig_dir
):
    for b_metric in b_metrics:
        for i_metric in i_metrics:
            jnt_grids = []
            fitness_lbls = []
            for (b_weight, i_weight) in zip(b_weights, i_weights):
                fitness_lbl = f"{b_weight:.2f}*{b_metric} + {i_weight:.2f}*{i_metric}"
                fitness_lbls.append(fitness_lbl)

                all_designs = []
                all_tau_hats = []
                all_score_accepted_b = []
                all_score_accepted_i = []

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

                all_designs = np.concatenate(all_designs)
                all_tau_hats = np.concatenate(all_tau_hats)
                all_score_accepted_b = np.concatenate(all_score_accepted_b)
                all_score_accepted_i = np.concatenate(all_score_accepted_i)
                all_score_accepted = agg_fn(all_score_accepted_b, all_score_accepted_i, w1=b_weight, w2=i_weight)

                df = pd.DataFrame({
                    "design": all_designs,
                    "score": all_score_accepted,
                    "err": all_tau_hats - tau_true
                })

                hue_order = get_hue_order(df["design"].unique())
                palette = get_palette(df["design"].unique(), fitness_lbl)

                jnt_grid = sns.JointGrid(
                    data=df,
                    x="score",
                    y="err",
                    hue="design",
                    height=5,
                    hue_order=hue_order,
                    palette=palette
                )
                # Make scatterplot with marginal histograms
                jnt_grid.plot_joint(sns.scatterplot,s=8, linewidth=0, alpha=0.5)
                jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)

                jnt_grid.figure.suptitle(f"{fitness_lbl}", fontsize=20)
                jnt_grid.figure.subplots_adjust(top=0.85)

                jnt_grid.ax_joint.set_xlabel(r"$f(\mathbf{z})$", fontsize=20, loc="right")
                jnt_grid.ax_joint.set_ylabel(r"$\hat{\tau} - \tau$", fontsize=20)
                jnt_grid.ax_joint.set_xlim(0, jnt_grid.ax_joint.get_xlim()[1])
                format_ax(jnt_grid.ax_joint)

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
                jnt_grids.append(jnt_grid)

            # Set y-axis limits to be the same for all joint grids
            save_fnames = [f"{data_iter}_{fitness_lbl}_tau_hat_scatter.svg" for fitness_lbl in fitness_lbls]
            adjust_joint_grid_limits(jnt_grids, fig_dir, save_fnames)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="res/vig3_interference")
    parser.add_argument("--n-enum", type=int, default=int(1e5))
    parser.add_argument("--n-accept", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau-size", type=float, default=0.3)
    parser.add_argument("--mirror-type", type=str, default="all")
    parser.add_argument("--data-iter", type=int, default=0)

    parser.add_argument("--balance-metric", type=str, nargs="+", default=["MaxMahalanobis"])
    parser.add_argument("--interference-metric", type=str, nargs="+", default=["FracExpo", "InvEuclidDist"])
    parser.add_argument("--w1", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--w2", type=float, nargs="+", default=[0.75, 0.5, 0.25])

    args = parser.parse_args()

    out_dir = Path(args.out_dir) 
    tau_subdir = f"tau_size-{args.tau_size:.1f}"
    dgp_subdir = f"gamma-{args.gamma}" 
    enum_subdir = f"n_enum-{args.n_enum}"
    accept_subdir = f"n_accept-{args.n_accept}"
    mirror_subdir = f"mirror_type-{args.mirror_type}"

    interference_bias_rmse_rr_vs_weight(
        n_enum=args.n_enum,
        n_accept=args.n_accept,
        tau_size=args.tau_size,
        mirror_type=args.mirror_type,
        res_dir=out_dir / dgp_subdir,
        fig_dir=out_dir / dgp_subdir,
    )
    interference_bias_rmse_rr_vs_enum(
        b_weights=args.w1,
        i_weights=args.w2,
        tau_size=args.tau_size,
        mirror_type=args.mirror_type,
        res_dir=out_dir / dgp_subdir,
        fig_dir=out_dir / dgp_subdir
    )
    
    res_dir = out_dir / dgp_subdir / f"{args.data_iter}" / tau_subdir / enum_subdir / accept_subdir / mirror_subdir 
    fig_dir = res_dir / 'res_figs'
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    with open(out_dir / dgp_subdir / f"{args.data_iter}" / tau_subdir / "tau_true.pkl", "rb") as f:
        tau_true = pickle.load(f)

    interference_err_scatter(
        b_metrics=args.balance_metric,
        i_metrics=args.interference_metric,
        b_weights=args.w1,
        i_weights=args.w2,
        agg_fn = LinComb,
        data_iter=args.data_iter,
        tau_true=tau_true,
        res_dir=res_dir,
        fig_dir=fig_dir
    )
