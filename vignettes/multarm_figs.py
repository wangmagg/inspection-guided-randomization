from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns

from src.metrics import SMD, SignedMaxAbsSMD
from src.utils.aesthetics import get_palette, get_hue_order


def multarm_bal_boxplot(
    designs,
    metric_lbls,
    n_arms,
    X,
    comps,
    res_dir,
    fig_dir
):
    cov_names = X.columns
    cov_names = [cov_name.capitalize() for cov_name in cov_names]
    X = X.to_numpy()

    all_df_list = []
    for metric_lbl in metric_lbls:
        for design in designs:
            if design == "CR" or design == "GFR":
                with open(res_dir / design / "z.pkl", "rb") as f:
                    z_accepted = pickle.load(f)
            elif design == "QB":
                z_accepted = pd.read_csv(res_dir / design / "z.csv").to_numpy()
            elif design == "IGR" or design == "IGRg":
                with open(res_dir / design / f"{metric_lbl}_z.pkl", "rb") as f:
                    z_accepted = pickle.load(f)
                design = f"{design} - {metric_lbl}"
            else:
                raise ValueError(f"Unknown design: {design}")

            smd = SignedMaxAbsSMD(z_accepted, n_arms, comps, X)
            smd_df = pd.DataFrame(smd, columns=cov_names)
            smd_df['design'] = design
            all_df_list.append(smd_df)

    df = pd.concat(all_df_list)
    df = pd.melt(df, id_vars=['design'], var_name='Covariate', value_name='MaxSMD')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.boxplot(
        data=df,
        x="Covariate",
        y="MaxSMD",
        hue="design",
        fliersize=1,
        hue_order=get_hue_order(df['design'].unique()),
        palette=get_palette(df['design'].unique()),
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Covariate", fontsize=18)
    ax.set_ylabel("MaxSMD", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    legend = ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.5),
        loc="lower center",
        ncol=len(designs) // 2,
        edgecolor='black'
        )

    # Save figure
    save_path = fig_dir / "bal_boxplot.svg"
    print(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

def multarm_pairwise_bal_boxplot(
    designs,
    metric_lbls,
    n_arms,
    X,
    comps,
    res_dir,
    fig_dir
):
    cov_names = X.columns
    cov_names = [cov_name.capitalize() for cov_name in cov_names]
    X = X.to_numpy()

    all_df_list = []
    for metric_lbl in metric_lbls:
        for design in designs:
            if design == "CR" or design == "GFR":
                with open(res_dir / design / "z.pkl", "rb") as f:
                    z_accepted = pickle.load(f)
            elif design == "QB":
                z_accepted = pd.read_csv(res_dir / design / "z.csv").to_numpy()
            elif design == "IGR" or design == "IGRg":
                with open(res_dir / design / f"{metric_lbl}_z.pkl", "rb") as f:
                    z_accepted = pickle.load(f)
                design = f"{design} - {metric_lbl}"
            else:
                raise ValueError(f"Unknown design: {design}")
            
            smd = SMD(z_accepted, n_arms, comps, X)
            for i, comp in enumerate(comps):
                smd_comp = smd[:, i, :]
                smd_df = pd.DataFrame(smd_comp, columns=cov_names)
                smd_df['design'] = design
                smd_df['group_comp'] = f"{comp[0]} vs {comp[1]}"
                all_df_list.append(smd_df)
    df = pd.concat(all_df_list)

    fig, axs = plt.subplots(1, len(cov_names), figsize=(5 * len(cov_names), 5), sharey=True, sharex=True)
    hue_order = get_hue_order(df['design'].unique())
    palette = get_palette(df['design'].unique())

    for ax, cov in zip(axs, cov_names):
        df_cov = df[["group_comp", "design", cov]]
        sns.boxplot(
            data=df_cov,
            y="group_comp",
            x=cov,
            hue="design",
            hue_order=hue_order,
            palette=palette,
            width=0.75,
            linewidth=0.75,
            fliersize=1,
            ax=ax,
        )
        ax.axvline(0, color="black", linestyle="--")

        ax.set_title(f"{cov}".capitalize(), fontsize=20)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"SMD", fontsize=18)
        ax.set_ylabel(f"Group Comparison", fontsize=18)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.get_legend().set_visible(False)
    
    handles, labels = axs[0].get_legend_handles_labels()
    ncol = 1
    loc = "center right"
    bbox_to_anchor = (1.2, 0.5)
    fig.legend(
        handles,
        labels,
        title=None,
        fontsize=14,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        edgecolor="black",
    )

    # Save figure
    save_path = fig_dir / "pairwise_bal_boxplot.svg"
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    

def _add_se(df):
    if df["score_accepted"].nunique() == 1:
        df["score_bin_mid"] = df["score_accepted"].iloc[0]
        df["se"] = 0
    else:
        labels, bins = pd.qcut(
            df["score_accepted"],
            q=50,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        bins_mid = (bins[1:] + bins[:-1]) / 2
        df["score_bin_mid"] = bins_mid[labels]
        df = df.groupby(["design", "score_bin_mid"]).agg(
            se=("tau_hat", "sem")
        )
        df = df.reset_index()
    return df

def multarm_err_scatter(
    designs,
    metric_lbls,
    res_dir,
    fig_dir,
    arm_idx=1
):
    jnt_grids = []
    for metric_lbl in metric_lbls:
        metric_df = []
        for design in designs:
            if design == "CR" or design == "QB":
                tau_hat_fname = f"tau_hat.pkl"
            elif design == "IGR" or design == "IGRg":
                tau_hat_fname = f"{metric_lbl}_tau_hat.pkl"
            else:
                raise ValueError(f"Unknown design: {design}")
            
            score_fname = f"{metric_lbl}_scores.pkl"
            with open(res_dir / design / tau_hat_fname, "rb") as f:
                design_tau_hat = pickle.load(f)
            with open(res_dir / design / score_fname, "rb") as f:
                score_accepted = pickle.load(f)
            design_dict = {
                "design": design,
                "tau_hat": design_tau_hat[:, arm_idx],
                "score_accepted": score_accepted
            }
            design_df = pd.DataFrame(design_dict)
            design_df = _add_se(design_df)
            metric_df.append(design_df)

        df = pd.concat(metric_df)
        hue_order = get_hue_order(df["design"].unique())
        palette = get_palette(df["design"].unique(), metric_lbl)

        jnt_grid = sns.JointGrid(
            data=df,
            x="score_bin_mid",
            y="se",
            hue="design",
            height=5,
            hue_order=hue_order,
            palette=palette
        )
        # Make scatterplot with marginal histograms
        jnt_grid.plot_joint(sns.scatterplot, s=14, linewidth=0.1, edgecolor="black", alpha=0.8)
        jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)

        jnt_grid.figure.suptitle(f"{metric_lbl}", fontsize=20)
        jnt_grid.figure.subplots_adjust(top=0.85)

        jnt_grid.ax_joint.set_xlabel(r"$f(\mathbf{z})$", fontsize=20, loc="right")
        jnt_grid.ax_joint.set_ylabel(r"$SE_{\hat{\tau}}$", fontsize=20)
        jnt_grid.ax_joint.set_xlim(0, jnt_grid.ax_joint.get_xlim()[1])
        jnt_grid.ax_joint.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        jnt_grid.ax_joint.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        jnt_grid.ax_joint.tick_params(axis='both', which='major', labelsize=16)

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
    y_max = np.max([jnt_grid.ax_joint.get_ylim()[1] for jnt_grid in jnt_grids])
    y_min = np.min([jnt_grid.ax_joint.get_ylim()[0] for jnt_grid in jnt_grids])
    for jnt_grid in jnt_grids:
        jnt_grid.ax_joint.set_ylim(y_min, y_max)

    # Save figures
    for jnt_grid, metric_lbl in zip(jnt_grids, metric_lbls):
        save_fname = f"{metric_lbl}_tau_hat_scatter.svg"
        save_path = fig_dir / save_fname

        print(save_path)
        jnt_grid.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close()

def multarm_rmse_rr_vs_enum(
    res_dir,
    fig_dir
):
    res_df = pd.read_csv(res_dir / "res_collated.csv")
    n_accepts = res_df["n_accept"].unique()

    fig, axs = plt.subplots(2, len(n_accepts), figsize=(6*len(n_accepts), 12), sharey='row', sharex='all')
    hue_order = get_hue_order(res_df['design'].unique())
    palette = get_palette(res_df['design'].unique())

    for i, n_accept in enumerate(n_accepts):
        res_igr_subdf = res_df[(res_df["n_accept"] == n_accept) & ~res_df["design"].str.contains("CR")]
        sns.lineplot(
            data=res_igr_subdf,
            x="n_enum",
            y="perc_CR_rmse_2",
            hue="design",
            linewidth=1,
            alpha=0.5,
            palette=palette,
            hue_order=hue_order,
            units="data_iter",
            estimator=None,
            ax=axs[0][i],
            legend=False,
        )
        sns.lineplot(
            data=res_igr_subdf,
            x="n_enum",
            y="perc_CR_rmse_2",
            hue="design",
            marker='o', 
            markersize=4,
            markeredgecolor="black",
            linewidth=3,
            ax=axs[0][i],
            palette=palette,
            hue_order=hue_order,
            errorbar=None,
        )
        axs[0][i].set_title(f"m = {n_accept}")
        axs[0][i].set_ylabel("% CR RMSE", fontsize=16)
        axs[0][i].tick_params(axis='y', which='major', labelsize=16)

        res_subdf = res_df[(res_df["n_accept"] == n_accept)]
        sns.lineplot(
            data=res_subdf,
            x="n_enum",
            y="rr_2",
            hue="design",
            linewidth=1,
            alpha=0.5,
            palette=palette,
            hue_order=hue_order,
            units="data_iter",
            estimator=None,
            ax=axs[1][i],
            legend=False,
        )
        sns.lineplot(
            data=res_subdf,
            x="n_enum",
            y="rr_2",
            hue="design",
            marker='o', 
            markersize=4,
            markeredgecolor="black",
            linewidth=3,
            ax=axs[1][i],
            palette=palette,
            hue_order=hue_order,
            errorbar=None,
        )
        axs[1][i].set_ylabel("Rejection Rate", fontsize=16)
        axs[1][i].tick_params(axis='both', which='major', labelsize=16)
        axs[1][i].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

        if i == len(n_accepts) - 1:
            axs[0][i].get_legend().remove()
            axs[1][i].legend(bbox_to_anchor=(0.1, -0.2), ncol=3, fontsize=14)
        else:
            axs[0][i].get_legend().remove()
            axs[1][i].get_legend().remove()

    # Save figure
    save_path = fig_dir / "rmse_rr_vs_enum.svg"
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="res/vig1_multarm")
    parser.add_argument("--qb-dir", type=str, default="qb")
    parser.add_argument("--n-stu", type=int, default=80)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--n-enum", type=int, default=int(1e5))
    parser.add_argument("--n-accept", type=int, default=500)
    parser.add_argument("--data-iter", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir) 
    dgp_subdir = f"n-{args.n_stu}_arms-{args.n_arms}" 
    enum_subdir = f"n_enum-{args.n_enum}"
    accept_subdir = f"n_accept-{args.n_accept}"

    res_dir = out_dir / dgp_subdir / f"{args.data_iter}" / enum_subdir / accept_subdir
    fig_dir = res_dir / "res_figs"
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    X = pd.read_csv(out_dir / dgp_subdir / f"{args.data_iter}" / "X.csv")
    designs = ['CR', 'QB', 'IGR', 'IGRg']
    metric_lbls = ["SumMaxAbsSMD", "MaxMahalanobis"]
    comps = np.column_stack((np.zeros(args.n_arms - 1, dtype=int), np.arange(1, args.n_arms, dtype=int)))
        
    multarm_bal_boxplot(
        designs,
        metric_lbls,
        args.n_arms,
        X,
        comps,
        res_dir,
        fig_dir
    )

    multarm_pairwise_bal_boxplot(
        designs,
        metric_lbls,
        args.n_arms,
        X,
        comps,
        res_dir,
        fig_dir
    )

    multarm_err_scatter(
        designs,
        metric_lbls,
        res_dir,
        fig_dir,
        arm_idx=1
    )

    multarm_rmse_rr_vs_enum(
        res_dir=out_dir / dgp_subdir,
        fig_dir=out_dir / dgp_subdir,
    )