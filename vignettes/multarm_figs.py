from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns

from src.metrics import SMD, SignedMaxAbsSMD
from src.aesthetics import get_design_palette, get_design_hue_order, adjust_joint_grid_limits, save_joint_grids


def multarm_bal_boxplot(
    designs: list[str],
    metric_lbls: list[str],
    n_arms: int,
    X: pd.DataFrame,
    comps: np.ndarray,
    res_dir: Path,
    fig_dir: Path
) -> None:
    """
    Create a boxplot of the max SMD for each covariate across designs.

    Args:
        - designs: list of design names to plot (e.g., ["CR", "QB", "IGR", "IGRg"])
        - metric_lbls: list of metric labels (e.g., ["SumMaxAbsSMD", "MaxMahalanobis"])
        - n_arms: number of arms in the experiment
        - X: covariate dataframe
        - comps: array of pairwise arm comparisons to make 
        - res_dir: directory containing the results
        - fig_dir: directory to save the figure
    """

    # Get covariate names to use as x-axis labels
    cov_names = X.columns
    cov_names = [cov_name.capitalize() for cov_name in cov_names]
    X = X.to_numpy()

    # Load results and compute SMD for each design for each covariate
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

    # Combine results into a single dataframe
    df = pd.concat(all_df_list)
    df = pd.melt(df, id_vars=['design'], var_name='Covariate', value_name='MaxSMD')
    
    # Create boxplot
    fig, ax = plt.subplots(1, 1, figsize=(2*len(cov_names), 4))
    sns.boxplot(
        data=df,
        x="Covariate",
        y="MaxSMD",
        hue="design",
        fliersize=1,
        hue_order=get_design_hue_order(df['design'].unique()),
        palette=get_design_palette(df['design'].unique()),
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Covariate", fontsize=26)
    ax.set_ylabel("MaxSMD", fontsize=26)
    ax.tick_params(axis='x', labelsize=20, length=6, width=1.5)
    ax.tick_params(axis='y', labelsize=20, length=6, width=1.5)

    # Add legend
    legend = ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=20,
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
    designs: list[str],
    metric_lbls: list[str],
    n_arms: int,
    X: pd.DataFrame,
    comps: np.ndarray,
    res_dir: Path,
    fig_dir: Path
):
    """
    Create a boxplot of the SMD for each covariate for each pairwise comparison 
    between arms across designs.
    
    Args:
        - designs: list of design names to plot (e.g., ["CR", "QB", "IGR", "IGRg"])
        - metric_lbls: list of metric labels (e.g., ["SumMaxAbsSMD", "MaxMahalanobis"])
        - n_arms: number of arms in the experiment
        - X: covariate dataframe
        - comps: array of pairwise arm comparisons to make 
        - res_dir: directory containing the results
        - fig_dir: directory to save the figure
    """
    # Get covariate names to use as x-axis labels
    cov_names = X.columns
    cov_names = [cov_name.capitalize() for cov_name in cov_names]
    X = X.to_numpy()

    # Load results and compute SMD for each design for each covariate and pairwise comparison
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

            max_smd = SignedMaxAbsSMD(z_accepted, n_arms, comps, X)
            max_smd_df = pd.DataFrame(max_smd, columns=cov_names)
            max_smd_df['design'] = design
            max_smd_df['group_comp'] = "Max"
            all_df_list.append(max_smd_df)

    # Combine results into a single dataframe
    df = pd.concat(all_df_list)

    # Set up boxplot
    fig, axs = plt.subplots(1, len(cov_names), figsize=(3 * len(cov_names), 5), sharey=True, sharex=True)
    hue_order = get_design_hue_order(df['design'].unique())
    palette = get_design_palette(df['design'].unique())

    # Create boxplot for each covariate
    for ax, cov in zip(axs, cov_names):
        df_cov = df[["group_comp", "design", cov]]
        sns.boxplot(
            data=df_cov,
            y="group_comp",
            x=cov,
            hue="design",
            hue_order=hue_order,
            order=["Max"] + [f"{comp[0]} vs {comp[1]}" for comp in comps],
            palette=palette,
            width=0.85,
            linewidth=1,
            fliersize=1.5,
            ax=ax,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_title(f"{cov}".capitalize(), fontsize=24)
        ax.set_xlim(-1.2, 1.2)
        ax.set_xlabel(f"SMD", fontsize=22)
        ax.set_ylabel(f"Group Comparison", fontsize=22)
        ax.tick_params(axis="x", labelsize=20, length=6, width=1.5)
        ax.tick_params(axis="y", labelsize=20, length=6, width=1.5)
        ax.get_legend().set_visible(False)
    
    # Add legend
    handles, labels = axs[0].get_legend_handles_labels()
    ncol = 1
    loc = "center right"
    bbox_to_anchor = (1.2, 0.5)
    fig.legend(
        handles,
        labels,
        title=None,
        fontsize=18,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        edgecolor="black",
    )
    fig.suptitle("Pairwise Per-Covariate Balance", fontsize=26, fontweight="demibold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    # Save figure
    save_path = fig_dir / "pairwise_bal_boxplot.svg"
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    

def _add_se(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column for the standard error of the estimated effects within 
    each bins of the scores of accepted allocations

    Args:
        - df: dataframe containing the estimated effects and scores of accepted allocations
    """
    # If all scores are the same, set the standard error to 0
    if df["score_accepted"].nunique() == 1:
        df["score_bin_mid"] = df["score_accepted"].iloc[0]
        df["se"] = 0
    else:
        # Separate scores into 50 bins and get the midpoints of each bin
        labels, bins = pd.qcut(
            df["score_accepted"],
            q=50,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        bins_mid = (bins[1:] + bins[:-1]) / 2
        df["score_bin_mid"] = bins_mid[labels]

        # Compute the standard error of the estimated effects within each bin
        df = df.groupby(["design", "score_bin_mid"]).agg(
            se=("tau_hat", "sem")
        )
        df = df.reset_index()
    return df

def multarm_err_scatter(
    designs: list[str],
    metric_lbls: list[str],
    res_dir: Path,
    fig_dir: Path,
    arm_idx: int=1
):
    """
    Create a scatterplot of the standard error of the estimated treatment effects
    against the score of the accepted allocations for each design and metric.

    Args:
        - designs: list of design names to plot (e.g., ["CR", "QB", "IGR", "IGRg"])
        - metric_lbls: list of metric labels (e.g., ["SumMaxAbsSMD", "MaxMahalanobis"])
        - res_dir: directory containing the results
        - fig_dir: directory to save the figure
        - arm_idx: index of the arm to plot the effect estimate errors for
    """

    # Create joint grid for each metric
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
            
            # Load scores and effect estimates
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
            # Add standard error column
            design_df = pd.DataFrame(design_dict)
            design_df = _add_se(design_df)
            metric_df.append(design_df)

        # Combine results into a single dataframe
        df = pd.concat(metric_df)
        hue_order = get_design_hue_order(df["design"].unique())
        palette = get_design_palette(df["design"].unique(), metric_lbl)

        # Create joint grid
        jnt_grid = sns.JointGrid(
            data=df,
            x="score_bin_mid",
            y="se",
            hue="design",
            height=5,
            hue_order=hue_order,
            palette=palette
        )

        # Make scatterplot of standard errors of effect estimates versus scores,
        # including marginal histograms for each axis
        jnt_grid.plot_joint(sns.scatterplot, s=14, linewidth=0.1, edgecolor="black", alpha=0.8)
        jnt_grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)

        # Format plot
        jnt_grid.figure.suptitle(f"{metric_lbl}", fontsize=24)
        jnt_grid.figure.subplots_adjust(top=0.85)

        jnt_grid.ax_joint.set_xlabel(r"$f(\mathbf{z})$", fontsize=24, loc="right")
        jnt_grid.ax_joint.set_ylabel(r"$SE_{\hat{\tau}}$", fontsize=24)
        jnt_grid.ax_joint.set_xlim(0, jnt_grid.ax_joint.get_xlim()[1])
        jnt_grid.ax_joint.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        jnt_grid.ax_joint.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        jnt_grid.ax_joint.tick_params(axis='both', which='major', labelsize=16)

        # Add legend
        jnt_grid.ax_joint.legend(
            title=None,
            markerscale=1.5,
            fontsize=18,
            handlelength=1,
            labelspacing=0.2,
            handletextpad=0.2,
            borderpad=0.2,
            borderaxespad=0.2
        )
        jnt_grids.append(jnt_grid)

    # Set y-axis limits to be the same for all joint grids
    adjust_joint_grid_limits(jnt_grids, adjust_x=False, adjust_y=True)
    
    # Save figures
    save_fnames = [f"{metric_lbl}_tau_hat_scatter.svg" for metric_lbl in metric_lbls]
    save_joint_grids(jnt_grids, fig_dir, save_fnames)

def multarm_rmse_rr_vs_enum(
    res_dir: Path,
    fig_dir: Path,
    arm_idx: int=2
) -> ModuleNotFoundError:
    """
    Make a lineplot of the RMSE and rejection rate versus the number of enumerated allocations
    across data iterations. Make subplots for unique values of the number of accepted allocations.

    Args:
        - res_dir: directory containing the RMSE and rejection rate results
        - fig_dir: directory to save the figure to
    """

    # Load results
    res_df = pd.read_csv(res_dir / "res_collated.csv")
    n_accepts = res_df["n_accept"].unique()

    # Set up figure
    fig, axs = plt.subplots(2, len(n_accepts), figsize=(6*len(n_accepts), 12), sharey='row', sharex='all')
    hue_order = get_design_hue_order(res_df['design'].unique())
    palette = get_design_palette(res_df['design'].unique())

    # Plot RMSE and rejection rate against the number of enumerated allocations
    for i, n_accept in enumerate(n_accepts):
        res_igr_subdf = res_df[(res_df["n_accept"] == n_accept) & ~res_df["design"].str.contains("CR")]

        # Plot RMSE
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
        axs[0][i].set_title(f"m = {n_accept}", fontsize=20)
        axs[0][i].set_xlabel("")
        axs[0][i].set_ylabel("% CR RMSE", fontsize=20)
        axs[0][i].tick_params(axis='y', which='major', labelsize=20)

        # Plot rejection rate
        res_subdf = res_df[(res_df["n_accept"] == n_accept)]
        sns.lineplot(
            data=res_subdf,
            x="n_enum",
            y=f"rr_{arm_idx}",
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
            y=f"rr_{arm_idx}",
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
        axs[1][i].set_xlabel("M", fontsize=20)
        axs[1][i].set_ylabel("Rejection Rate", fontsize=20)
        axs[1][i].tick_params(axis='both', which='major', labelsize=20)
        axs[1][i].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        axs[1][i].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        # Add legend
        if i == len(n_accepts) - 1:
            axs[0][i].get_legend().remove()
            axs[1][i].legend(bbox_to_anchor=(0.1, -0.2), ncol=3, fontsize=18)
        else:
            axs[0][i].get_legend().remove()
            axs[1][i].get_legend().remove()

    # Save figure
    save_path = fig_dir / "rmse_rr_vs_enum.svg"
    print(save_path)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="res/vig0_multarm")
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

    # Create directory for saving figures
    res_dir = out_dir / dgp_subdir / f"{args.data_iter}" / enum_subdir / accept_subdir
    fig_dir = res_dir / "res_figs"
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    # Load data and set parameters
    X = pd.read_csv(out_dir / dgp_subdir / f"{args.data_iter}" / "X.csv")
    designs = ['CR', 'QB', 'IGR', 'IGRg']
    metric_lbls = ["SumMaxAbsSMD", "MaxMahalanobis"]
    comps = np.column_stack((np.zeros(args.n_arms - 1, dtype=int), np.arange(1, args.n_arms, dtype=int)))

    # Plot balance on each covariate across all arm comparisons
    multarm_bal_boxplot(
        designs,
        metric_lbls,
        args.n_arms,
        X,
        comps,
        res_dir,
        fig_dir
    )

    # Plot balance on each covariate across each pairwise comparisons between arms
    multarm_pairwise_bal_boxplot(
        designs,
        metric_lbls,
        args.n_arms,
        X,
        comps,
        res_dir,
        fig_dir
    )

    # Plot standard error of the estimated effects against the score of the accepted allocations
    multarm_err_scatter(
        designs,
        metric_lbls,
        res_dir,
        fig_dir,
        arm_idx=1
    )

    # Plot RMSE and rejection rate versus the number of enumerated allocations
    multarm_rmse_rr_vs_enum(
        res_dir=out_dir / dgp_subdir,
        fig_dir=out_dir / dgp_subdir,
    )