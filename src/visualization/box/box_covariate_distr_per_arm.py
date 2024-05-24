from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict

from src.sims.trial import SimulatedTrial


def make_covariate_distr_per_arm_fig(
    trial_set: Dict[str, SimulatedTrial], fig_dir: Path, save_prefix: str
) -> None:
    """
    Helper function to make boxplot of covariate means per arm for each covariate,
    with a separate figure for each data replicate and randomization design

    Args:
        - data_rep_to_trials_dict: maps data replicate to trials under each randomization design
        - fig_dir: directory to save figure
    """
    plt.rcParams["text.usetex"] = True

    for rand_mdl, trial in trial_set.items():
        all_z_df = []
        df = trial.X
        # Get mean covariate values per arm for
        # each treatment assignment in the pool
        for z in trial.z_pool:
            df["z"] = z.astype(int)
            df_mean = df.groupby("z").mean().reset_index()
            all_z_df.append(df_mean)

        # Melt dataframe into long format
        df = pd.concat(all_z_df)
        df_long = df.melt(id_vars=["z"], var_name="covariate", value_name="value")

        # Make boxplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.boxplot(data=df_long, x="covariate", y="value", hue="z", ax=ax)

        legend = ax.legend_
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
        ax.legend(
            handles=handles,
            labels=labels,
            title="Arm",
            fontsize=12,
            title_fontsize=12,
            bbox_to_anchor=(0.5, -0.2),
            loc="lower center",
            ncols=df["z"].nunique(),
        )

        # Save figure
        save_fname = f"{save_prefix}_covariate_distr_per_arm.svg"
        save_subdir = fig_dir / f"rand-{rand_mdl}"
        if not save_subdir.exists():
            save_subdir.mkdir(parents=True)
        fig.savefig(
            save_subdir / save_fname, dpi=300, bbox_inches="tight", transparent=True
        )
        plt.close()
