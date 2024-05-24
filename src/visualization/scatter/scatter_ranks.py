from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from src.design.fitness_functions import Fitness
from src.sims import trial_loader
from src.sims.trial import SimulatedTrial


def _plot_scatter_ranks(df: pd.DataFrame, save_dir: Path, save_prefix: str) -> None:
    for rand_mdl_name in df["rand_mdl_name"].unique():
        df_rand_mdl = df[df["rand_mdl_name"] == rand_mdl_name]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(
            df_rand_mdl["error_rank"],
            df_rand_mdl["score_rank"],
            c=df_rand_mdl["error"],
            cmap="viridis_r",
            s=6,
            alpha=0.8,
        )
        fig.colorbar(ax.collections[0], ax=ax, orientation="vertical")
        ax.axhline(1000, color="black", linestyle="--")
        ax.axvline(1000, color="black", linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Error Rank", fontsize=24)
        ax.set_ylabel("Score Rank", fontsize=24)
        save_path = save_dir / f"{save_prefix}_scatter_rank.png"
        fig.savefig(save_path)
        plt.close()


def _get_rank_df(
    trial_dict: Dict[str, SimulatedTrial], fitness_fn: Fitness
) -> pd.DataFrame:

    rank_dfs = []
    for rand_mdl_name, trial in trial_dict.items():
        trial.rand_mdl.n_z = 50000
        z_pool = trial.rand_mdl.sample_mult_z()
        if "Set" in rand_mdl_name:
            mapping = (
                trial.X[["settlement_id", "school_id"]]
                .groupby("school_id")
                .agg(max)["settlement_id"]
            )
            z_pool = np.vstack([z[mapping] for z in z_pool])
        elif "GR" in rand_mdl_name:
            z_pool = np.vstack([z[trial.mapping] for z in z_pool])

        scores = fitness_fn(z_pool)

        trial.observed_outcome_mdl = trial_loader.get_observed_outcome_mdl(trial)
        y_obs_pool = trial.observed_outcome_mdl(z_pool, trial.y_0, trial.y_1)
        estimator = trial_loader.get_estimator(trial)
        tau_hats = np.array(
            [estimator.estimate(z, y_obs) for z, y_obs in zip(z_pool, y_obs_pool)]
        )
        tau_true = (trial.y_1 - trial.y_0).mean()
        sq_error = (tau_hats - tau_true) ** 2

        score_sorted = np.argsort(scores)
        error_sorted = np.argsort(sq_error)

        score_rank = np.zeros(len(scores), dtype=int)
        error_rank = np.zeros(len(sq_error), dtype=int)
        score_rank[score_sorted] = np.arange(len(scores))
        error_rank[error_sorted] = np.arange(len(sq_error))
        rank_df = pd.DataFrame(
            {
                "score_rank": score_rank,
                "error_rank": error_rank,
                "error": sq_error,
                "rand_mdl_name": rand_mdl_name,
            }
        )
        rank_dfs.append(rank_df)

    rank_df = pd.concat(rank_dfs)
    return rank_df


def make_scatter_ranks_fig(
    trial_set: Dict[str, SimulatedTrial],
    fitness_fn: Fitness,
    save_dir: Path,
    save_prefix: str
):

    plt.rcParams["text.usetex"] = True

    rank_df = _get_rank_df(trial_set, fitness_fn)
    _plot_scatter_ranks(rank_df, save_dir, save_prefix)
