import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from src.design.fitness_functions import SumMaxAbsSMD
from src.sims.run_multarm_trial import SimulatedMultiArmTrial


def rand_mdl_subdir_order(rand_mdl_names):
    benchmark_names = [name for name in rand_mdl_names if "restricted" not in name]
    rand_mdl_names_sorted = np.sort(rand_mdl_names)
    igr_names = [
        name
        for name in rand_mdl_names_sorted
        if "restricted" in name and "genetic" not in name
    ]
    igrg_names = [
        name for name in rand_mdl_names_sorted if "restricted-genetic" in name
    ]
    return benchmark_names + igr_names + igrg_names


def se_handle_none(x):
    x_no_none = [el for el in x if el is not None]
    if len(x_no_none) == 0:
        return None
    return np.std(x.to_numpy()) / np.sqrt(len(x))


def perc_of_cr(group, col):
    ref_value = group[group["rand_mdl"] == "rand-complete"][col].values[0]
    perc_reduc = group[col].apply(lambda x: x / ref_value * 100)
    return perc_reduc


def get_trial_res(trial_fname, arms, n_per_arm, n_z=None):
    with open(trial_fname, "rb") as f:
        trial = pickle.load(f)

    trial.mapping, trial.use_cols = None, None
    trial.set_data_from_config()

    data_rep = (trial_fname.stem.split("_")[0].split("-")[-1],)
    run_seed = (trial_fname.stem.split("_")[1].split("-")[-1],)

    smd = SumMaxAbsSMD(trial.X)
    smd_scores = smd(trial.z_pool)
    mean_smd = np.mean(smd_scores)
    sd_smd = np.std(smd_scores)
    max_smd = np.max(smd_scores)
    min_smd = np.min(smd_scores)

    trial_res = {
        "data_rep": data_rep,
        "run_seed": run_seed,
        "rand_mdl": rand_mdl_subdir.name,
        "arms": arms,
        "n_per_arm": n_per_arm,
        "n_z": n_z,
        "n_cutoff": trial.config.n_cutoff,
        "n_accepted": trial.z_pool.shape[0],
        "mean_smd": mean_smd,
        "sd_smd": sd_smd,
        "max_smd": max_smd,
        "min_smd": min_smd,
        "tau_true": trial.tau_true,
        "bias": trial.bias,
        "rmse": trial.rmse,
        "rr": trial.rr,
    }

    return trial_res


if __name__ == "__main__":
    """
    Collect results from simulated multi-arm trials across different configurations
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--exp-subdir", type=str, default="mult-arm")
    parser.add_argument("--res-dir", type=str, default="res")
    args = parser.parse_args()

    res_dir = Path(args.output_dir) / args.exp_subdir
    all_trial_res = []

    for n_arm_subdir in res_dir.glob("arms-*"):
        arms = n_arm_subdir.name.split("-")[-1]
        for n_per_arm_subdir in n_arm_subdir.glob("n-per-arm-*"):
            n_per_arm = n_per_arm_subdir.name.split("-")[-1]
            for rand_mdl_subdir in n_per_arm_subdir.glob("rand-*"):
                for trial_fname in rand_mdl_subdir.glob("*.pkl"):
                    trial_res = get_trial_res(trial_fname, arms, n_per_arm)
                    all_trial_res.append(trial_res)
            for n_z_subdir in n_per_arm_subdir.glob("n-z-*"):
                n_z = n_z_subdir.name.split("-")[-1]
                for rand_mdl_subdir in n_z_subdir.glob("rand-*"):
                    for trial_fname in rand_mdl_subdir.glob("*.pkl"):
                        trial_res = get_trial_res(trial_fname, arms, n_per_arm, n_z)
                        all_trial_res.append(trial_res)

    all_trial_res = pd.DataFrame.from_records(all_trial_res)

    # Compute percentage reduction in RMSE from complete randomization for IGR
    all_trial_res["perc_cr_mse"] = (
        all_trial_res.groupby(["data_rep", "arms", "n_per_arm", "n_z"])
        .apply(lambda x: perc_of_cr(x, "rmse"))
        .reset_index(level=(0, 1, 2, 3), drop=True)
        .sort_index()
    )
    # Compute percentage reduction in RMSE from complete randomization for QB
    all_trial_res.loc[
        all_trial_res["rand_mdl"] == "rand-quick-block", "perc_cr_mse"
    ] = (
        all_trial_res.loc[
            all_trial_res["rand_mdl"].isin(["rand-quick-block", "rand-complete"])
        ]
        .groupby(["data_rep", "arms", "n_per_arm"])
        .apply(lambda x: perc_of_cr(x, "rmse"))
        .reset_index(level=(0, 1, 2), drop=True)
        .sort_index()
        .loc[all_trial_res["rand_mdl"] == "rand-quick-block"]
    )
    # Sort rand_mdl categories
    all_trial_res["rand_mdl"] = pd.Categorical(
        all_trial_res["rand_mdl"],
        categories=rand_mdl_subdir_order(all_trial_res["rand_mdl"].unique()),
        ordered=True,
    )
    all_trial_res.sort_values(
        by=["data_rep", "arms", "n_per_arm", "n_z", "rand_mdl"], inplace=True
    )
    all_trial_res_agg = (
        all_trial_res.groupby(["rand_mdl", "arms", "n_per_arm", "n_z"])
        .agg(
            mean_mean_smd=("mean_smd", "mean"),
            se_mean_smd=("mean_smd", lambda x: np.std(x.to_numpy()) / np.sqrt(len(x))),
            mean_sd_smd=("sd_smd", "mean"),
            se_sd_smd=("sd_smd", lambda x: np.std(x.to_numpy()) / np.sqrt(len(x))),
            mean_max_smd=("max_smd", "mean"),
            se_max_smd=("max_smd", lambda x: np.std(x.to_numpy()) / np.sqrt(len(x))),
            mean_min_smd=("min_smd", "mean"),
            se_min_smd=("min_smd", lambda x: np.std(x.to_numpy()) / np.sqrt(len(x))),
            mean_tau_true=("tau_true", "mean"),
            se_tau_true=("tau_true", lambda x: se_handle_none(x)),
            mean_bias=("bias", "mean"),
            se_bias=("bias", lambda x: se_handle_none(x)),
            mean_rmse=("rmse", "mean"),
            se_rmse=("rmse", lambda x: se_handle_none(x)),
            mean_rr=("rr", "mean"),
            se_rr=("rr", lambda x: se_handle_none(x)),
        )
        .reset_index()
    )

    res_dir = Path(args.res_dir) / args.exp_subdir
    if not res_dir.exists():
        res_dir.mkdir(parents=True)

    all_trial_res.to_csv(res_dir / f"{args.exp_subdir}_results.csv", index=False)
    all_trial_res_agg.to_csv(
        res_dir / f"{args.exp_subdir}_results_rep-agg.csv", index=False
    )
