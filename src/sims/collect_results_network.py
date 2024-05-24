import pickle
from pathlib import Path
import pandas as pd
import argparse

from src.sims.trial import SimulatedTrial
from src.sims.run_network_trial import SimulatedNetworkTrial
from src.sims.collect_utils import rand_mdl_subdir_order, perc_of_cr

if __name__ == "__main__":
    """
    Collect results from simulated network trials across different configurations
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--exp-subdir', type=str, default='network')
    parser.add_argument('--res-dir', type=str, default='res')
    args = parser.parse_args()

    exp_output_subdir = Path(args.output_dir) / args.exp_subdir
    all_trial_res = []
    for net_mdl_subdir in exp_output_subdir.glob('net-*'):
        for n_z_subdir in net_mdl_subdir.glob('n-z-*'):
            for po_mdl_subdir in n_z_subdir.glob('po-*'):
                for rand_mdl_subdir in po_mdl_subdir.glob('rand-*'):
                    for trial_fname in rand_mdl_subdir.glob('*.pkl'):
                        print(trial_fname)
                        with open(trial_fname, 'rb') as f:
                            trial = pickle.load(f)
                        trial_res = {
                                    "data_rep": trial_fname.stem.split("_")[0].split('-')[-1],
                                    "run_seed": trial_fname.stem.split("_")[1].split('-')[-1],
                                    'rand_mdl': rand_mdl_subdir.name,
                                    'net_mdl': net_mdl_subdir.name,
                                    'po_mdl': po_mdl_subdir.name,
                                    'n_z': trial.config.n_z,
                                    "n_cutoff": trial.config.n_cutoff,
                                    'n_accepted': trial.z_pool.shape[0],
                                    'tau_true': trial.tau_true, 
                                    'bias': trial.bias, 
                                    'rmse': trial.rmse, 
                                    'rr': trial.rr}
                        all_trial_res.append(trial_res)

    all_trial_res = pd.DataFrame.from_records(all_trial_res)

    if all_trial_res["rand_mdl"].str.contains("rand-complete").any():
        # Compute percentage reduction in RMSE from complete randomization for IGR
        all_trial_res["perc_cr_rmse"] = (
            all_trial_res.groupby(["data_rep", "run_seed", "net_mdl", "po_mdl", "n_z", "n_cutoff"])
            .apply(lambda x: perc_of_cr(x, "rmse"))
            .reset_index(level=(0, 1, 2, 3, 4, 5), drop=True)
            .sort_index()
        )
        # Compute percentage reduction in bias from complete randomization for IGR
        all_trial_res["perc_cr_bias"] = (
            all_trial_res.groupby(["data_rep", "run_seed", "net_mdl", "po_mdl", "n_z", "n_cutoff"])
            .apply(lambda x: perc_of_cr(x, "bias"))
            .reset_index(level=(0, 1, 2, 3, 4, 5), drop=True)
            .sort_index()
        )

        all_trial_res_agg = all_trial_res.groupby(['rand_mdl', 'net_mdl', 'n_z']).agg(
            mean_tau_true = ('tau_true', 'mean'),
            se_tau_true = ('tau_true', 'sem'),
            mean_bias = ('bias', 'mean'),
            se_bias = ('bias', 'sem'),
            mean_perc_cr_bias = ('perc_cr_bias', 'mean'),
            se_perc_cr_bias = ('perc_cr_bias', 'sem'),
            mean_rmse = ('rmse', 'mean'),
            se_rmse = ('rmse', 'sem'),
            mean_perc_cr_rmse = ('perc_cr_rmse', 'mean'),
            se_perc_cr_rmse = ('perc_cr_rmse', 'sem'),
            mean_rr = ('rr', 'mean'),
            se_rr = ('rr', 'sem')).reset_index()
    else:
        all_trial_res_agg = all_trial_res.groupby(['rand_mdl', 'net_mdl', 'n_z']).agg(
            mean_tau_true = ('tau_true', 'mean'),
            se_tau_true = ('tau_true', 'sem'),
            mean_bias = ('bias', 'mean'),
            se_bias = ('bias', 'sem'),
            mean_rmse = ('rmse', 'mean'),
            se_rmse = ('rmse', 'sem'),
            mean_rr = ('rr', 'mean'),
            se_rr = ('rr', 'sem')).reset_index()
        
    res_dir = Path(args.res_dir) / args.exp_subdir
    if not res_dir.exists():
        res_dir.mkdir(parents=True)

    all_trial_res.to_csv(res_dir / f'{args.exp_subdir}_results.csv', index=False)
    all_trial_res_agg.to_csv(res_dir / f'{args.exp_subdir}_results_rep-agg.csv', index=False)