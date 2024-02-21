import pickle
from pathlib import Path
import pandas as pd
import argparse

from src.sims.trial import *
from src.sims.run_kenya_trial import *

if __name__ == "__main__":
    """
    Collect results from simulated Kenya trials across different configurations
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--exp-subdir', type=str, default='kenya')
    parser.add_argument('--res-dir', type=str, default='res')
    args = parser.parse_args()

    res_dir = Path(args.output_dir) / args.exp_subdir
    all_trial_res = []
    for param_subdir in res_dir.glob('params_*'):
        for outcome_mdl_subdir in param_subdir.glob('outcome-*'):
            for net_mdl_subdir in outcome_mdl_subdir.glob('net-*'):
                for n_z_subdir in net_mdl_subdir.glob('n-z-*'):
                    for rand_mdl_subdir in n_z_subdir.glob('rand-*'):
                        for trial_fname in rand_mdl_subdir.glob('*.pkl'):
                            print(trial_fname)
                            with open(trial_fname, 'rb') as f:
                                trial = pickle.load(f)
                            trial_res = {'params' : param_subdir.name,
                                        'rand_mdl': rand_mdl_subdir.name,
                                        'outcome_mdl': outcome_mdl_subdir.name,
                                        'net_mdl': net_mdl_subdir.name,
                                        'n_z': n_z_subdir.name.split('-')[-1],
                                        'data_rep': trial_fname.stem.split('_')[0],
                                        'tau_true': trial.tau_true, 
                                        'bias': trial.bias, 
                                        'rmse': trial.rmse, 
                                        'rr': trial.rr}
                            all_trial_res.append(trial_res)

    all_trial_res = pd.DataFrame.from_records(all_trial_res)
    all_trial_res_agg = all_trial_res.groupby(['params', 'rand_mdl', 'outcome_mdl', 'net_mdl', 'n_z']).agg(
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