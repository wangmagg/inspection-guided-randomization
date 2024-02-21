from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from src.sims.trial import SimulatedTrial, SimulatedTrialConfig
from typing import List, Optional

class MultiArmTrialConfig(SimulatedTrialConfig):
    """
    Configuration for a simulated multi-arm trial
    """
    def __init__(self):
        super().__init__()

        self.add_argument('--exp-dir', type=str, default='mult-arm')
        
        self.add_argument('--potential-outcome-mdl-name', type=str, default='classroom')
        self.add_argument('--n-per-arm', type=int, default=20)
        self.add_argument('--sigma', type=float, default=1)

        self.add_argument('--observed-outcome-mdl-name', type=str, default='consistent')
        
        self.add_argument('--rand-mdl-name', type=str, default='complete')
        self.add_argument('--n-z', type=int, default=int(1e5))
        self.add_argument('--n-cutoff', type=int, default=500)
        self.add_argument('--fitness-fn-name', type=str, default=None)
        self.add_argument('--fitness-fn-weights', type=float, nargs='+', default=None)
        self.add_argument('--name-covar-to-weight', type=str, default='male')

        self.add_argument('--tourn-size', type=int, default=2)
        self.add_argument('--cross-k', type=int, default=2)
        self.add_argument('--cross-rate', type=float, default=0.95)
        self.add_argument('--mut-rate', type=float, default=0.01)
        self.add_argument('--genetic-iters', type=int, default=3)

        self.add_argument('--qb-dir', type=str, default='data/mult-arm/qb')
        self.add_argument('--min-block-factor', type=int, default=2)
        
        self.add_argument('--estimator-name', type=str, default='diff-in-means-mult-arm')
        self.add_argument('--alpha', type=float, default=0.05)


class SimulatedMultiArmTrial(SimulatedTrial):
    """
    Multi-arm trial

    Args:
        - trial_config: configuration for trial 
    """
    def __init__(self, trial_config: MultiArmTrialConfig):
        super().__init__(trial_config)
        self.config.n = config.n_arms * config.n_per_arm 
        self.arm_pairs = None
        
        # Set treatment effects for each arm
        if self.config.n_arms == 3:
            self.config.tau_sizes = [0, 0.3]
        if self.config.n_arms == 4:
            self.config.tau_sizes = [0, 0.3, 0.6]
        else:
            n_small = self.config.n_arms // 4
            n_mod = self.config.n_arms // 4
            n_large = self.config.n_arms // 4
            n_0 = self.config.n_arms - n_small - n_mod - n_large - 1
            self.config.tau_sizes = [0] * n_0 + [0.1] * n_small + [0.3] * n_mod + [0.6] * n_large
    
    def _generate_data(self):
        """
        Generate data (covariates and potential outcomes) for trial
        """
        return self.potential_outcome_mdl()

    def _save_data(self, 
                   data: tuple):
        """
        Save data to file
        """
        if not self.data_path.parent.exists():
            self.data_path.parent.mkdir(parents=True)

        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)

        # Save covariates to CSV to use in threshold blocking
        _, _, X = data
        X_data_path = self.data_path.with_suffix('.csv')
        pd.DataFrame(X).to_csv(X_data_path, index=False)

    def _load_data(self):
        """
        Load data from file
        """
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        return data
    
    def _path(self, 
              top_dir: str, 
              inner_dirs: Optional[List[str]]=None):
        """
        Get path for saving trial data or for saving trial object
        Args:
            - top_dir: top-level directory
            - inner_dirs: inner directories
        """
        top_dir = Path(top_dir)
        save_arms_subdir = f"arms-{self.config.n_arms}"
        save_n_per_arm_subdir = f"n-per-arm-{self.config.n_per_arm}"
        save_fname = f"{self.config.rep_to_run}.pkl"

        if inner_dirs is None:
            return top_dir / save_arms_subdir / save_n_per_arm_subdir / save_fname
        else:
            return top_dir / save_arms_subdir / save_n_per_arm_subdir / inner_dirs / save_fname
    
    @property
    def covar_to_weight(self):
        """
        Index of covariate to weight in fitness function
        Only used in weighted SumMaxSMD
        """
        return np.where(self.X.columns == self.config.name_covar_to_weight)[0][0]
    
    @property
    def data_path(self):
        """
        Path for saving trial data
        """
        data_dir = Path(self.config.data_dir) / self.config.exp_dir
        return self._path(data_dir)
    
    @property
    def pickle_path(self):
        """
        Path for saving trial object 
        """
        save_dir = Path(self.config.out_dir) / self.config.exp_dir
        save_n_z_subdir = f"n-z-{self.config.n_z}"
        save_rand_mdl_subdir = f"rand-{self.rand_mdl.name}"
        return self._path(save_dir, f"{save_n_z_subdir}/{save_rand_mdl_subdir}")
    
    def _set_attributes_from_data(self, data):
        """
        Set additional class attributes after generating data
        """
        self.y_0, self.y_1, self.X = data
    
    def set_data_from_config(self):
        super().set_data_from_config()
        
    def set_design_from_config(self):
        super().set_design_from_config()

    def run_trial(self):
        super().run_trial()

    def analyze_trial(self):
        super().analyze_trial()

    def pickle(self):
        super().pickle()

if __name__ == "__main__":
    config = MultiArmTrialConfig().parse_args()
    trial = SimulatedMultiArmTrial(config)
    trial.simulate()
  