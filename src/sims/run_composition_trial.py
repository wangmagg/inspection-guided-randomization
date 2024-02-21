from pathlib import Path
import pickle
import numpy as np

from src.sims.trial import SimulatedTrial, SimulatedTrialConfig
from typing import List, Optional

class CompositionTrialConfig(SimulatedTrialConfig):
    """
    Configuration for a simulated composition trial
    """
    def __init__(self):
        super().__init__()

        self.add_argument('--exp-dir', type=str, default='composition')
        
        self.add_argument('--potential-outcome-mdl-name', type=str, default='classroom-composition')
        self.add_argument('--p-comps', type=float, nargs='+', default = [0.5, 0.3, 0.7])
        self.add_argument('--tau-size', type=float, default=0.3)
        self.add_argument('--tau-comp-sizes',type=float, nargs='+', default = [0, 0.2, -0.2])
        self.add_argument('--n-per-arm', type=int, default=20)
        self.add_argument('--sigma', type=float, default=1)

        self.add_argument('--observed-outcome-mdl-name', type=str, default='additive-composition')
        
        self.add_argument('--rand-mdl-name', type=str, default='group-formation')
        self.add_argument('--n-z', type=int, default=int(1e5))
        self.add_argument('--n-cutoff', type=int, default=500)
        self.add_argument('--fitness-fn-name', type=str, default=None)
        self.add_argument('--fitness-fn-weights', type=float, nargs='+', default=None)
        self.add_argument('--tourn-size', type=int, default=2)
        self.add_argument('--cross-k', type=int, default=2)
        self.add_argument('--cross-rate', type=float, default=0.95)
        self.add_argument('--mut-rate', type=float, default=0.01)
        self.add_argument('--genetic-iters', type=int, default=3)
        
        self.add_argument('--estimator-name', type=str, default='diff-in-means-mult-arm')
        self.add_argument('--alpha', type=float, default=0.05)

class SimulatedCompositionTrial(SimulatedTrial):
    """
    Composition trial where treatment effects depend on the composition of the group
    and composition is defined by the proportion of individuals in the group with a certain
    salient attribute

    Args:
        - trial_config: configuration for trial
    """
    def __init__(self, trial_config: CompositionTrialConfig):
        super().__init__(trial_config)

        # Initialize mask of individuals with the salient attribute
        self.X_on = None

        # Set number of compositions
        self.n_comps = len(self.config.p_comps)

        # Set number of individuals
        self.config.n = self.config.n_arms * self.config.n_per_arm * self.n_comps  

        # Set pairs of groups with the same composition but different treatment 
        self.arm_pairs = np.arange(self.config.n_arms * self.n_comps).reshape(self.n_comps, self.config.n_arms)
        
        return
    
    def _generate_data(self):
        """
        Generate covariates and potential outcomes for the trial
        """
        return self.potential_outcome_mdl()

    def _save_data(self, data):
        """
        Save data to a file
        """
        if not self.data_path.parent.exists():
            self.data_path.parent.mkdir(parents=True)
        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_data(self):
        """
        Load data from a file
        """
        print(self.data_path)
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def _path(self, 
              top_dir: str, 
              inner_dirs: Optional[List[str]] = None):
        """
        Get path for saving trial data or for saving trial object
        Args:
            - top_dir: top-level directory
            - inner_dirs: inner directories
        """
        top_dir = Path(top_dir) 
        save_p_comps_subdir = f"p-comps-{self.config.p_comps}"
        save_n_per_arm_subdir = f"n-per-arm-{self.config.n_per_arm}"
        save_fname = f"{self.config.rep_to_run}.pkl"
        
        if inner_dirs is None:
            return top_dir / save_p_comps_subdir / save_n_per_arm_subdir / save_fname
        else:
            return top_dir / save_p_comps_subdir / save_n_per_arm_subdir / inner_dirs / save_fname
    
    @property
    def data_path(self):
        """
        Path for saving trial data
        """
        save_dir = Path(self.config.data_dir) / self.config.exp_dir
        return self._path(save_dir)
    
    @property
    def pickle_path(self):
        """
        Path for saving trial object
        """
        save_dir = Path(self.config.out_dir) / self.config.exp_dir
        save_rand_mdl_subdir = f"rand-{self.rand_mdl.name}"
        save_n_z_subdir = f"n-z-{self.config.n_z}"
        
        return self._path(save_dir, f"{save_n_z_subdir}/{save_rand_mdl_subdir}")
    
    def _set_attributes_from_data(self, data):
        """
        Set additional class attributes after generating data
        """

        # Set potential outcomes and covariates
        self.y_0, self.y_1, self.X = data

        # Set mask of individuals with the salient attribute
        self.X_on = (self.X['male'] == 0)

        # Set columns to use in fitness functions
        self.use_cols = self.X.columns[self.X.columns != 'male'] 
        
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
    config = CompositionTrialConfig().parse_args()
    trial = SimulatedCompositionTrial(config)
    trial.simulate()