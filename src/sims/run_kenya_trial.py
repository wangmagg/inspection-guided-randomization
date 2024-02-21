import numpy as np
from itertools import combinations
from pathlib import Path
import pickle

from src.sims.trial import SimulatedTrial, SimulatedTrialConfig

class KenyaTrialConfig(SimulatedTrialConfig):
    """
    Configuration for a simulated Kenya trial with interference
    """
    def __init__(self):
        super().__init__()

        self.add_argument('--exp-dir', type=str, default='kenya')

        self.add_argument('--intxn-mdl-name', type=str, default='power-decay')
        self.add_argument('--gamma', type=float, default=0.5)
        self.add_argument('--net-mdl-name', type=str, default='sb')
        self.add_argument('--wi-p', type=float, default=0.9)
        self.add_argument('--bw-p', type=float, default=0.9)

        self.add_argument('--expo-mdl-name', type=str, default='frac-nbr-expo')
        self.add_argument('--q', type=float, default=0.5)

        self.add_argument('--potential-outcome-mdl-name', type=str, default='kenya-hierarchical')
        self.add_argument('--params-fname', type=str, default='params/kenya/params_sigma_scale-1.0.csv')
        self.add_argument('--settlement-info-fname', type=str, default='params/kenya/settlement_info.csv')
        self.add_argument('--coords-range', type=float, default=0.005)

        self.add_argument('--n-per-schl', type=float, default=40)
        self.add_argument('--tau-size', type=int, default=0.3)
        self.add_argument('--sigma', type=float, default=0.015)

        self.add_argument('--observed-outcome-mdl-name', type=str, default='additive-interference')
        self.add_argument('--delta-size', type=float, default=0.3)
        
        self.add_argument('--rand-mdl-name', type=str, default='complete')
        self.add_argument('--cluster-lvl', type=str, default='school')
        self.add_argument('--n-z', type=int, default=int(1e5))
        self.add_argument('--n-cutoff', type=int, default=500)
        self.add_argument('--fitness-fn-name', type=str, default=None)
        self.add_argument('--fitness-fn-weights', type=float, nargs='+', default=None)
        self.add_argument('--tourn-size', type=int, default=2)
        self.add_argument('--cross-k', type=int, default=2)
        self.add_argument('--cross-rate', type=float, default=0.95)
        self.add_argument('--mut-rate', type=float, default=0.01)
        self.add_argument('--genetic-iters', type=int, default=3)
        
        self.add_argument('--estimator-name', type=str, default='clustered-diff-in-means')
        self.add_argument('--alpha', type=float, default=0.05)

class SimulatedKenyaTrial(SimulatedTrial):
    """
    Simulated Kenya trial with interference

    Args:
        - trial_config: configuration for trial
    """
    def __init__(self, 
                 trial_config: KenyaTrialConfig):
        super().__init__(trial_config)
        
        # Initialize data attributes
        self.beta = None
        self.X_school = None
        self.G = None
        self.A = None
        
        self.config.n = None

        return
    
    def _generate_data(self):
        """
        Generate data for the trial
        """
    
        # Generate potential outcomes and covariates
        y_0, y_1, X, beta = self.potential_outcome_mdl()
        X_school = X.drop(['settlement', 'settlement_id', 'school'], axis=1).groupby('school_id').mean()
        
        # Generate network
        self.spatial_intxn_mdl = self.loader.get_spatial_intxn_mdl()
        self.net_mdl = self.loader.get_net_mdl()
        G, A = self.net_mdl(
            n_clusters=X_school.shape[0],
            cluster_sizes=X.groupby('school_id').size().values,
            intxn_mdl = self.spatial_intxn_mdl,
            cluster_coords=X_school[['coord_lon', 'coord_lat']].values,
            pairwise_dists=self._pairwise_dists(X_school, X_school.shape[0])
        )
        return y_0, y_1, X, beta, X_school, G, A

    def _save_data(self, data):
        """
        Save data to a file
        """
        if not self.data_path.parent.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_data(self):
        """
        Load data from a file
        """
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        return data
    
    def _pairwise_dists(self, 
                        X: np.ndarray, 
                        n: int) -> np.ndarray:
        """
        Compute pairwise distances between units

        Args:
            - X: dataframe of covariates
            - n: number of units
        """
        pairwise_dists = np.zeros((n, n))
        for (i, j) in combinations(range(n), 2):
            i_coords = X.iloc[i][["coord_lat", "coord_lon"]]
            j_coords = X.iloc[j][["coord_lat", "coord_lon"]]
            dist = np.linalg.norm(i_coords - j_coords)
            pairwise_dists[i, j] = dist
            pairwise_dists[j, i] = dist
        return pairwise_dists
    
    def _path(self, top_dir, inner_dirs = None):
        """
        Get path for saving trial data or for saving trial object
        Args:
            - top_dir: top-level directory
            - inner_dirs: inner directories
        """
        potential_outcome_mdl_name = self.loader.get_full_potential_outcome_mdl_name()
        net_mdl_name = self.loader.get_full_net_mdl_name()
        spatial_intxn_mdl_name = self.loader.get_full_spatial_intxn_mdl_name()

        top_dir = Path(top_dir)
        data_params_subdir = Path(self.config.params_fname).stem
        data_outcome_mdl_subdir = f"outcome-{potential_outcome_mdl_name}"
        data_net_mdl_subdir = f"net-{net_mdl_name}_intxn-{spatial_intxn_mdl_name}"
        data_fname = f"{self.config.rep_to_run}.pkl"

        if inner_dirs == None:
            return top_dir / data_params_subdir / data_outcome_mdl_subdir / data_net_mdl_subdir / data_fname
        else:
            return top_dir / data_params_subdir / data_outcome_mdl_subdir / data_net_mdl_subdir / inner_dirs / data_fname

    @property
    def pairwise_dists(self) -> np.ndarray:
        """
        Pairwise distances between schools or settlements
        """
        if self.config.cluster_lvl == 'school':
            return self._pairwise_dists(self.X_school, self.X_school.shape[0])
        else:
            X_settlement = self.X.drop(['school', 'school_id'], axis=1).groupby('settlement_id').mean()
            return self._pairwise_dists(X_settlement, X_settlement.shape[0])
    
    @property
    def data_path(self) -> Path:
        """
        Path for saving trial data
        """
        data_dir = Path(self.config.data_dir) / self.config.exp_dir
        return self._path(data_dir)

    @property
    def pickle_path(self) -> Path:
        """
        Path for saving trial object
        """
        save_dir = Path(self.config.out_dir) / self.config.exp_dir
        save_n_z_subdir = f"n-z-{self.config.n_z}"
        save_rand_mdl_subdir = f"rand-{self.rand_mdl.name}_cluster-{self.config.cluster_lvl}"
        
        return self._path(save_dir, f"{save_n_z_subdir}/{save_rand_mdl_subdir}")
    
    def _set_attributes_from_data(self, data):
        """
        Set additional class attributes after generating data
        """
        self.y_0, self.y_1, self.X, self.beta, self.X_school, self.G, self.A = data    
        if self.config.cluster_lvl == 'school':
            self.config.n = self.X_school.shape[0]
            self.mapping = self.X['school_id'].values
        elif self.config.cluster_lvl == 'settlement':
            self.config.n = len(self.X['settlement'].unique())
            self.mapping = self.X['settlement_id'].values
        elif self.config.cluster_lvl == 'individual':
            self.config.n = self.X.shape[0]

        self.use_cols = self.X.columns[self.X.columns.str.startswith('x')]   
    
    def set_data_from_config(self):
        super().set_data_from_config()

    def set_design_from_config(self):
        self.expo_mdl = self.loader.get_expo_mdl()
        super().set_design_from_config()

    def run_trial(self):
        super().run_trial()

    def analyze_trial(self):
        super().analyze_trial()

    def pickle(self):
        super().pickle()

if __name__ == "__main__":
    config = KenyaTrialConfig().parse_args()
    trial = SimulatedKenyaTrial(config)
    trial.simulate()