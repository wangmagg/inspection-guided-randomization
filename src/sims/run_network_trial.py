import networkx as nx
import numpy as np
from pathlib import Path
import pickle

from src.sims.trial import SimulatedTrial, SimulatedTrialConfig
from src.sims import trial_loader

class NetworkTrialConfig(SimulatedTrialConfig):
    """
    Configuration for a simulated network trial with interference
    """

    def __init__(self):
        super().__init__()

        self.add_argument("--data-subdir", type=str, default="network")
        self.add_argument("--output-subdir", type=str, default="network")
        self.add_argument("--n", type=int, default=500)

        self.add_argument("--net-mdl-name", type=str, default="sb")
        self.add_argument("--n-blocks", type=int, default=5)
        self.add_argument("--wi-p", type=float, default=0.05)
        self.add_argument("--bw-p", type=float, default=0.01)
        self.add_argument("--p-er", type=float, default=0.02)
        self.add_argument("--p-ws", type=float, default=0.1)
        self.add_argument("--k", type=float, default=10)
        self.add_argument("--m", type=float, default=5)

        self.add_argument("--expo-mdl-name", type=str, default="frac-nbr-expo")
        self.add_argument("--q", type=float, default=0.5)

        self.add_argument("--potential-outcome-mdl-name", type=str, default="norm-sum")
        self.add_argument("--tau-size", type=int, default=0.3)
        self.add_argument("--mu", type=float, default=1)
        self.add_argument("--sigma", type=float, default=2)
        self.add_argument("--gamma", type=float, default=1)

        self.add_argument(
            "--observed-outcome-mdl-name", type=str, default="additive-interference"
        )
        self.add_argument("--delta-size", type=float, default=0.3)

        self.add_argument("--rand-mdl-name", type=str, default="complete")
        self.add_argument("--n-z", type=int, default=int(1e5))
        self.add_argument("--n-cutoff", type=int, default=500)
        self.add_argument("--add-all-mirrors", action="store_true")
        self.add_argument("--n-batches", type=int, default=None)
        self.add_argument("--fitness-fn-name", type=str, default=None)
        self.add_argument("--fitness-fn-weights", type=float, nargs="+", default=None)
        self.add_argument("--p-misspec", type=float, default=0)
        self.add_argument("--misspec-seed", type=int, default=42)
        self.add_argument("--tourn-size", type=int, default=2)
        self.add_argument("--cross-k", type=int, default=2)
        self.add_argument("--cross-rate", type=float, default=0.95)
        self.add_argument("--mut-rate", type=float, default=0.01)
        self.add_argument("--genetic-iters", type=int, default=3)
        self.add_argument('--eps', type=float, default=0.05)

        self.add_argument("--estimator-name", type=str, default="diff-in-means")
        self.add_argument("--alpha", type=float, default=0.05)


class SimulatedNetworkTrial(SimulatedTrial):
    """
    Simulated network trial with interference

    Args:
        config: configuration for the trial
    """

    def __init__(self, trial_config: NetworkTrialConfig):
        super().__init__(trial_config)

        self.G = None
        self.A = None

    def _generate_data(self):
        """
        Generate data for the trial
        """

        self.net_mdl = trial_loader.get_net_mdl(self.config)
        if "sb" in self.net_mdl.name:
            G, A = self.net_mdl(n=self.config.n, n_clusters=self.config.n_blocks)
        else:
            G, A = self.net_mdl(self.config.n)
        if "cluster" in self.potential_outcome_mdl.name:
            y_0, y_1, X = self.potential_outcome_mdl(A, G)
        else:
            y_0, y_1, X = self.potential_outcome_mdl(A)

        return y_0, y_1, X, G, A

    def _save_data(self, data):
        """
        Save data to a file
        """
        if not self.data_path.parent.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_path, "wb") as f:
            pickle.dump(data, f)

    def _load_data(self):
        """
        Load data from a file
        """
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        return data

    def _path(self, top_dir, inner_dirs=None):
        """
        Get path for saving trial data or for saving trial object
        Args:
            - top_dir: top-level directory
            - inner_dirs: inner directories
        """
        net_mdl_name = trial_loader.get_full_net_mdl_name(self.config)

        top_dir = Path(top_dir)
        data_net_mdl_subdir = f"net-{net_mdl_name}"

        if inner_dirs == None:
            return top_dir / data_net_mdl_subdir 
        else:
            return top_dir / data_net_mdl_subdir / inner_dirs

    @property
    def network_dists(self):
        return dict(nx.all_pairs_bellman_ford_path_length(self.G))

    @property
    def data_path(self) -> Path:
        """
        Path for saving trial data
        """
        data_po_mdl_subdir = f"po-{self.config.potential_outcome_mdl_name}"
        data_dir = Path(self.config.data_dir) / self.config.data_subdir
                
        return self._path(data_dir, data_po_mdl_subdir) / f"{self.config.rep_to_run}.pkl"

    @property
    def pickle_path(self) -> Path:
        """
        Path for saving trial object
        """
        save_dir = Path(self.config.out_dir) / self.config.output_subdir

        save_po_mdl_subdir = f"po-{self.config.potential_outcome_mdl_name}"
        save_n_z_subdir = f"n-z-{self.config.n_z}"
        save_rand_mdl_subdir = f"rand-{self.rand_mdl.name}"

        return self._path(save_dir, f"{save_n_z_subdir}/{save_po_mdl_subdir}/{save_rand_mdl_subdir}/data-rep-{self.config.rep_to_run}_run-seed-{self.config.seed}.pkl")

    def _set_attributes_from_data(self, data):
        """
        Set additional class attributes after generating data
        """
        self.y_0, self.y_1, self.X, self.G, self.A = data

        # if X is a 1D array, expand to 2D
        if self.X.ndim == 1:
            self.X = np.expand_dims(self.X, axis=1)

        self.use_cols = None
        self.arm_compare_pairs = None
        self.mapping = None

    def set_data_from_config(self):
        super().set_data_from_config()

    def set_design_from_config(self):
        self.expo_mdl = trial_loader.get_expo_mdl(self.config)
        super().set_design_from_config()

    def run_trial(self):
        super().run_trial()

    def analyze_trial(self):
        super().analyze_trial()

    def pickle(self):
        super().pickle()


if __name__ == "__main__":
    config = NetworkTrialConfig().parse_args()
    trial = SimulatedNetworkTrial(config)
    trial.simulate()
