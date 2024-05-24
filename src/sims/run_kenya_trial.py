from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from src.sims.trial import SimulatedTrial, SimulatedTrialConfig
from src.sims import trial_loader


class KenyaTrialConfig(SimulatedTrialConfig):
    """
    Configuration for a simulated Kenya trial with interference
    """

    def __init__(self):
        super().__init__()

        self.add_argument("--data-subdir", type=str, default="kenya")
        self.add_argument("--output-subdir", type=str, default="kenya")

        self.add_argument("--intxn-mdl-name", type=str, default="power-decay")
        self.add_argument("--gamma", type=float, default=0.5)
        self.add_argument("--net-mdl-name", type=str, default="sb")
        self.add_argument("--wi-p", type=float, default=0.9)
        self.add_argument("--bw-p", type=float, default=0.9)
        self.add_argument("--p-same-in", type=float, default=0.9)
        self.add_argument("--p-diff-in-same-out", type=float, default=0.9)
        self.add_argument("--p-diff-in-diff-out", type=float, default=0.7)

        self.add_argument("--expo-mdl-name", type=str, default="frac-nbr-expo")
        self.add_argument("--q", type=float, default=0.25)

        self.add_argument(
            "--potential-outcome-mdl-name", type=str, default="kenya-hierarchical"
        )
        self.add_argument(
            "--param-fname",
            type=str,
            default="params/ki-mu-neg_ko-da-pos_sd-sm.csv",
        )
        self.add_argument("--beta", type=float, nargs="+", default=[1, 1])
        self.add_argument("--sigma-sis-scale", type=float, default=0.2)
        self.add_argument("--sigma-iis-scale", type=float, default=0.1)
        self.add_argument("--tau-size", type=int, default=0.3)
        self.add_argument("--sigma", type=float, default=0.015)

        self.add_argument(
            "--observed-outcome-mdl-name", type=str, default="additive-interference"
        )
        self.add_argument("--delta-size", type=float, default=0.3)

        self.add_argument("--rand-mdl-name", type=str, default="complete")
        self.add_argument("--cluster-lvl", type=str, default="school")
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
        self.add_argument("--eps", type=float, default=0.05)

        self.add_argument(
            "--estimator-name", type=str, default="clustered-diff-in-means"
        )
        self.add_argument("--alpha", type=float, default=0.05)


class SimulatedKenyaTrial(SimulatedTrial):
    """
    Simulated Kenya trial with interference

    Args:
        - trial_config: configuration for trial
    """

    def __init__(self, trial_config: KenyaTrialConfig):
        super().__init__(trial_config)

        # Initialize data attributes
        self.beta = None
        self.X_school = None
        self.inner_to_outer_mapping = None

        self.G = None
        self.A = None

        self.config.n = None
        self.arm_compare_pairs = np.array([0, 1])

        self.param = pd.read_csv(self.config.param_fname)

        return
    
    def _sample_coords(self):
        """
        Sample school coordinates

        Returns:
            dataframe of school coordinates
        """
        rng = np.random.default_rng(self.config.seed)

        all_sch_coords = []

        # Sample coordinates of schools for each settlement
        for i, (settlement, data) in enumerate(self.param.iterrows()):
            n_sch = int(data["n_sch"])
            sch_coords = {}

            # Define settlement bounds for school coordinates
            set_coords_lon_bounds = (
                data["coord_lon"] + np.array([-1, 1]) * data["coords_range"]
            )
            set_coords_lat_bounds = (
                data["coord_lat"] + np.array([-1, 1]) * data["coords_range"]
            )

            # Sample school coordinates in a grid without replacement
            sch_coords["coord_lon"] = rng.choice(
                np.linspace(set_coords_lon_bounds[0], set_coords_lon_bounds[1], 200),
                size=n_sch,
                replace=False,
            )
            sch_coords["coord_lat"] = rng.choice(
                np.linspace(set_coords_lat_bounds[0], set_coords_lat_bounds[1], 200),
                size=n_sch,
                replace=False,
            )

            # Add settlement id and within-settlement school id
            sch_coords["settlement"] = settlement
            sch_coords["settlement_id"] = i
            sch_coords["school"] = range(n_sch)
            sch_coords_df = pd.DataFrame.from_dict(sch_coords)

            all_sch_coords.append(sch_coords_df)

        all_sch_coords = pd.concat(all_sch_coords)

        # Add global school id
        all_sch_coords["school_id"] = np.arange(all_sch_coords.shape[0])

        return all_sch_coords

    def _generate_data(self):
        """
        Generate data for the trial
        """

        # Sample school coordinates
        sch_coords = self._sample_coords()
        n_sch = self.param['n_sch'].sum()
        sch_sizes = None
        for _, data in self.param.iterrows():
            sch_sizes_in_set = np.repeat(data['n_per_sch'], data['n_sch'])
            if sch_sizes is None:
                sch_sizes = sch_sizes_in_set
            else:
                sch_sizes = np.concatenate((sch_sizes, sch_sizes_in_set))
        n_set = self.param.shape[0]
        inner_to_outer_mapping = np.repeat(np.arange(n_set), self.param['n_sch'])

        # Generate network
        self.spatial_intxn_mdl = trial_loader.get_spatial_intxn_mdl(self.config)
        self.net_mdl = trial_loader.get_net_mdl(self.config, inner_to_outer_mapping)
        G, A = self.net_mdl(
            n_clusters=n_sch,
            cluster_sizes=sch_sizes,
            intxn_mdl=self.spatial_intxn_mdl,
            pairwise_dists=self._pairwise_dists(sch_coords, n_sch),
        )

        # Generate potential outcomes and covariates
        if self.config.potential_outcome_mdl_name == "kenya-hierarchical":
            y_0, y_1, X = self.potential_outcome_mdl()
        else:
            y_0, y_1, X = self.potential_outcome_mdl(A)
        X_school = (
            X.drop(["settlement", "settlement_id", "school"], axis=1)
            .groupby("school_id")
            .mean()
        )
        
        return y_0, y_1, X, X_school, G, A, sch_coords, inner_to_outer_mapping

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

    def _pairwise_dists(self, X: np.ndarray, n: int) -> np.ndarray:
        """
        Compute pairwise distances between units

        Args:
            - X: dataframe of covariates
            - n: number of units
        """
        pairwise_dists = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            i_coords = X.iloc[i][["coord_lat", "coord_lon"]]
            j_coords = X.iloc[j][["coord_lat", "coord_lon"]]
            dist = np.linalg.norm(i_coords - j_coords)
            pairwise_dists[i, j] = dist
            pairwise_dists[j, i] = dist
        return pairwise_dists

    def _path(self, top_dir, inner_dirs=None):
        """
        Get path for saving trial data or for saving trial object
        Args:
            - top_dir: top-level directory
            - inner_dirs: inner directories
        """
        net_mdl_name = trial_loader.get_full_net_mdl_name(self.config)
        spatial_intxn_mdl_name = trial_loader.get_full_spatial_intxn_mdl_name(
            self.config
        )
        po_mdl_name = self.config.potential_outcome_mdl_name

        top_dir = Path(top_dir)
        data_params_subdir = Path(self.config.param_fname).stem
        data_po_mdl_subdir = po_mdl_name
        data_net_mdl_subdir = f"net-{net_mdl_name}_intxn-{spatial_intxn_mdl_name}"

        if inner_dirs == None:
            return top_dir / data_params_subdir / data_net_mdl_subdir / data_po_mdl_subdir 
        else:
            return (
                top_dir
                / data_params_subdir
                / data_net_mdl_subdir
                / data_po_mdl_subdir
                / inner_dirs
            )

    @property
    def pairwise_dists(self) -> np.ndarray:
        """
        Pairwise distances between schools or settlements
        """
        if self.config.cluster_lvl == "school":
            return self._pairwise_dists(self.sch_coords, self.sch_coords.shape[0])
        else:
            set_coords = (
                self.sch_coords.drop(["school", "school_id"], axis=1)
                .groupby("settlement_id")
                .mean()
            )
            return self._pairwise_dists(set_coords, set_coords.shape[0])

    @property
    def data_path(self) -> Path:
        """
        Path for saving trial data
        """
        data_dir = Path(self.config.data_dir) / self.config.data_subdir
        return self._path(data_dir) / f"{self.config.rep_to_run}.pkl"

    @property
    def pickle_path(self) -> Path:
        """
        Path for saving trial object
        """
        save_dir = Path(self.config.out_dir) / self.config.output_subdir
        save_n_z_subdir = f"n-z-{self.config.n_z}_n-cutoff-{self.config.n_cutoff}"
        save_rand_mdl_subdir = (
            f"rand-{self.rand_mdl.name}_cluster-{self.config.cluster_lvl}"
        )

        return self._path(save_dir, f"{save_n_z_subdir}/{save_rand_mdl_subdir}/data-rep-{self.config.rep_to_run}_run-seed-{self.config.seed}.pkl")

    def _set_attributes_from_data(self, data):
        """
        Set additional class attributes after generating data
        """
        (
            self.y_0,
            self.y_1,
            self.X,
            self.X_school,
            self.G,
            self.A,
            self.sch_coords,
            self.inner_to_outer_mapping,
        ) = data
        if self.config.cluster_lvl == "school":
            self.config.n = self.X_school.shape[0]
            self.mapping = self.X["school_id"].values
        elif self.config.cluster_lvl == "settlement":
            self.config.n = len(self.X["settlement"].unique())
            self.mapping = self.X["settlement_id"].values
        elif self.config.cluster_lvl == "individual":
            self.config.n = self.X.shape[0]

        drop_cols = [
            "settlement",
            "settlement_id",
            "school",
            "school_id"
        ]
        self.use_cols = self.X.columns[~self.X.columns.isin(drop_cols)]

    def set_design_from_config(self):
        self.expo_mdl = trial_loader.get_expo_mdl(self.config)
        super().set_design_from_config()


if __name__ == "__main__":
    config = KenyaTrialConfig().parse_args()
    trial = SimulatedKenyaTrial(config)
    trial.simulate()
