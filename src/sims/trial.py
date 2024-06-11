import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from joblib import Parallel, delayed

from src.models.exposure_models import *
from src.models.outcome_models import *
from src.models.network_models import *

from src.design.fitness_functions import *
from src.design.genetic_algorithms import *
from src.design.randomization_designs import *

from src.analysis.estimators import *
from src.sims import trial_loader


class SimulatedTrialConfig(ArgumentParser):
    """
    Configuration for a simulated trial
    """

    def __init__(self):
        super().__init__()
        self.add_argument("--seed", type=int, default=42)

        self.add_argument(
            "--make-data", action="store_true", help="Flag for whether to generate data"
        )
        self.add_argument(
            "--run-trial", action="store_true", help="Flag for whether to run trial"
        )
        self.add_argument(
            "--analyze-trial",
            action="store_true",
            help="Flag for whether to analyze trial",
        )
        self.add_argument(
            "--n-arms", type=int, default=2, help="Number of arms in the trial"
        )
        self.add_argument(
            "--n-data-reps",
            type=int,
            default=4,
            help="Number of repeated data samples to generate",
        )
        self.add_argument(
            "--rep-to-run",
            type=int,
            default=0,
            help="Index of data sample to run trial on",
        )

        self.add_argument(
            "--data-dir", type=str, default="data", help="Directory to save/load data"
        )
        self.add_argument(
            "--out-dir", type=str, default="output", help="Directory to save output"
        )


class SimulatedTrial(ABC):
    """
    Abstract class for a simulated trial
    """

    def __init__(self, config: Namespace):

        # Set configuration
        self.config = config

        # Set data attributes
        self.potential_outcome_mdl = None
        self.y_0 = None
        self.y_1 = None
        self.X = None

        # Set design attributes
        self.rand_mdl = None
        self.z_pool = None
        self.chosen_idx = None
        self.mapping = None
        self.use_cols = None
        self.arm_compare_pairs = None

        # Set analysis attributes
        self.observed_outcome_mdl = None
        self.y_obs_pool = None
        self.scores = None
        self.fn_scores = None

        self.tau_hat = None
        self.tau_hat_pool = None
        self.tau_true = None
        self.bias = None
        self.rmse = None
        self.rr = None

        return

    @abstractmethod
    def _generate_data(self):
        """
        Generate data for the trial
        """
        pass

    @abstractmethod
    def _save_data(self):
        """
        Save data to a file
        """
        pass

    @abstractmethod
    def _load_data(self):
        """
        Load data from file
        """
        pass

    @abstractmethod
    def _set_attributes_from_data(self, data):
        """
        Set additional class attributes after generating data
        """
        pass

    @property
    def pickle_path(self):
        """
        Path for saving trial data/results
        """
        pass

    @property
    def X_fit(self) -> Union[pd.DataFrame, np.ndarray]:
        """
        Subset of covariates to use as input to fitness functions
        """
        if self.use_cols is not None:
            return self.X[self.use_cols]
        return self.X

    def set_data_from_config(self):
        """
        Generate and save data or load data (based on configuration)
        """
        if self.config.make_data:
            # Generate and save data
            self.potential_outcome_mdl = trial_loader.get_potential_outcome_mdl(
                self.config
            )
            data = self._generate_data()
            self._save_data(data)
        else:
            # Load data and set data attributes in trial class
            data = self._load_data()
            self._set_attributes_from_data(data)

    def set_design_from_config(self):
        """
        Set randomization design based on configuration
        """
        self.rand_mdl = trial_loader.get_design(self)

    def run_trial(self):
        """
        Run the trial
        """
        # Sample accepted allocations and official chosen allocation
        self.z_pool, self.chosen_idx = self.rand_mdl()
        if hasattr(self.rand_mdl, "scores"):
            self.scores = self.rand_mdl.scores
        if hasattr(self.rand_mdl, "fitness_fn") and hasattr(
            self.rand_mdl.fitness_fn, "all_fn_scores"
        ):
            self.fn_scores = self.rand_mdl.fitness_fn.all_fn_scores

        # Set mapping attribute if using graph randomization
        if self.rand_mdl.name == "graph":
            self.mapping = self.rand_mdl.mapping

        # Generate observed outcomes for all allocations in accepted pool
        self.observed_outcome_mdl = trial_loader.get_observed_outcome_mdl(self)
        self.y_obs_pool = self.observed_outcome_mdl(self.z_pool, self.y_0, self.y_1)

    def analyze_trial(self):
        """
        Do estimation and inference on trial
        """
        estimator, p_val_fn = trial_loader.get_estimator(self)

        print("Getting treatment effect estimates")
        tau_hat_pool = Parallel(n_jobs=4, max_nbytes=int(1e6))(
            delayed(estimator)(z, y_obs) for z, y_obs in tqdm(
                zip(self.z_pool, self.y_obs_pool), total=self.z_pool.shape[0]
            )
        )

        print("Getting p-values")
        pvals = Parallel(n_jobs=4, max_nbytes=int(1e6))(
            delayed(p_val_fn)(z_pool=self.z_pool, y_obs_pool=self.y_obs_pool, idx=idx)
            for idx in tqdm(range(self.z_pool.shape[0]))
        )
        
        self.tau_hat = tau_hat_pool[self.chosen_idx]
        self.tau_hat_pool = tau_hat_pool

        # Calculate bias, rmse, and rejection rate across allocations in the pool
        if np.isscalar(self.tau_hat):
            self.tau_true = (self.y_1 - self.y_0).mean()
            self.bias = np.mean(tau_hat_pool) - self.tau_true
            self.rel_bias = self.bias / self.tau_true
            self.rmse = np.sqrt(np.mean((self.tau_hat_pool - self.tau_true) ** 2))
            self.rr = np.mean(np.array(pvals) <= self.config.alpha)

        else:
            if self.arm_compare_pairs is not None and self.arm_compare_pairs.max() == (
                2 * self.y_1.shape[0] - 1
            ):
                y_0_tiled = np.tile(self.y_0, (self.y_1.shape[0], 1))
                y = np.stack([y_0_tiled, self.y_1], axis=1)
                y = y.reshape(-1, y.shape[-1])
                self.tau_true = np.array(
                    [
                        (y[pair[1]] - y[pair[0]]).mean()
                        for pair in self.arm_compare_pairs
                    ]
                )
            elif self.arm_compare_pairs is not None:
                y = np.vstack([self.y_0, self.y_1])
                self.tau_true = np.array(
                    [
                        (y[pair[1]] - y[pair[0]]).mean()
                        for pair in self.arm_compare_pairs
                    ]
                )
            else:
                self.tau_true = (self.y_1 - self.y_0).mean(axis=1)
            self.bias = np.mean(tau_hat_pool, axis=0) - self.tau_true
            self.rel_bias = np.divide(
                self.bias, self.tau_true, where=self.tau_true != 0
            )
            self.rmse = np.sqrt(
                np.mean((self.tau_hat_pool - self.tau_true) ** 2, axis=0)
            )
            self.rr = np.mean(np.array(pvals) <= self.config.alpha, axis=0)

    def results_summary(self):
        """
        Print summary of trial results
        """
        if np.isscalar(self.tau_hat):
            tau_true_fmt = f"{self.tau_true:.4f}"
            tau_hat_fmt = f"{self.tau_hat:.4f}"
            bias_fmt = f"{self.bias:.4f}"
            rel_bias_fmt = f"{self.rel_bias:.4f}"
            rmse_fmt = f"{self.rmse:.4f}"
            rr_fmt = f"{self.rr:.4f}"
        else:
            tau_true_fmt = np.array2string(self.tau_true, precision=4, separator=", ")
            tau_hat_fmt = np.array2string(self.tau_hat, precision=4, separator=", ")
            bias_fmt = np.array2string(self.bias, precision=4, separator=", ")
            rel_bias_fmt = np.array2string(self.rel_bias, precision=4, separator=", ")
            rmse_fmt = np.array2string(self.rmse, precision=4, separator=", ")
            rr_fmt = np.array2string(self.rr, precision=4, separator=", ")

        print(f"True treatment effect: {tau_true_fmt}")
        print(f"Estimated treatment effect: {tau_hat_fmt}")
        print(f"Bias: {bias_fmt}")
        print(f"Relative bias: {rel_bias_fmt}")
        print(f"MSE: {rmse_fmt}")
        print(f"Rejection rate: {rr_fmt}")

    def pickle(self):
        """
        Save trial data/results to a pickle file, excluding large objects
        """
        full_attr_dict = self.__dict__.copy()

        attr_to_save = [
            "config",
            "arm_compare_pairs",
            "rr",
            "bias",
            "rmse",
            "z_pool",
            "mapping",
            "scores",
            "fn_scores",
            "tau_hat_pool",
            "y_obs_pool",
            "tau_true",
        ]
        attr_dict = {attr: getattr(self, attr, None) for attr in attr_to_save}

        if not self.pickle_path.parent.exists():
            self.pickle_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to {self.pickle_path}")
        with open(self.pickle_path, "wb") as f:
            self.__dict__ = attr_dict
            pickle.dump(self, f)

        self.__dict__ = full_attr_dict

    def simulate(self):
        """
        Simulate the trial
        """

        # Generate data for the trial and save data
        if self.config.make_data:
            # Sample multiple data replicates
            for data_rep in range(self.config.n_data_reps):
                self.config.rep_to_run = data_rep
                self.config.seed += 1
                self.set_data_from_config()

        # Run trial
        if self.config.run_trial:
            # Set data attributes from config
            self.set_data_from_config()

            # Set randomization design from config
            self.set_design_from_config()

            if self.pickle_path.exists():
                print(f"{self.pickle_path} already exists. Skipping...")
            else:
                print(f"Running trial {self.pickle_path}")
                self.run_trial()

                # Analyze trial and save results
                if self.config.analyze_trial:
                    self.analyze_trial()
                    self.results_summary()
                    self.pickle()
                else:
                    self.pickle()
