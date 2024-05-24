import numpy as np
import pandas as pd
import networkx as nx

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional


class PotentialOutcomeDGP(ABC):
    """
    Abstract class for covariate and potential outcome data generating processes.
    """

    def __init__(self):
        self.name = None
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ClassroomPODGP(PotentialOutcomeDGP):
    """
    Data generating process for simulated classroom experiment

    Args:
        - n_students: number of students
        - sigma: standard deviation of noise
        - n_arms: number of treatment arms
        - tau_sizes: sizes of treatment effects
        - seed: random seed
    """

    def __init__(
        self,
        n_students: int,
        sigma: float,
        n_arms: int,
        tau_sizes: Union[List[float], np.ndarray[float]],
        seed,
    ):
        self.name = "classroom"
        self.n_students = n_students
        self.sigma = sigma
        self.n_arms = n_arms
        self.tau_sizes = tau_sizes
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Simulate covariates and potential outcomes

        Returns:
            - y_0: potential outcomes for control
            - y_arms: potential outcomes for treatment arms
            - X: covariates
        """
        male = self.rng.binomial(1, 0.7, self.n_students)

        age = self.rng.uniform(19, 25, self.n_students)
        age_z = (age - np.mean(age)) / np.std(age)

        major = self.rng.choice(3, self.n_students, p=[0.5, 0.3, 0.2])

        ability = (
            age_z
            - (major == 1)
            - male
            + self.rng.normal(0, self.sigma, self.n_students)
        )
        confidence = (
            male + (major == 2) + self.rng.normal(0, self.sigma, self.n_students)
        )

        hw = ability + self.rng.normal(0, self.sigma, self.n_students)

        X = pd.DataFrame.from_dict(
            {"male": male, "age": age, "major": major, "ability": ability, "hw": hw}
        )

        # Simulate potential outcomes for control
        y_0 = ability + confidence + self.rng.normal(0, self.sigma, self.n_students)

        # Simulate potential outcomes for treatment arms
        y_arms = []
        for tau_size in self.tau_sizes:
            y_arm = y_0 + tau_size * np.std(y_0)
            y_arms.append(y_arm)
        y_arms = np.vstack(y_arms)

        return y_0, y_arms, X


class ClassroomFixedMalePODGP(PotentialOutcomeDGP):
    """
    Data generating process for simulated classroom experiment with fixed total
    number of male students (used for simulated group formation experiment)

    Args:
        - n_students: number of students
        - sigma: standard deviation of noise
        - n_arms: number of treatment arms
        - tau_size: size of treatment effect
        - tau_comp_sizes: sizes of treatment effects in different group compositions
        - seed: random seed
    """

    def __init__(
        self,
        n_students: int,
        sigma: float,
        tau_size: float,
        tau_comp_sizes: Union[List[float], np.ndarray[float]],
        seed: int,
    ):
        self.name = "classroom-composition"
        self.n_students = n_students
        self.sigma = sigma
        self.tau_size = tau_size
        self.tau_comp_sizes = tau_comp_sizes
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Simulate covariates and potential outcomes

        Returns:
            - y_0: potential outcomes for control
            - y_1_comps: potential outcomes for treatment arms in different group compositions
            - X: covariates
        """
        male = np.zeros(self.n_students)
        male[: self.n_students // 2] = 1
        self.rng.shuffle(male)

        age = self.rng.uniform(19, 25, self.n_students)
        age_z = (age - np.mean(age)) / np.std(age)

        major = self.rng.choice(3, self.n_students, p=[0.5, 0.3, 0.2])

        ability = (
            age_z
            - (major == 1)
            - male
            + self.rng.normal(0, self.sigma, self.n_students)
        )
        confidence = (
            male + (major == 2) + self.rng.normal(0, self.sigma, self.n_students)
        )

        hw = ability + self.rng.normal(0, self.sigma, self.n_students)

        X = pd.DataFrame.from_dict(
            {"male": male, "age": age, "major": major, "ability": ability, "hw": hw}
        )

        # Simulate potential outcomes for control
        y_0 = ability + confidence + self.rng.normal(0, self.sigma, self.n_students)

        # Simulate potential outcomes for treatment arms, for each group composition
        y_1_comps = []
        for tau_comp_size in self.tau_comp_sizes:
            y_1_comp = y_0 + (self.tau_size + tau_comp_size) * np.std(y_0)
            y_1_comps.append(y_1_comp)
        y_1_comps = np.vstack(y_1_comps)

        return y_0, y_1_comps, X


class KenyaPODGP(PotentialOutcomeDGP):
    """
    Data generating process for simulated Kenyan school experiment

    Args:
        - param: parameters for data simulation
        - tau_size: size of treatment effect
        - sigma: standard deviation of noise
        - seed: random seed
    """

    def __init__(
        self,
        param: pd.DataFrame,
        beta: np.ndarray,
        sigma_sis_scale: float,
        sigma_iis_scale: float,
        tau_size: float,
        sigma: float,
        seed: int,
    ):
        self.name = f"kenya-hierarchical_tau-{tau_size:.2f}"
        self.param = param
        self.beta = beta
        self.sigma_sis_scale = sigma_sis_scale
        self.sigma_iis_scale = sigma_iis_scale
        self.tau_size = tau_size
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def _sample_mus(
        self, mus: np.ndarray, sigmas: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample normally distributed means

        Args:
            - mus: means
            - sigmas: standard deviations
            - rng: random number generator

        Returns:
            - mus: sampled means
        """
        return rng.normal(loc=mus, scale=sigmas)

    def _sample_nested_covariates(
        self, mus: np.ndarray, sigmas: np.ndarray, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Sample covariates

        Args:
            - mus: means
            - sigmas: standard deviations
            - n: number of covariates
            - rng: random number generator

        Returns:
            - covariate_df: dataframe of covariates
        """
        covariates = [rng.normal(loc=mus, scale=sigmas) for _ in range(n)]
        covariates = np.vstack(covariates)
        covariate_df = pd.DataFrame(covariates)

        return covariate_df

    def sample_individual_covariates(self) -> pd.DataFrame:
        """
        Sample individual-level covariates in a hierarchical structure

        Returns:
            dataframe of individual-level covariates
        """

        all_ind_covariates = []

        # Sample individual-level covariates for each school in each settlement
        set_mus_sd = self.param.filter(like="mu").std(axis=0).values

        cum_sch_cnt = 0
        for i, (settlement, data) in enumerate(self.param.iterrows()):
            set_mus = data[data.index.str.startswith("mu")].astype(float).values
            n_sch = int(data["n_sch"])
            for school in range(n_sch):
                sch_mus = self._sample_mus(
                    mus=set_mus, sigmas=self.sigma_sis_scale * set_mus_sd, rng=self.rng
                )

                # Sample individual-level covariates, centered around individual-level means
                ind_covariates = self._sample_nested_covariates(
                    mus=sch_mus,
                    sigmas=self.sigma_iis_scale * set_mus_sd,
                    n=int(data["n_per_sch"]),
                    rng=self.rng,
                )

                # Add settlement, settlement id, and school id
                ind_covariates["settlement"] = settlement
                ind_covariates["settlement_id"] = i
                ind_covariates["school"] = school
                ind_covariates["school_id"] = school + cum_sch_cnt
                all_ind_covariates.append(ind_covariates)

            cum_sch_cnt += n_sch

        all_ind_covariates = pd.concat(all_ind_covariates).reset_index(drop=True)

        return all_ind_covariates

    def __call__(self):
        """
        Simulate covariates and potential outcomes

        Returns:
            - y_0: potential outcomes for control
            - y_1: potential outcomes for treatment
            - X: covariates
            - beta_shared: coefficients
        """

        # Sample individual-level covariates and school coordinates
        X = self.sample_individual_covariates()
        X_vals = X.drop(
            [
                "settlement",
                "settlement_id",
                "school",
                "school_id"
            ],
            axis=1,
        ).to_numpy()

        # Simulate potential outcomes for control and treatment using linear model
        beta = np.array(self.beta)
        eps = self.rng.normal(0, self.sigma, X_vals.shape[0])
        y_0 = np.matmul(X_vals, beta.transpose()).squeeze() + eps
        y_1 = y_0 + self.tau_size * np.std(y_0)

        return y_0, y_1, X


class KenyaNbrSumPODGP(KenyaPODGP):

    def __init__(
        self,
        param: pd.DataFrame,
        beta: np.ndarray,
        sigma_sis_scale: float,
        sigma_iis_scale: float,
        tau_size: float,
        sigma: float,
        seed: int,
    ):
        super().__init__(
            param, beta, sigma_sis_scale, sigma_iis_scale, tau_size, sigma, seed
        )
        self.name = f"kenya-hierarchical-nbr-sum_tau-{tau_size:.2f}"

    def __call__(self, A):
        """
        Simulate covariates and potential outcomes

        Parameters:
            - A: adjacency matrix

        Returns:
            - y_0: potential outcomes for control
            - y_1: potential outcomes for treatment
            - X: covariates
            - beta_shared: coefficients
        """

        # Sample individual-level covariates and school coordinates
        X = self.sample_individual_covariates()
        X_vals = X.drop(
            ["settlement", "settlement_id", "school", "school_id"],
            axis=1,
        ).to_numpy()
        X_vals_nbr_sum = np.matmul(A, X_vals)
        X_vals_nbr_sum_df = pd.DataFrame(X_vals_nbr_sum)
        X = pd.merge(
            X[["settlement", "settlement_id", "school", "school_id"]],
            X_vals_nbr_sum_df,
            left_index=True,
            right_index=True
        )
        # Simulate potential outcomes for control and treatment using linear model
        beta = np.array(self.beta)
        eps = self.rng.normal(0, self.sigma, X_vals_nbr_sum.shape[0])
        y_0 = np.matmul(X_vals_nbr_sum, beta.transpose()).squeeze() + eps
        y_1 = y_0 + self.tau_size * np.std(y_0)

        return y_0, y_1, X


class ObservedOutcomeDGP(ABC):
    """
    Abstract class for observed outcome data generating processes.
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Consistent(ObservedOutcomeDGP):
    """
    Data generating process for observed outcomes consistent with generated potential outcomes
    """

    def __init__(self):
        self.name = "consistent"

    def __call__(self, z: np.ndarray, y0: np.ndarray, y_arms: np.ndarray) -> np.ndarray:
        """
        Generate observed outcomes

        Args:
            - z: treatment assignment
            - y0: potential outcomes for control
            - y_arms: potential outcomes for treatment arms
        """
        y_obs = (z == 0) * y0
        for i in range(len(y_arms)):
            y_obs += (z == (i + 1)) * y_arms[i]
        return y_obs


class AdditiveComposition(ObservedOutcomeDGP):
    """
    Data generating process for observed outcomes with additive composition effects

    Args:
        - arm_pairs: pairs of groups with control and treatment labels
    """

    def __init__(self, arm_pairs: np.ndarray):
        self.name = "additive-composition"
        self.arm_pairs = arm_pairs

    def __call__(
        self, z: np.ndarray, y0: np.ndarray, y_1_comps: np.ndarray
    ) -> np.ndarray:
        """
        Generate observed outcomes

        Args:
            - z: treatment assignment
            - y0: potential outcomes for control
            - y_1_comps: potential outcomes for treatment arms in different group compositions
        """
        y_obs = np.zeros((z.shape[0], z.shape[1]))

        # Generate observed outcomes for each composition type
        for comp_lbl, (ctrl_grp_lbl, trt_grp_lbl) in enumerate(self.arm_pairs):
            y_obs += (z == ctrl_grp_lbl) * y0 + (z == trt_grp_lbl) * (
                y_1_comps[comp_lbl]
            )
        return y_obs


class NormSumPODGP(PotentialOutcomeDGP):
    def __init__(
        self, n: int, mu: float, sigma: float, gamma: float, tau_size: float, seed: int
    ):
        self.name = f"norm-sum"
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.tau_size = tau_size
        self.rng = np.random.default_rng(seed)

    def __call__(self, A):
        X = self.rng.normal(self.mu, self.sigma, size=self.n)

        mu_y = np.matmul(A, X)
        y_0 = self.rng.normal(mu_y, self.gamma)
        y_1 = y_0 + self.tau_size * np.std(y_0)

        return y_0, y_1, y_0


class NormSumDegCorPODGP(PotentialOutcomeDGP):
    def __init__(
        self, n: int, mu: float, sigma: float, gamma: float, tau_size: float, seed: int
    ):
        self.name = f"norm-sum-deg-cor"
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.tau_size = tau_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, A):
        mu_scaled = self.mu * np.mean(A, axis=1)
        X = self.rng.normal(mu_scaled, self.sigma, size=self.n)

        mu_y = np.matmul(A, X)

        y_0 = self.rng.normal(mu_y, self.gamma)
        y_1 = y_0 + self.tau_size * np.std(y_0)

        return y_0, y_1, y_0

class NormSumDegCorClusterPODGP(PotentialOutcomeDGP):
    def __init__(
        self, n: int, mu: List[float], sigma: float, gamma: float, tau_size: float, seed: int
    ):
        self.name = f"norm-sum-deg-cor-cluster"
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.tau_size = tau_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, A, G):
        communities = nx.community.louvain_communities(G, seed=self.seed)
        mu = np.zeros(self.n)
        for community in communities:
            mu[list(community)] = self.rng.normal(self.mu, 2 * self.sigma)

        mu_scaled = mu * np.mean(A, axis=1)
        X = self.rng.normal(mu_scaled, self.sigma, size=self.n)

        mu_y = np.matmul(A, X)

        y_0 = self.rng.normal(mu_y, self.gamma)
        y_1 = y_0 + self.tau_size * np.std(y_0)

        return y_0, y_1, y_0


class AdditiveInterference(ObservedOutcomeDGP):
    """
    Data generating process for observed outcomes with additive interference effects

    Args:
        - delta_size: size of spillover effect
        - expo_mdl (ExposureModel): instance of exposure model
        - A: adjacency matrix
        - mapping: mapping from individual-level to group-level treatment assignment
    """

    def __init__(
        self,
        delta_size: float,
        expo_mdl: callable,
        A: np.ndarray,
        mapping: Optional[np.ndarray] = None,
    ):
        self.name = f"additive-{delta_size:.2f}"
        self.delta_size = delta_size
        self.expo_mdl = expo_mdl
        self.A = A
        self.mapping = mapping

    def __call__(self, z: np.ndarray, y_0: np.ndarray, y_1: np.ndarray) -> np.ndarray:
        """
        Generate observed outcomes

        Args:
            - z: treatment assignment
            - y_0: potential outcomes for control
            - y_1: potential outcomes for treatment
        """
        if self.mapping is not None:
            if z.ndim == 2:
                z = np.vstack([z_single[self.mapping] for z_single in z])
            else:
                z = z[self.mapping]

        # Generate observed outcomes with additive effect for interference
        y_obs = (
            z * y_1
            + (1 - z) * y_0
            + (1 - z) * self.delta_size * np.std(y_0) * self.expo_mdl(z, self.A)
        )

        return y_obs
