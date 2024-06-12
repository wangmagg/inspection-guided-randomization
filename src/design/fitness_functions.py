import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from scipy.spatial.distance import mahalanobis
from itertools import combinations
import time
from joblib import Parallel, delayed

from typing import List, Union, Optional

try:
    import cupy as cp
    USE_GPU = True
except ModuleNotFoundError:
    USE_GPU = False

class Fitness(ABC):
    """
    Abstract base class for fitness functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, z_pool, **kwargs):
        pass


class SMD(Fitness):
    """
    Standardized Mean Difference (SMD)

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - weights: weight array (n_arms) if using weighted SMD
        - covar_to_weight: index of the covariate to apply weight to (if using weighted SMD)

    """

    def __init__(
        self,
        X: Union[np.ndarray],
        arm_compare_pairs: Optional[np.ndarray] = None,
        mapping: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        covar_to_weight: Optional[int] = None,
    ):
        if weights is None:
            self.name = f"smd"
        else:
            self.name = f"smd_weighted-{weights}-{covar_to_weight}"

        self.X = X
        self.arm_compare_pairs = arm_compare_pairs
        self.mapping = mapping
        self.weights = weights
        self.covar_to_weight = covar_to_weight

        self.plotting_name = "SMD"

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for SMD that
        creates instance of SMD using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        Returns:
            - instance of SMD
        """
        if hasattr(trial, "covar_to_weight") and trial.covar_to_weight is not None:
            return self(
                trial.X_fit,
                trial.arm_compare_pairs,
                trial.mapping,
                trial.config.fitness_fn_weights,
                trial.covar_to_weight,
            )
        else:
            return self(trial.X_fit, trial.arm_compare_pairs, trial.mapping)

    def weight_matrix(self, n_arms: int) -> np.ndarray:
        """
        Calculate weight matrix for weighted SMD using weight array

        Args:
            - n_arms: number of arms
        Returns:
            Matrix of weights (n_arms x n_arms) to apply to pairwise arm comparisons
        """
        weight_matrix = np.zeros((n_arms, n_arms))
        weight_matrix[0] = self.weights

        for arm_idx in range(1, n_arms):
            # Create next row of weights by renormalizing relative to the weights from previous row
            weights_prev_row = weight_matrix[arm_idx - 1]
            weights_prev_row_renormed = weights_prev_row - weights_prev_row[arm_idx] + 1
            weight_matrix[arm_idx, arm_idx:] = weights_prev_row_renormed[arm_idx:]

        weight_matrix = weight_matrix - weight_matrix.T

        return weight_matrix

    def _single_z(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate SMD for a single treatment assignment

        Args:
            - z: treatment assignment vector
        Returns:
            SMDs for pairwise comparisons of arms for each covariate
        """
        n_arms = int(np.max(z) + 1)

        # calculate mean and standard deviation of each covariate for each arm
        arm_means = np.vstack(
            [np.mean(self.X[z == arm, :], axis=0) for arm in range(n_arms)]
        )
        arm_stds = np.vstack(
            [np.std(self.X[z == arm, :], axis=0) for arm in range(n_arms)]
        )

        if self.arm_compare_pairs is None:
            self.arm_compare_pairs = np.array(list(combinations(range(n_arms), 2)))

        if self.weights is not None:
            weight_matrix = self.weight_matrix(n_arms)

        # calculate pairwise differences in means and standard deviations
        all_smd = np.zeros((self.arm_compare_pairs.shape[0], self.X.shape[1]))
        for i, pair in enumerate(self.arm_compare_pairs):
            mean_diff = arm_means[pair[0], :] - arm_means[pair[1], :]
            std_sum = np.sqrt(arm_stds[pair[0], :] ** 2 + arm_stds[pair[1], :] ** 2)

            if self.covar_to_weight is not None:
                mean_diff[self.covar_to_weight] = (
                    weight_matrix[pair[0], pair[1]] * mean_diff[self.covar_to_weight]
                )

            smd = np.divide(mean_diff, std_sum, out=mean_diff, where=std_sum != 0)
            all_smd[i, :] = smd

        return all_smd

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate SMD for a pool of treatment assignments

        Args:
            - z_pool: pool of treatment assignment vectors
        Returns:
            Array of SMDs for each treatment assignment in the pool
        """
        if self.mapping is not None:
            # map cluster treatment assigment to individual treatment assignment
            z_pool = np.vstack([z[self.mapping] for z in z_pool])
        return np.array([self._single_z(z) for z in z_pool])


class MaxAbsSMD(SMD):
    """
    Maximum absolute SMD across all covariates for each pairwise comparison of arms

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - weights: weight matrix (n_arms x n_arms) if using weighted SMD
        - covar_to_weight: index of the covariate to apply weight to (if using weighted SMD)
    """

    def __init__(
        self,
        X,
        arm_compare_pairs=None,
        mapping=None,
        weights=None,
        covar_to_weight=None,
    ):
        super().__init__(X, arm_compare_pairs, mapping, weights, covar_to_weight)
        self.name = f"max-abs-{self.name}"

        self.plotting_name = "MaxAbsSMD"
        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor that
        creates an instance of MaxAbsSMD using data from trial
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return super().from_trial(trial)

    def _single_z(self, z: np.ndarray) -> np.array:
        """
        Calculate maximum absolute SMD for a single treatment assignment
        Args:
            - z: treatment assignment vector
        Returns:
            Maximum absolute SMD for each covariate across all pairwise comparison of arms
            (n_covariates, )
        """
        smd = super()._single_z(z)
        abs_smd = np.abs(smd)

        return np.max(abs_smd, axis=0)

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate maximum absolute SMD for a pool of treatment assignments
        Args:
            - z_pool: pool of treatment assignment vectors
        Returns:
            Array of maximum absolute SMD for each treatment assignment in the pool
        """
        if self.mapping is not None:
            # map cluster treatment assigment to individual treatment assignment
            z_pool = np.vstack([z[self.mapping] for z in z_pool])
        return np.vstack([self._single_z(z) for z in z_pool])


class SignedMaxAbsSMD(SMD):
    """
    Maximum SMD across all covariates for each pairwise comparison of arms, with sign

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - weights: weight matrix (n_arms x n_arms) if using weighted SMD
        - covar_to_weight: index of the covariate to apply weight to (if using weighted SMD)
    """

    def __init__(
        self,
        X,
        arm_compare_pairs=None,
        mapping=None,
        weights=None,
        covar_to_weight=None,
    ):
        super().__init__(X, arm_compare_pairs, mapping, weights, covar_to_weight)
        self.name = f"signed-max-abs-{self.name}"

        self.plotting_name = "MaxSMD"

        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for SignedMaxAbsSMD that
        creates an instance of SignedMaxAbsSMD using data from trial
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return super().from_trial(trial)

    def _single_z(self, z: np.array) -> np.ndarray:
        """
        Calculate maximum absolute SMD for a single treatment assignment
        Args:
            - z: treatment assignment vector
        Returns:
            Maximum absolute SMD for each covariate across pairwise comparison of arms, with sign
            (n_covariates,)
        """
        smd = super()._single_z(z)
        where_max_abs_smd = np.argmax(np.abs(smd), axis=0)
        signed_max_abs_smd = smd[where_max_abs_smd, np.arange(smd.shape[1])]
        return signed_max_abs_smd

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate maximum absolute SMD for a pool of treatment assignments
        Args:
            - z_pool: pool of treatment assignment vectors
        """
        if self.mapping is not None:
            # map cluster treatment assigment to individual treatment assignment
            z_pool = np.vstack([z[self.mapping] for z in z_pool])
        return np.vstack([self._single_z(z) for z in z_pool])


class SumAbsSMD(SMD):
    """
    Sum of absolute SMD across all covariates for each pairwise comparison of arms
    """

    def __init__(
        self,
        X,
        arm_compare_pairs=None,
        mapping=None,
        weights=None,
        covar_to_weight=None,
    ):
        super().__init__(X, arm_compare_pairs, mapping, weights, covar_to_weight)
        self.name = f"sum-abs-{self.name}"
        self.plotting_name = f"SumAbs{self.plotting_name}"

        return

    @classmethod
    def from_trial(self, trial):
        """
        Create an instance of SumMaxSMD using data from trial
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return super().from_trial(trial)

    def _single_z(self, z: np.ndarray) -> np.float32:
        """
        Calculate sum of SMD for a single treatment assignment
        Args:
            - z: treatment assignment vector
        Returns:
            Sum of SMD for each pairwise comparison of arms
        """
        smd = super()._single_z(z)

        # Take absolute value of all but the weighted covariate
        abs_smd = np.abs(smd)
        if self.covar_to_weight is not None:
            abs_smd[:, self.covar_to_weight] = smd[:, self.covar_to_weight]

        return np.sum(abs_smd)

    def __call__(self, z_pool):
        """
        Calculate sum of absolute SMD for a pool of treatment assignments
        Args:
            - z_pool: pool of treatment assignment vectors
        Returns:
            Array of sum of absolute SMD for each treatment assignment in the pool
        """
        if self.mapping is not None:
            # map cluster treatment assigment to individual treatment assignment
            z_pool = np.vstack([z[self.mapping] for z in z_pool])
        return np.array([self._single_z(z) for z in z_pool])


class SumMaxAbsSMD(MaxAbsSMD):
    """
    Sum of maximum absolute SMD across all covariates for each pairwise comparison of arms

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - weights: weight matrix (n_arms x n_arms) if using weighted SMD
        - covar_to_weight: index of the covariate to apply weight to (if using weighted SMD)
    """

    def __init__(
        self,
        X,
        arm_compare_pairs=None,
        mapping=None,
        weights=None,
        covar_to_weight=None,
    ):
        super().__init__(X, arm_compare_pairs, mapping, weights, covar_to_weight)
        self.name = f"sum-{self.name}"
        self.plotting_name = f"Sum{self.plotting_name}"

    @classmethod
    def from_trial(self, trial):
        """
        Create an instance of SumMaxSMD using data from trial
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return super().from_trial(trial)

    def _single_z(self, z: np.ndarray) -> np.float32:
        """
        Calculate sum of maximum absolute SMD for a single treatment assignment
        Args:
            - z: treatment assignment vector
        Returns:
            Sum of maximum absolute SMD for each pairwise comparison of arms
        """
        max_smd = super()._single_z(z)
        return np.sum(max_smd)

    def __call__(self, z_pool):
        """
        Calculate sum of maximum absolute SMD for a pool of treatment assignments
        Args:
            - z_pool: pool of treatment assignment vectors
        Returns:
            Array of sum of maximum absolute SMD for each treatment assignment in the pool
        """
        if self.mapping is not None:
            # map cluster treatment assigment to individual treatment assignment
            z_pool = np.vstack([z[self.mapping] for z in z_pool])
        return np.array([self._single_z(z) for z in z_pool])


class MaxMahalanobis(Fitness):
    """
    Maximum Mahalanobis distance across all pairwise arm comparisons

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
    """

    def __init__(
        self,
        X: np.ndarray,
        arm_compare_pairs: Optional[np.ndarray] = None,
        mapping: Optional[np.ndarray] = None,
    ):
        self.name = "max-mahalanobis"
        if arm_compare_pairs is not None and np.ndim(arm_compare_pairs) == 1:
            self.plotting_name = "Mahalanobis"
        else:
            self.plotting_name = "MaxMahalanobis"

        self.X = X
        self.arm_compare_pairs = arm_compare_pairs
        self.mapping = mapping

        if USE_GPU:
            X = cp.asarray(X)
            xp = cp
        else:
            xp = np

        # calculate sample covariance matrix
        x_bar = xp.expand_dims(xp.mean(self.X, axis=0), 1)
        N = self.X.shape[0]
        S_hat = 1 / (N - 1) * xp.matmul(self.X.T - x_bar, (self.X.T - x_bar).T)
        s, u = xp.linalg.eigh(S_hat)
        self.S_hat_inv = u @ (1 / s[..., None] * u.T)

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for MaxMahalanobis that
        creates an instance of MaxMahalanobis using data from trial
        """
        return self(trial.X_fit, trial.arm_compare_pairs, trial.mapping)

    def _get_arm_means(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate covariate mean vectors for each arm

        Args:
            - z: treatment assignment vector
        Returns:
            - arm_means: covariate mean vectors for each arm (n_arms x n_covariates)
        """
        n_arms = int(np.max(z) + 1)

        # calculate mean of each covariate for each arm
        arm_means = np.zeros((n_arms, self.X.shape[1]))
        for arm in range(n_arms):
            arm_means[arm, :] = np.mean(self.X[z == arm, :], axis=0)

        return arm_means

    def _single_z(self, z: np.ndarray) -> np.float32:
        """
        Calculate maximum Mahalanobis distance for a single treatment assignment

        Args:
            - z: treatment assignment vector
        Returns:
            Maximum Mahalanobis distance across all pairwise arm comparisons
        """
        arm_means = self._get_arm_means(z)
        if self.arm_compare_pairs is None:
            self.arm_compare_pairs = np.array(
                list(combinations(range(arm_means.shape[0]), 2))
            )
        if np.ndim(self.arm_compare_pairs) == 1:
            self.arm_compare_pairs = np.expand_dims(self.arm_compare_pairs, axis=0)

        dists = [
            mahalanobis(arm_means[p[0], :], arm_means[p[1], :], self.S_hat_inv)
            for p in self.arm_compare_pairs
        ]

        # get maximum distance across all pairwise arm comparisons
        max_dist = np.max(dists)

        return max_dist

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate maximum Mahalanobis distance for a pool of treatment assignments

        Args:
            - z_pool: pool of treatment assignment vectors
        Returns:
            Array of maximum Mahalanobis distance for each treatment assignment in the pool
        """
        if self.mapping is not None:
            z_pool = np.vstack([z[self.mapping] for z in z_pool])

        def _single_z(z, X, arm_compare_pairs):
            n_arms = int(np.max(z) + 1)

            # calculate mean of each covariate for each arm
            arm_means = np.zeros((n_arms, X.shape[1]))
            for arm in range(n_arms):
                arm_means[arm, :] = np.mean(X[z == arm, :], axis=0)

            if arm_compare_pairs is None:
                arm_compare_pairs = np.array(
                    list(combinations(range(arm_means.shape[0]), 2))
                )
            if np.ndim(arm_compare_pairs) == 1:
                arm_compare_pairs = np.expand_dims(arm_compare_pairs, axis=0)

            dists = [
                mahalanobis(arm_means[p[0], :], arm_means[p[1], :], self.S_hat_inv)
                for p in arm_compare_pairs
            ]

            # get maximum distance across all pairwise arm comparisons
            max_dist = np.max(dists)

            return max_dist
        
        time_start = time.time()
        if USE_GPU:
            scores = [_single_z(z, self.X, self.arm_compare_pairs) for z in z_pool]
        else:
            scores = Parallel(n_jobs=2, max_nbytes=int(1e6))(
                delayed(_single_z)(z, self.X, self.arm_compare_pairs) for z in z_pool
            )
        scores = np.array(scores)
        time_end = time.time()
        print(f"max-mahalanobis scores: {time_end - time_start:.2f}s")

        return scores


class WeightedMaxMahalanobis(MaxMahalanobis):
    """
    Maximum Mahalanobis distance across all pairwise arm comparisons,
    with weights on more "important" covariates (hypothesized to be more prognostic of outcome)

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - betas: weights to apply to each covariate
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
    """

    def __init__(self, X, betas, arm_compare_pairs=None, mapping=None):
        super().__init__(X, mapping)
        self.name = f"weighted-{self.name}"
        self.plotting_name = f"Weighted {self.plotting_name}"
        self.betas = betas
        self.arm_compare_pairs = arm_compare_pairs

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for WeightedMaxMahalanobis that
        creates an instance of WeightedMaxMahalanobis using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """

        return self(trial.X_fit, trial.beta, trial.arm_compare_pairs, trial.mapping)

    def _single_z(self, z: np.ndarray) -> np.float32:
        """
        Calculate weighted maximum Mahalanobis distance for a single treatment assignment

        Args:
            - z: treatment assignment vector
        Returns:
            Weighted maximum Mahalanobis distance across all pairwise arm comparisons
        """
        arm_means = self._get_arm_means(z)
        if self.arm_compare_pairs is None:
            self.arm_compare_pairs = np.array(
                list(combinations(range(arm_means.shape[0]), 2))
            )

        dists = [
            mahalanobis(
                arm_means[p[0], :] * self.betas,
                arm_means[p[1], :] * self.betas,
                self.S_hat_inv,
            )
            for p in self.arm_compare_pairs
        ]

        max_dist = np.max(dists)

        return max_dist

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate weighted maximum Mahalanobis distance for a pool of treatment assignments

        Args:
            - z_pool: pool of treatment assignment vectors
        """
        if self.mapping is not None:
            # map cluster treatment assigment to individual treatment assignment
            z_pool = np.vstack([z[self.mapping] for z in z_pool])

        return np.array([self._single_z(z) for z in z_pool])


class FracExposed(Fitness):
    """
    Fraction of control group exposed to treatment through network interference

    Args:
        - expo_mdl (ExposureModel): exposure model instance that calculates exposure to treatment
        - A: adjacency matrix
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
    """

    def __init__(
        self,
        expo_mdl,
        A: np.ndarray,
        mapping: Optional[np.ndarray] = None,
        p_misspec: Optional[float] = 0,
        misspec_seed: Optional[int] = 42,
        mask_misspec: Optional[np.ndarray] = None,
    ):
        if p_misspec == 0:
            self.name = "frac-exposed"
        else:
            self.name = f"frac-exposed-misspec-{p_misspec:.3f}-seed-{misspec_seed}"
        self.plotting_name = "FracCtrlExposed"

        self.expo_mdl = expo_mdl
        self.A = A
        self.mapping = mapping
        self.min = 0
        self.max = 1

        self.p_misspec = p_misspec
        self.mask_misspec = mask_misspec

        self.rng = np.random.default_rng(misspec_seed)

        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for FracExposed that
        creates an instance of FracExposed using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        # if clustered, set misspecification mask such that only edges between clusters are misspecified
        if trial.mapping is not None and trial.config.p_misspec > 0:
            mask_misspec = np.not_equal.outer(trial.mapping, trial.mapping).astype(bool)
        else:
            mask_misspec = None

        return self(
            trial.expo_mdl,
            trial.A,
            trial.mapping,
            trial.config.p_misspec,
            trial.config.misspec_seed,
            mask_misspec,
        )

    @property
    def A_misspec(self):
        # introduce edge misspecification (by adding and removing edges with probability p_misspec)
        A_misspec = self.A.copy()

        # Find the indices of existing edges and non-edges in the upper triangle of the adjacency matrix
        triu_indices = np.triu_indices_from(A_misspec, k=1)
        if self.mask_misspec is None:
            n_edges = np.sum(A_misspec) // 2
            n_misspec = int(self.p_misspec * n_edges)

            where_A_1 = np.where(A_misspec[triu_indices] == 1)[0]
            where_A_0 = np.where(A_misspec[triu_indices] == 0)[0]
        else:
            n_edges = np.sum((A_misspec == 1) & self.mask_misspec) // 2
            n_misspec = int(self.p_misspec * n_edges)

            where_A_1 = np.where(
                (A_misspec[triu_indices] == 1) & self.mask_misspec[triu_indices]
            )[0]
            where_A_0 = np.where(
                (A_misspec[triu_indices] == 0) & self.mask_misspec[triu_indices]
            )[0]

        # Sample the indices for edges to be removed and added
        idx_misspec_1_to_0 = self.rng.choice(where_A_1, n_misspec, replace=False)
        idx_misspec_0_to_1 = self.rng.choice(where_A_0, n_misspec, replace=False)

        # Apply the misspecifications to the adjacency matrix
        r_1_to_0, c_1_to_0 = (
            triu_indices[0][idx_misspec_1_to_0],
            triu_indices[1][idx_misspec_1_to_0],
        )
        r_0_to_1, c_0_to_1 = (
            triu_indices[0][idx_misspec_0_to_1],
            triu_indices[1][idx_misspec_0_to_1],
        )
        A_misspec[r_1_to_0, c_1_to_0] = 0
        A_misspec[r_0_to_1, c_0_to_1] = 1

        # Ensure the matrix is symmetric
        A_misspec = np.maximum(A_misspec, A_misspec.T)

        return A_misspec

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate fraction of control group exposed to treatment

        Args:
            - z_pool: pool of treatment assignment vectors
        Returns:
            Array of exposed fraction for each treatment assignment in the pool
        """
        if self.mapping is not None:
            # map cluster treatment assigment to individual treatment assignment
            z_pool = np.vstack([z[self.mapping] for z in z_pool])

        # get mask of individuals exposed to treatment
        time_start = time.time()
        if self.p_misspec > 0:
            is_exposed = self.expo_mdl(z_pool, self.A_misspec)
        else:
            is_exposed = self.expo_mdl(z_pool, self.A)
        time_end = time.time()
        print(f"frac-expo is_exposed: {time_end - time_start:.2f}s")

        if z_pool.ndim == 1:
            z_pool = np.expand_dims(z_pool, axis=0)
        n_0 = np.sum(1 - z_pool, axis=1)

        time_start = time.time()
        frac_exposed = np.divide(
            np.sum(is_exposed * (1 - z_pool), axis=1), n_0, where=n_0 != 0
        )
        time_end = time.time()
        print(f"frac-expo frac_exposed: {time_end - time_start:.2f}s")

        return frac_exposed


class MinPairwiseEuclideanDist(Fitness):
    """
    Minimum pairwise Euclidean distance between arms (inverted so that higher values are better)

    Args:
        - pairwise_dists: pairwise distance matrix between units of randomization
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
    """

    def __init__(self, pairwise_dists: np.ndarray):
        self.name = "min-pairwise-euclidean-dist"
        self.plotting_name = "InvMinEuclideanDist"
        self.pairwise_dists = pairwise_dists
        self.min = 0
        self.max = np.inf
        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for MinPairwiseEuclideanDist that
        creates an instance of MinPairwiseEuclideanDist using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return self(trial.pairwise_dists)

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate minimum pairwise Euclidean distance between units assigned to different arms
        for a pool of treatment assignments

        Args:
            - z_pool: pool of treatment assignment vectors
        """
        min_dists = np.zeros(z_pool.shape[0])
        for i, z in tqdm(enumerate(z_pool), total=z_pool.shape[0]):
            z_expand = np.expand_dims(z, axis=0)

            # get mask of units assigned to different arms
            Z_diff_mask = np.matmul(z_expand.T, 1 - z_expand) + np.matmul(
                (1 - z_expand).T, z_expand
            )

            # calculate minimum pairwise Euclidean distance between units assigned to different arms
            dists_diff_z = self.pairwise_dists[Z_diff_mask.astype(bool)]

            # take inverse
            min_dists[i] = 1 / np.min(dists_diff_z)

        return min_dists


class LinearCombinationFitness(Fitness, ABC):
    """
    Abstract class for linear combination of two fitness functions

    Args:
        - fitness_fns: list of fitness function instances
        - weights: weights to apply to each fitness function
    """

    def __init__(
        self, fitness_fns: List[Fitness], weights: Union[List[float], np.ndarray]
    ):
        self.name = "lin-comb"
        ff_plotting_names = []
        for fn, weight in zip(fitness_fns, weights):
            self.name += f"_{fn.name}-{weight:.2f}"
            ff_plotting_names.append(f"{weight}" + r"$\cdot$" + f"{fn.plotting_name}")
        self.plotting_name = "+\n".join(ff_plotting_names)

        self.weights = weights
        self.fitness_fns = fitness_fns
        self.all_fn_scores = None

        return

    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate linear combination of two fitness functions for a pool of treatment assignments

        Args:
            - z_pool: pool of treatment assignment vectors
        """
        scores = np.zeros(z_pool.shape[0])
        all_fn_scores = np.zeros((len(self.fitness_fns), z_pool.shape[0]))
        # calculate scores for each fitness function and take linear combination
        for i, (fn, weight) in enumerate(zip(self.fitness_fns, self.weights)):
            time_start = time.time()
            fn_scores = fn(z_pool)
            time_end = time.time()
            print(f"{fn.name}: {time_end - time_start:.2f}s")
            all_fn_scores[i] = fn_scores

            # standardize scores between 0 and 1 before taking linear combination
            if (np.max(fn_scores) - np.min(fn_scores)) > 0:
                fn_scores_stand = (fn_scores - np.min(fn_scores)) / (
                    np.max(fn_scores) - np.min(fn_scores)
                )
            else:
                fn_scores_stand = np.zeros_like(fn_scores)
            scores += weight * fn_scores_stand

        self.all_fn_scores = all_fn_scores

        return scores


class SumMaxAbsSMDMinPairwiseEuclideanDist(LinearCombinationFitness):
    """
    Linear combination of sum of maximum absolute SMD and minimum pairwise Euclidean distance between arms

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - pairwise_dists: pairwise distance matrix between units of randomization
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - max_abs_dist_weight: weight to apply to sum of maximum absolute SMD
        - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        arm_compare_pairs: np.ndarray,
        pairwise_dists: np.ndarray,
        mapping: np.ndarray,
        max_abs_dist_weight: float,
        euclidean_dist_weight: float,
    ):
        fitness_fns = [
            SumMaxAbsSMD(X, arm_compare_pairs, mapping),
            MinPairwiseEuclideanDist(pairwise_dists),
        ]
        super().__init__(fitness_fns, [max_abs_dist_weight, euclidean_dist_weight])

        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for SumMaxSMDMinPairwiseEuclideanDist that
        creates an instance of SumMaxSMDMinPairwiseEuclideanDist using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return self(
            trial.X_fit,
            trial.arm_compare_pairs,
            trial.pairwise_dists,
            trial.mapping,
            *trial.config.fitness_fn_weights,
        )

    def __call__(self, z_pool):
        return super().__call__(z_pool)


class SumMaxAbsSMDFracExpo(LinearCombinationFitness):
    """
    Linear combination of sum of maximum absolute SMD and fraction of control group exposed to treatment

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - expo_mdl (ExposureModel): exposure model instance that calculates exposure to treatment
        - A: adjacency matrix
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - max_abs_dist_weight: weight to apply to sum of maximum absolute SMD
        - frac_expo_weight: weight to apply to fraction of control group exposed to treatment
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        arm_compare_pairs: np.ndarray,
        expo_mdl,
        A: np.ndarray,
        mapping: np.ndarray,
        max_abs_dist_weight: float,
        frac_expo_weight: float,
        p_misspec: Optional[float] = 0,
        misspec_seed: Optional[int] = 42,
        mask_misspec: Optional[np.ndarray] = None,
    ):
        fitness_fns = [
            SumMaxAbsSMD(X, arm_compare_pairs, mapping),
            FracExposed(expo_mdl, A, mapping, p_misspec, misspec_seed, mask_misspec),
        ]
        super().__init__(fitness_fns, [max_abs_dist_weight, frac_expo_weight])
        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for SumMaxSMDFracExpo that
        creates an instance of SumMaxSMDFracExpo using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        # if clustered, set misspecification mask such that only edges between clusters are misspecified
        if trial.mapping is not None and trial.config.p_misspec > 0:
            mask_misspec = np.not_equal.outer(trial.mapping, trial.mapping).astype(bool)
        else:
            mask_misspec = None

        return self(
            trial.X_fit,
            trial.arm_compare_pairs,
            trial.expo_mdl,
            trial.A,
            trial.mapping,
            *trial.config.fitness_fn_weights,
            trial.config.p_misspec,
            trial.config.misspec_seed,
            mask_misspec,
        )

    def __call__(self, z_pool):
        return super().__call__(z_pool)


class MaxMahalanobisMinPairwiseEuclideanDist(LinearCombinationFitness):
    """
    Linear combination of maximum Mahalanobis distance and minimum pairwise Euclidean distance between arms

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - pairwise_dists: pairwise distance matrix between units of randomization
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
        - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        arm_compare_pairs,
        pairwise_dists: np.ndarray,
        mapping: np.ndarray,
        max_mahalanobis_weight: float,
        euclidean_dist_weight: float,
    ):
        fitness_fns = [
            MaxMahalanobis(X, arm_compare_pairs, mapping),
            MinPairwiseEuclideanDist(pairwise_dists),
        ]
        super().__init__(fitness_fns, [max_mahalanobis_weight, euclidean_dist_weight])
        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for MaxMahalanobisMinPairwiseEuclideanDist that
        creates an instance of MaxMahalanobisMinPairwiseEuclideanDist using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
            - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
            - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
        """
        return self(
            trial.X_fit,
            trial.arm_compare_pairs,
            trial.pairwise_dists,
            trial.mapping,
            *trial.config.fitness_fn_weights,
        )

    def __call__(self, z_pool):
        return super().__call__(z_pool)


class MaxMahalanobisFracExpo(LinearCombinationFitness):
    """
    Linear combination of maximum Mahalanobis distance and fraction of control group exposed to treatment

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - expo_mdl (ExposureModel): exposure model instance that calculates exposure to treatment
        - A: adjacency matrix
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
        - frac_expo_weight: weight to apply to fraction of control group exposed to treatment
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        arm_compare_pairs,
        expo_mdl,
        A: np.ndarray,
        mapping: np.ndarray,
        max_mahalanobis_weight: float,
        frac_expo_weight: float,
        p_misspec: Optional[float] = 0,
        misspec_seed: Optional[int] = 42,
        mask_misspec: Optional[np.ndarray] = None,
    ):
        fitness_fns = [
            MaxMahalanobis(X, arm_compare_pairs, mapping),
            FracExposed(expo_mdl, A, mapping, p_misspec, misspec_seed, mask_misspec),
        ]
        super().__init__(fitness_fns, [max_mahalanobis_weight, frac_expo_weight])
        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for MaxMahalanobisFracExpo that
        creates an instance of MaxMahalanobisFracExpo using data from trial

        Args:
            - trial (SimulatedTrial): instance of SimulatedTrial
        """
        if trial.mapping is not None and trial.config.p_misspec > 0:
            mask_misspec = np.not_equal.outer(trial.mapping, trial.mapping).astype(bool)
        else:
            mask_misspec = None
        return self(
            trial.X_fit,
            trial.arm_compare_pairs,
            trial.expo_mdl,
            trial.A,
            trial.mapping,
            *trial.config.fitness_fn_weights,
            trial.config.p_misspec,
            trial.config.misspec_seed,
            mask_misspec,
        )

    def __call__(self, z_pool):
        return super().__call__(z_pool)


class WeightedMaxMahalanobisMinPairwiseEuclideanDist(LinearCombinationFitness):
    """
    Linear combination of weighted maximum Mahalanobis distance and minimum pairwise Euclidean distance between arms

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - betas: weights to apply to each covariate
        - pairwise_dists: pairwise distance matrix between units of randomization
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
        - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
    """

    def __init__(
        self,
        X,
        betas,
        arm_compare_pairs,
        pairwise_dists,
        mapping,
        max_mahalanobis_weight,
        euclidean_dist_weight,
    ):
        fitness_fns = [
            WeightedMaxMahalanobis(X, betas, arm_compare_pairs, mapping),
            MinPairwiseEuclideanDist(pairwise_dists),
        ]
        super().__init__(fitness_fns, [max_mahalanobis_weight, euclidean_dist_weight])
        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for WeightedMaxMahalanobisMinPairwiseEuclideanDist that
        creates an instance of WeightedMaxMahalanobisMinPairwiseEuclideanDist using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
            - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
            - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
        """
        return self(
            trial.X_fit,
            trial.beta,
            trial.arm_compare_pairs,
            trial.pairwise_dists,
            trial.mapping,
            *trial.config.fitness_fn_weights,
        )

    def __call__(self, z_pool):
        return super().__call__(z_pool)


class WeightedMaxMahalanobisFracExpo(LinearCombinationFitness):
    """
    Linear combination of weighted maximum Mahalanobis distance and fraction of control group exposed to treatment

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - betas: weights to apply to each covariate
        - expo_mdl (ExposureModel): exposure model instance that calculates exposure to treatment
        - A: adjacency matrix
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
        - frac_expo_weight: weight to apply to fraction of control group exposed to treatment
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        betas: np.ndarray,
        arm_compare_pairs: np.ndarray,
        expo_mdl,
        A: np.ndarray,
        mapping: np.ndarray,
        max_mahalanobis_weight: float,
        frac_expo_weight: float,
        p_misspec: Optional[float] = 0,
        misspec_seed: Optional[int] = 42,
        mask_misspec: Optional[np.ndarray] = None,
    ):
        fitness_fns = [
            WeightedMaxMahalanobis(X, betas, arm_compare_pairs, mapping),
            FracExposed(expo_mdl, A, mapping, p_misspec, misspec_seed, mask_misspec),
        ]
        super().__init__(fitness_fns, [max_mahalanobis_weight, frac_expo_weight])
        return

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for WeightedMaxMahalanobisFracExpo that
        creates an instance of WeightedMaxMahalanobisFracExpo using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        if trial.mapping is not None and trial.config.p_misspec > 0:
            mask_misspec = np.not_equal.outer(trial.mapping, trial.mapping).astype(bool)
        else:
            mask_misspec = None
        return self(
            trial.X_fit,
            trial.beta,
            trial.arm_compare_pairs,
            trial.expo_mdl,
            trial.A,
            trial.mapping,
            *trial.config.fitness_fn_weights,
            trial.config.p_misspec,
            trial.config.misspec_seed,
            mask_misspec,
        )

    def __call__(self, z_pool):
        return super().__call__(z_pool)
