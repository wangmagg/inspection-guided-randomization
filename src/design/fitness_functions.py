import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from abc import ABC, abstractmethod
from scipy.spatial.distance import mahalanobis

from typing import List, Union, Optional

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
        - weights: weight matrix (n_arms x n_arms) if using weighted SMD
        - covar_to_weight: index of the covariate to apply weight to (if using weighted SMD)

    """
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray], 
                 mapping: Optional[np.ndarray] = None, 
                 weights: Optional[np.ndarray] = None, 
                 covar_to_weight: Optional[int] = None):
        if weights is None:
            self.name = f'smd'
        else:
            self.name = f'smd_weighted-{weights}-{covar_to_weight}'

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.X = X
        self.mapping = mapping
        self.weights = weights
        self.covar_to_weight = covar_to_weight
    
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
        return self(trial.X_fit, trial.mapping)
    
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
            Tensor of SMDs for each pairwise comparison of arms for each covariate 
            (n_arms x n_arms x n_covariates)
        """
        n_arms = int(np.max(z) + 1)

        arm_means = np.vstack([np.mean(self.X[z == arm, :], axis=0) for arm in range(n_arms)])
        arm_stds = np.vstack([np.std(self.X[z == arm, :], axis=0) for arm in range(n_arms)])

        # calculate pairwise differences in means and standard deviations
        # results in (n_arms * n_arms) x n_covariates matrix
        mean_diff = arm_means[:, np.newaxis, :] - arm_means[np.newaxis, :, :]
        std_sum = np.sqrt(arm_stds[:, np.newaxis, :]**2 + arm_stds[np.newaxis, :, :]**2)

        all_smd = np.zeros((mean_diff.shape[0], mean_diff.shape[1], self.X.shape[1]))
        for col_idx in range(self.X.shape[1]):
            mean_diff_col = mean_diff[:, :, col_idx]
            std_sum_col = std_sum[:, :, col_idx]
            # multiply by weight matrix if using weighted SMD and covariate is the one to weight
            if self.covar_to_weight is not None and col_idx == self.covar_to_weight:
                weight_matrix = self.weight_matrix(n_arms)
                mean_diff_col = mean_diff_col * weight_matrix
            smd_col = np.divide(mean_diff_col, std_sum_col, out=mean_diff_col, where=std_sum_col != 0)
            all_smd[:, :, col_idx] = smd_col
        
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

class SumSMD(SMD):
    """
    Sum of SMDs across all covariates

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - weights: weight matrix (n_arms x n_arms) if using weighted SMD
        - covar_to_weight: index of the covariate to apply weight to (if using weighted SMD)
    """
    def __init__(self, X, mapping=None, weights=None, covar_to_weight=None):
        super().__init__(X, mapping, weights, covar_to_weight)
        self.name = f'sum-{self.name}'
        return
    
    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for SumSMD that
        creates an instance of SMD using data from trial
        
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        Returns:
            - instance of SumSMD
        """
        return self(trial.X_fit, trial.mapping)
    
    def _single_z(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate sum of SMDs for a single treatment assignment

        Args:
            - z: treatment assignment vector

        Returns:
            Sum of SMDs for each pairwise comparison of arms (n_arms x n_arms)
        """
        smd = super()._single_z(z)
        return np.sum(smd)
    
    def __call__(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate sum of SMDs for a pool of treatment assignments

        Returns:
            Array of sum of SMDs for each treatment assignment in the pool
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
    def __init__(self, X, mapping=None, weights=None, covar_to_weight=None):
        super().__init__(X, mapping, weights, covar_to_weight)
        self.name = f'max-{self.name}'
        return
    
    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor that
        creates an instance of MaxAbsSMD using data from trial
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return self(trial.X_fit, trial.mapping)
    
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
        return np.max(abs_smd, axis=(0, 1))
    
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
    def __init__(self, X, mapping=None, weights=None, covar_to_weight=None):
        super().__init__(X, mapping, weights, covar_to_weight)
        self.name = f'max-{self.name}'
        return
    
    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for SignedMaxAbsSMD that
        creates an instance of SignedMaxAbsSMD using data from trial
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return self(trial.X_fit, trial.mapping)
    
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
        smd_reshaped = smd.reshape(-1, smd.shape[2])
        where_max_abs_smd = np.argmax(np.abs(smd_reshaped), axis=0)
        signed_max_abs_smd = smd_reshaped[where_max_abs_smd, np.arange(smd_reshaped.shape[1])]
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

class SumMaxSMD(MaxAbsSMD):
    """
    Sum of maximum absolute SMD across all covariates for each pairwise comparison of arms

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - weights: weight matrix (n_arms x n_arms) if using weighted SMD
        - covar_to_weight: index of the covariate to apply weight to (if using weighted SMD)
    """
    def __init__(self, X, mapping=None, weights=None, covar_to_weight=None):
        super().__init__(X, mapping, weights, covar_to_weight)
        self.name = f'sum-{self.name}'
    
    @classmethod
    def from_trial(self, trial):
        """
        Create an instance of SumMaxSMD using data from trial
        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return self(trial.X_fit, trial.mapping)
    
    def _single_z(self, z: np.ndarray) -> np.float32:
        """
        Calculate sum of maximum absolute SMD for a single treatment assignment
        Args:
            - z: treatment assignment vector
        Returns:
            Sum of maximum absolute SMD for each pairwise comparison of arms (n_arms x n_arms)
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
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray],
                 mapping: Optional[np.ndarray]=None):
        self.name = 'max-mahalanobis'
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.X = X
        self.mapping = mapping

        # calculate sample covariance matrix
        x_bar = np.expand_dims(np.mean(self.X, axis=0), 1)
        N = self.X.shape[0]
        S_hat = 1 / (N-1) * np.matmul(self.X.T - x_bar, (self.X.T - x_bar).T)
        self.S_hat_inv = np.linalg.inv(S_hat)

    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for MaxMahalanobis that
        creates an instance of MaxMahalanobis using data from trial
        """
        return self(trial.X_fit, trial.mapping)
    
    def _get_arm_means_pairs(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate covariate mean vectors for each arm 

        Args:
            - z: treatment assignment vector
        Returns:
            - arm_means: covariate mean vectors for each arm (n_arms x n_covariates)
            - arm_pairs: array of arm comparisons to make (n_arms^2 x 2) 
        """
        n_arms = int(np.max(z) + 1)

        # calculate mean of each covariate for each arm
        arm_means = np.zeros((n_arms, self.X.shape[1]))
        for arm in range(n_arms):
            arm_means[arm, :] = np.mean(self.X[z == arm, :], axis=0)
        
        arm_pairs = list(itertools.product(*np.tile(np.arange(n_arms), (2, 1))))

        return arm_means, arm_pairs

    
    def _single_z(self, z: np.ndarray) -> np.float32:
        """
        Calculate maximum Mahalanobis distance for a single treatment assignment

        Args:
            - z: treatment assignment vector
        Returns:
            Maximum Mahalanobis distance across all pairwise arm comparisons
        """
        arm_means, arm_pairs = self._get_arm_means_pairs(z)
        dists = [mahalanobis(arm_means[p[0], :], arm_means[p[1], :], self.S_hat_inv) for p in arm_pairs]

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

        return np.array([self._single_z(z) for z in z_pool])
    
class WeightedMaxMahalanobis(MaxMahalanobis):
    """
    Maximum Mahalanobis distance across all pairwise arm comparisons, 
    with weights on more "important" covariates (hypothesized to be more prognostic of outcome)

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - betas: weights to apply to each covariate
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
    """
    def __init__(self, X, betas, mapping=None):
        super().__init__(X, mapping)
        self.name = f'weighted-{self.name}'
        self.betas = betas
    
    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for WeightedMaxMahalanobis that
        creates an instance of WeightedMaxMahalanobis using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """

        return self(trial.X_fit, trial.beta, trial.mapping)
    
    def _single_z(self, z: np.ndarray) -> np.float32:
        """
        Calculate weighted maximum Mahalanobis distance for a single treatment assignment

        Args:
            - z: treatment assignment vector
        Returns:
            Weighted maximum Mahalanobis distance across all pairwise arm comparisons
        """
        arm_means, arm_pairs = self._get_arm_means_pairs(z)
        dists = [mahalanobis(arm_means[p[0], :]*self.betas, arm_means[p[1], :]*self.betas, self.S_hat_inv) for p in arm_pairs]
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
    def __init__(self, 
                 expo_mdl, 
                 A: np.ndarray, 
                 mapping: Optional[np.ndarray]=None):
        self.name = 'frac-exposed'
        self.expo_mdl = expo_mdl
        self.A = A
        self.mapping = mapping
        
        return
    
    @classmethod
    def from_trial(self, trial):
        """
        Alternate constructor for FracExposed that
        creates an instance of FracExposed using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
        """
        return self(trial.expo_mdl, trial.A, trial.mapping)
    
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
        is_exposed = self.expo_mdl(z_pool, self.A)

        # calculate fraction of control group exposed to treatment
        if z_pool.ndim == 1:
            z_pool = np.expand_dims(z_pool, axis=0)
        n_0 = np.sum(1 - z_pool, axis=1)
        return np.divide(np.sum(is_exposed * (1 - z_pool), axis=1), n_0, where = n_0 != 0) 

class MinPairwiseEuclideanDist(Fitness):
    """
    Minimum pairwise Euclidean distance between arms (multiplied by -1 so that higher values are better)

    Args:
        - pairwise_dists: pairwise distance matrix between units of randomization
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
    """
    def __init__(self, 
                 pairwise_dists: np.ndarray):
        self.name = 'min-pairwise-euclidean-dist'
        self.pairwise_dists = pairwise_dists        
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
        for i,z in tqdm(enumerate(z_pool), total=z_pool.shape[0]):
            z_expand = np.expand_dims(z, axis=0)

            # get mask of units assigned to different arms
            Z_diff_mask = np.matmul(z_expand.T, 1-z_expand) + np.matmul((1-z_expand).T, z_expand)

            # calculate minimum pairwise Euclidean distance between units assigned to different arms
            dists_diff_z = self.pairwise_dists[Z_diff_mask.astype(bool)]

            # multiply by -1 so that higher values are better
            min_dists[i] = -np.min(dists_diff_z)
        
        return min_dists
        
class LinearCombinationFitness(Fitness, ABC):
    """
    Abstract class for linear combination of two fitness functions

    Args:
        - fitness_fns: list of fitness function instances
        - weights: weights to apply to each fitness function
    """
    def __init__(self, 
                 fitness_fns: List[Fitness], 
                 weights: Union[List[float], np.ndarray]):
        self.name = 'lin-comb'
        for fn, weight in zip(fitness_fns, weights):
            self.name += f'_{fn.name}-{weight:.2f}'
        self.weights = weights
        self.fitness_fns = fitness_fns
        return
    
    def __call__(self, 
                 z_pool: np.ndarray) -> np.ndarray:
        """
        Calculate linear combination of two fitness functions for a pool of treatment assignments

        Args:
            - z_pool: pool of treatment assignment vectors
        """
        scores = np.zeros(z_pool.shape[0])
        # calculate scores for each fitness function and take linear combination
        for fn, weight in zip(self.fitness_fns, self.weights):
            fn_scores = fn(z_pool)
            # standardize scores between 0 and 1 before taking linear combination
            if (np.max(fn_scores) - np.min(fn_scores)) > 0:
                fn_scores_stand = (fn_scores - np.min(fn_scores)) / (np.max(fn_scores) - np.min(fn_scores))
            else:
                fn_scores_stand = np.zeros_like(fn_scores)
            scores += weight * fn_scores_stand
        return scores

class SumMaxSMDMinPairwiseEuclideanDist(LinearCombinationFitness):
    """
    Linear combination of sum of maximum absolute SMD and minimum pairwise Euclidean distance between arms

    Args:
        - X: data matrix (n_indivs x n_covariates)
        - pairwise_dists: pairwise distance matrix between units of randomization
        - mapping: mapping from cluster treatment assignment to individual treatment assignment
        - max_abs_dist_weight: weight to apply to sum of maximum absolute SMD
        - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
    """
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray],
                 pairwise_dists: np.ndarray,
                 mapping: np.ndarray, 
                 max_abs_dist_weight: float, 
                 euclidean_dist_weight: float):
        fitness_fns = [SumMaxSMD(X, mapping), MinPairwiseEuclideanDist(pairwise_dists)]
        super().__init__(fitness_fns, [max_abs_dist_weight, euclidean_dist_weight])
        return
    
    @classmethod
    def from_trial(self, 
                   trial, 
                   max_abs_dist_weight: float, 
                   euclidean_dist_weight: float):
        """
        Alternate constructor for SumMaxSMDMinPairwiseEuclideanDist that
        creates an instance of SumMaxSMDMinPairwiseEuclideanDist using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
            - max_abs_dist_weight: weight to apply to sum of maximum absolute SMD
            - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
        """
        return self(trial.X_fit, trial.pairwise_dists, trial.mapping,
                   max_abs_dist_weight, euclidean_dist_weight)
    
    def __call__(self, z_pool):
        return super().__call__(z_pool)
    
class SumMaxSMDFracExpo(LinearCombinationFitness):
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
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray],
                 expo_mdl, 
                 A: np.ndarray, 
                 mapping: np.ndarray, 
                 max_abs_dist_weight: float, 
                 frac_expo_weight: float):
        fitness_fns = [SumMaxSMD(X, mapping), FracExposed(expo_mdl, A, mapping)]
        super().__init__(fitness_fns, [max_abs_dist_weight, frac_expo_weight])
        return
    
    @classmethod
    def from_trial(self, 
                   trial, 
                   max_abs_dist_weight: float, 
                   frac_expo_weight: float):
        """
        Alternate constructor for SumMaxSMDFracExpo that
        creates an instance of SumMaxSMDFracExpo using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
            - max_abs_dist_weight: weight to apply to sum of maximum absolute SMD
            - frac_expo_weight: weight to apply to fraction of control group exposed to treatment
        """
        return self(trial.X_fit, trial.expo_mdl, trial.A, trial.mapping,
                   max_abs_dist_weight, frac_expo_weight)
    
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
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray],
                 pairwise_dists: np.ndarray, 
                 mapping: np.ndarray, 
                 max_mahalanobis_weight: float, 
                 euclidean_dist_weight: float):
        fitness_fns = [MaxMahalanobis(X, mapping), MinPairwiseEuclideanDist(pairwise_dists)]
        super().__init__(fitness_fns, [max_mahalanobis_weight, euclidean_dist_weight])
        return
    
    @classmethod
    def from_trial(self, 
                   trial,
                   max_mahalanobis_weight: float,
                   euclidean_dist_weight: float):
        """
        Alternate constructor for MaxMahalanobisMinPairwiseEuclideanDist that
        creates an instance of MaxMahalanobisMinPairwiseEuclideanDist using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
            - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
            - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
        """
        return self(trial.X_fit, trial.pairwise_dists, trial.mapping,
                   max_mahalanobis_weight, euclidean_dist_weight)
    
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
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray], 
                 expo_mdl, 
                 A: np.ndarray, 
                 mapping: np.ndarray, 
                 max_mahalanobis_weight: float, 
                 frac_expo_weight: float):
        fitness_fns = [MaxMahalanobis(X, mapping), FracExposed(expo_mdl, A, mapping)]
        super().__init__(fitness_fns, [max_mahalanobis_weight, frac_expo_weight])
        return
    
    @classmethod
    def from_trial(self, 
                   trial, 
                   max_mahalanobis_weight: float, 
                   frac_expo_weight: float):
        """
        Alternate constructor for MaxMahalanobisFracExpo that
        creates an instance of MaxMahalanobisFracExpo using data from trial

        Args:
            - trial (SimulatedTrial): instance of SimulatedTrial
            - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
            - frac_expo_weight: weight to apply to fraction of control group exposed to treatment
        """
        return self(trial.X_fit, trial.expo_mdl, trial.A, trial.mapping,
                   max_mahalanobis_weight, frac_expo_weight)
    
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
    def __init__(self, X, betas, pairwise_dists, mapping, max_mahalanobis_weight, euclidean_dist_weight):
        fitness_fns = [WeightedMaxMahalanobis(X, betas, mapping), MinPairwiseEuclideanDist(pairwise_dists)]
        super().__init__(fitness_fns, [max_mahalanobis_weight, euclidean_dist_weight])
        return
    
    @classmethod
    def from_trial(self, 
                   trial, 
                   max_mahalanobis_weight: float, 
                   euclidean_dist_weight: float):
        """
        Alternate constructor for WeightedMaxMahalanobisMinPairwiseEuclideanDist that
        creates an instance of WeightedMaxMahalanobisMinPairwiseEuclideanDist using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
            - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
            - euclidean_dist_weight: weight to apply to minimum pairwise Euclidean distance
        """
        return self(trial.X_fit, trial.pairwise_dists, trial.mapping,
                   max_mahalanobis_weight, euclidean_dist_weight)
    
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
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray], 
                 betas: np.ndarray, 
                 expo_mdl,
                 A: np.ndarray, 
                 mapping: np.ndarray, 
                 max_mahalanobis_weight: float, 
                 frac_expo_weight: float):
        fitness_fns = [WeightedMaxMahalanobis(X, betas, mapping), FracExposed(expo_mdl, A, mapping)]
        super().__init__(fitness_fns, [max_mahalanobis_weight, frac_expo_weight])
        return
    
    @classmethod
    def from_trial(self, 
                   trial, 
                   max_mahalanobis_weight: float, 
                   frac_expo_weight: float):
        """
        Alternate constructor for WeightedMaxMahalanobisFracExpo that
        creates an instance of WeightedMaxMahalanobisFracExpo using data from trial

        Args:
            - trial (SimulatedTrial): trial instance with data attributes set
            - max_mahalanobis_weight: weight to apply to maximum Mahalanobis distance
            - frac_expo_weight: weight to apply to fraction of control group exposed to treatment
        """
        return self(trial.X_fit, trial.expo_mdl, trial.A, trial.mapping,
                   max_mahalanobis_weight, frac_expo_weight)
    
    def __call__(self, z_pool):
        return super().__call__(z_pool)
    