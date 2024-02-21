import numpy as np
from abc import ABC, abstractmethod

class Estimator(ABC):
    def __init__(self):
        self.name = None
    
    @abstractmethod
    def estimate(self, z, y_obs):
        pass
    @abstractmethod
    def get_pval(self, z_pool, y_obs_pool, idx):
        pass

class DiffMeans(Estimator):
    def __init__(self, mapping=None):
        self.name = 'diff-in-means'
        self.mapping = mapping
    
    def estimate(self, z, y_obs):
        if self.mapping is not None:
            z = z[self.mapping]
        if y_obs.ndim > 1:
             mean_1 = np.diag(np.matmul(z, np.transpose(y_obs))) / np.sum(np.atleast_2d(z), axis=1)
             mean_0 = np.diag(np.matmul(1 - z, np.transpose(y_obs))) / np.sum(np.atleast_2d(1-z), axis=1)
        else:
            mean_1 = np.matmul(z, y_obs) / np.sum(np.atleast_2d(z), axis=1)
            mean_0 = np.matmul(1 - z, y_obs) / np.sum(np.atleast_2d(1-z), axis=1)

        return np.squeeze(mean_1 - mean_0)[()]
    
    def get_pval(self, z_pool, y_obs_pool, idx):
        t_obs = self.estimate(z_pool[idx, :], y_obs_pool[idx, :])
        t_null = np.array([self.estimate(z, y_obs_pool[idx, :]) for z in z_pool])
        return np.mean(abs(t_null) >= abs(t_obs))

class DiffMeansMultArm(Estimator):
    def __init__(self, n_arms, arm_pairs=None):
        self.name = 'diff-in-means-mult-arm'
        self.n_arms = n_arms
        self.arm_pairs = arm_pairs
    
    def estimate(self, z, y_obs):
        diff_means_estimator = DiffMeans()
        # Default to comparing each arm to the control arm
        if self.arm_pairs is None:
            n_0 = np.sum(z == 0)
            tau_hats = np.zeros(self.n_arms - 1)
            for arm in range(1, self.n_arms):
                n_arm = np.sum(z == arm)
                z_0_arm = np.zeros(n_0 + n_arm)
                z_0_arm[n_0:] = 1
                y_0_arm = np.hstack([y_obs[z == 0], y_obs[z == arm]])
                tau_hats[arm-1] = diff_means_estimator.estimate(z_0_arm, y_0_arm)
        else:
            tau_hats = np.zeros(len(self.arm_pairs))
            for pair_idx, pair in enumerate(self.arm_pairs):
                n_pair_0 = np.sum(z == pair[0])
                n_pair_1 = np.sum(z == pair[1])
                z_pair = np.zeros(n_pair_0 + n_pair_1)
                z_pair[n_pair_0:] = 1
                y_pair = np.hstack([y_obs[z == pair[0]], y_obs[z == pair[1]]])
                tau_hats[pair_idx] = diff_means_estimator.estimate(z_pair, y_pair)

        return tau_hats
    
    def get_pval(self, z_pool, y_obs_pool, idx):
        t_obs = self.estimate(z_pool[idx, :], y_obs_pool[idx, :])
        t_null = np.array([self.estimate(z, y_obs_pool[idx, :]) for z in z_pool])
        
        return np.mean(abs(t_null) >= abs(t_obs), axis=0)
    
class ClusteredDiffMeans(Estimator):
    def __init__(self, mapping):
        self.name = 'clustered-diff-in-means'
        self.mapping = mapping
        
    def _get_cluster_mask(self, z_cluster):
        cluster_mask = np.zeros((len(z_cluster), len(self.mapping)))
        for cluster_idx in range(cluster_mask.shape[0]):
            cluster_mask[cluster_idx, :] = (self.mapping == cluster_idx)
        return cluster_mask
    
    def cache_cluster_masks(self, z_pool):
        cluster_masks = []
        for z_cluster in z_pool:
            cluster_masks.append(self._get_cluster_mask(z_cluster))
        self.cluster_masks = cluster_masks

    def estimate(self, z_cluster, y_obs, cluster_mask=None):
        if cluster_mask is None:
            cluster_mask = self._get_cluster_mask(z_cluster)
        cluster_y_obs_means = np.matmul(cluster_mask, np.transpose(y_obs)) / np.sum(cluster_mask, axis=1)

        diff_means_estimator = DiffMeans()
        return diff_means_estimator.estimate(z_cluster, cluster_y_obs_means)
    
    def get_pval(self, z_pool, y_obs_pool, idx):
        t_obs = self.estimate(z_pool[idx, :], y_obs_pool[idx, :], self.cluster_masks[idx])
        t_null = np.array([self.estimate(z, y_obs_pool[idx, :], self.cluster_masks[idx]) for z in z_pool])
        return np.mean(abs(np.array(t_null)) >= abs(t_obs))