import numpy as np
from abc import ABC, abstractmethod

try:
    import cupy as cp
    USE_GPU=True
except ModuleNotFoundError:
    USE_GPU=False

def diff_in_means(z, y_obs):
    xp = cp.get_array_module(z) if USE_GPU else np

    z_2d = xp.atleast_2d(z)
    z_comp_2d = xp.atleast_2d(1 - z)
    
    if y_obs.ndim > 1:
        mean_1 = xp.diag(xp.matmul(z, y_obs.T)) / xp.sum(z_2d, axis=1)
        mean_0 = xp.diag(xp.matmul(1 - z, y_obs.T)) / xp.sum(z_comp_2d, axis=1)
    else:
        mean_1 = xp.matmul(z, y_obs) / xp.sum(z_2d, axis=1)
        mean_0 = xp.matmul(1 - z, y_obs) / xp.sum(z_comp_2d, axis=1)

    tau_hat = xp.squeeze(mean_1 - mean_0)

    return tau_hat


def diff_in_means_mult_arm(z, y_obs, n_arms, arm_compare_pairs=None):
    # Default to comparing each arm to the control arm
    if arm_compare_pairs is None:
        n_0 = np.sum(z == 0)
        tau_hats = np.zeros(n_arms - 1)
        for arm in range(1, n_arms):
            n_arm = np.sum(z == arm)
            z_0_arm = np.zeros(n_0 + n_arm)
            z_0_arm[n_0:] = 1
            y_0_arm = np.hstack([y_obs[z == 0], y_obs[z == arm]])
            tau_hats[arm - 1] = diff_in_means(z_0_arm, y_0_arm)
    else:
        tau_hats = np.zeros(len(arm_compare_pairs))
        for pair_idx, pair in enumerate(arm_compare_pairs):
            n_pair_0 = np.sum(z == pair[0])
            n_pair_1 = np.sum(z == pair[1])
            z_pair = np.zeros(n_pair_0 + n_pair_1)
            z_pair[n_pair_0:] = 1
            y_pair = np.hstack([y_obs[z == pair[0]], y_obs[z == pair[1]]])
            tau_hats[pair_idx] = diff_in_means(z_pair, y_pair)

    return tau_hats


def get_cluster_mask(z_cluster, mapping):
    cluster_mask = np.zeros((len(z_cluster), len(mapping)))
    for cluster_idx in range(cluster_mask.shape[0]):
        cluster_mask[cluster_idx, :] = mapping == cluster_idx
    return cluster_mask

def clustered_diff_in_means(z_cluster, y_obs, mapping):
    cluster_mask = get_cluster_mask(z_cluster, mapping)
    cluster_y_obs_means = np.matmul(cluster_mask, np.transpose(y_obs)) / np.sum(
        cluster_mask, axis=1
    )

    return diff_in_means(z_cluster, cluster_y_obs_means)


def qb_diff_in_means_mult_arm(z, y_obs, blocks, n_arms, arm_compare_pairs):
    def within_block_estimate(z_block, y_obs_block):
        if arm_compare_pairs is None:
            n_0 = np.sum(z_block == 0)
            tau_hats = np.zeros(n_arms - 1)
            for arm in range(1, n_arms):
                n_arm = np.sum(z_block == arm)
                z_0_arm = np.zeros(n_0 + n_arm)
                z_0_arm[n_0:] = 1
                y_0_arm = np.hstack(
                    [y_obs_block[z_block == 0], y_obs_block[z_block == arm]]
                )
                tau_hats[arm - 1] = diff_in_means(z_0_arm, y_0_arm)
        else:
            tau_hats = np.zeros(len(arm_compare_pairs))
            for pair_idx, pair in enumerate(arm_compare_pairs):
                n_pair_0 = np.sum(z_block == pair[0])
                n_pair_1 = np.sum(z_block == pair[1])
                z_pair = np.zeros(n_pair_0 + n_pair_1)
                z_pair[n_pair_0:] = 1
                y_pair = np.hstack(
                    [y_obs_block[z_block == pair[0]], y_obs_block[z_block == pair[1]]]
                )
                tau_hats[pair_idx] = diff_in_means(z_pair, y_pair)

        return tau_hats

    n_blocks = blocks.max() + 1

    # Default to comparing each arm to the control arm
    tau_hats_across_blocks = [
        within_block_estimate(z[blocks == block], y_obs[blocks == block])
        for block in range(n_blocks)
    ]

    tau_hats_across_blocks = np.array(tau_hats_across_blocks)
    weights = np.bincount(blocks) / len(blocks)
    weighted_tau_hats_across_blocks = weights[:, np.newaxis] * tau_hats_across_blocks

    return np.sum(weighted_tau_hats_across_blocks, axis=0)


def get_pval(z_pool, y_obs_pool, idx):
    t_obs = diff_in_means(z_pool[idx, :], y_obs_pool[idx, :])
    t_null = np.array([diff_in_means(z, y_obs_pool[idx, :]) for z in z_pool])

    return np.mean(abs(t_null) >= abs(t_obs), axis=0)


def get_pval_mult_arm(z_pool, y_obs_pool, idx, n_arms, arm_compare_pairs=None):
    t_obs = diff_in_means_mult_arm(
        z_pool[idx, :], y_obs_pool[idx, :], n_arms, arm_compare_pairs
    )
    t_null = np.array(
        [
            diff_in_means_mult_arm(z, y_obs_pool[idx, :], n_arms, arm_compare_pairs)
            for z in z_pool
        ]
    )

    return np.mean(abs(t_null) >= abs(t_obs), axis=0)


def get_pval_clustered(z_pool, y_obs_pool, mapping, idx):
    t_obs = clustered_diff_in_means(
        z_pool[idx, :], y_obs_pool[idx, :], mapping
    )
    t_null = np.array(
        [
            clustered_diff_in_means(z, y_obs_pool[idx, :], mapping)
            for z in z_pool
        ]
    )

    return np.mean(abs(np.array(t_null)) >= abs(t_obs))


def get_pval_qb(z_pool, y_obs_pool, idx, blocks, n_arms, arm_compare_pairs):
    t_obs = qb_diff_in_means_mult_arm(
        z_pool[idx, :], y_obs_pool[idx, :], blocks, n_arms, arm_compare_pairs
    )
    t_null = np.array(
        [
            qb_diff_in_means_mult_arm(
                z, y_obs_pool[idx, :], blocks, n_arms, arm_compare_pairs
            )
            for z in z_pool
        ]
    )

    return np.mean(abs(t_null) >= abs(t_obs), axis=0)


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
        self.name = "diff-in-means"
        self.mapping = mapping

    def estimate(self, z, y_obs):
        if self.mapping is not None:
            z = z[self.mapping]
        if y_obs.ndim > 1:
            mean_1 = np.diag(np.matmul(z, np.transpose(y_obs))) / np.sum(
                np.atleast_2d(z), axis=1
            )
            mean_0 = np.diag(np.matmul(1 - z, np.transpose(y_obs))) / np.sum(
                np.atleast_2d(1 - z), axis=1
            )
        else:
            mean_1 = np.matmul(z, y_obs) / np.sum(np.atleast_2d(z), axis=1)
            mean_0 = np.matmul(1 - z, y_obs) / np.sum(np.atleast_2d(1 - z), axis=1)

        return np.squeeze(mean_1 - mean_0)[()]

    def get_pval(self, z_pool, y_obs_pool, idx):
        t_obs = self.estimate(z_pool[idx, :], y_obs_pool[idx, :])
        t_null = np.array([self.estimate(z, y_obs_pool[idx, :]) for z in z_pool])
        return np.mean(abs(t_null) >= abs(t_obs))


class DiffMeansMultArm(Estimator):
    def __init__(self, n_arms, arm_compare_pairs=None):
        self.name = "diff-in-means-mult-arm"
        self.n_arms = n_arms
        self.arm_compare_pairs = arm_compare_pairs

    def estimate(self, z, y_obs):
        diff_means_estimator = DiffMeans()
        # Default to comparing each arm to the control arm
        if self.arm_compare_pairs is None:
            n_0 = np.sum(z == 0)
            tau_hats = np.zeros(self.n_arms - 1)
            for arm in range(1, self.n_arms):
                n_arm = np.sum(z == arm)
                z_0_arm = np.zeros(n_0 + n_arm)
                z_0_arm[n_0:] = 1
                y_0_arm = np.hstack([y_obs[z == 0], y_obs[z == arm]])
                tau_hats[arm - 1] = diff_means_estimator.estimate(z_0_arm, y_0_arm)
        else:
            tau_hats = np.zeros(len(self.arm_compare_pairs))
            for pair_idx, pair in enumerate(self.arm_compare_pairs):
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
        self.name = "clustered-diff-in-means"
        self.mapping = mapping

    def _get_cluster_mask(self, z_cluster):
        cluster_mask = np.zeros((len(z_cluster), len(self.mapping)))
        for cluster_idx in range(cluster_mask.shape[0]):
            cluster_mask[cluster_idx, :] = self.mapping == cluster_idx
        return cluster_mask

    def cache_cluster_masks(self, z_pool):
        cluster_masks = []
        for z_cluster in z_pool:
            cluster_masks.append(self._get_cluster_mask(z_cluster))
        self.cluster_masks = cluster_masks

    def estimate(self, z_cluster, y_obs, cluster_mask=None):
        if cluster_mask is None:
            cluster_mask = self._get_cluster_mask(z_cluster)
        cluster_y_obs_means = np.matmul(cluster_mask, np.transpose(y_obs)) / np.sum(
            cluster_mask, axis=1
        )

        diff_means_estimator = DiffMeans()

        return diff_means_estimator.estimate(z_cluster, cluster_y_obs_means)

    def get_pval(self, z_pool, y_obs_pool, idx):
        t_obs = self.estimate(
            z_pool[idx, :], y_obs_pool[idx, :], self.cluster_masks[idx]
        )
        t_null = np.array(
            [
                self.estimate(z, y_obs_pool[idx, :], self.cluster_masks[idx])
                for z in z_pool
            ]
        )
        return np.mean(abs(np.array(t_null)) >= abs(t_obs))


class QBDiffMeansMultArm(Estimator):
    def __init__(self, blocks, n_arms, arm_compare_pairs):
        self.name = "qb-diff-in-means-mult-arm"
        self.blocks = blocks
        self.n_arms = n_arms
        self.arm_compare_pairs = arm_compare_pairs
        self.weights = np.bincount(self.blocks) / len(self.blocks)

    def within_block_estimate(self, z_block, y_obs_block):
        diff_means_estimator = DiffMeans()
        if self.arm_compare_pairs is None:
            n_0 = np.sum(z_block == 0)
            tau_hats = np.zeros(self.n_arms - 1)
            for arm in range(1, self.n_arms):
                n_arm = np.sum(z_block == arm)
                z_0_arm = np.zeros(n_0 + n_arm)
                z_0_arm[n_0:] = 1
                y_0_arm = np.hstack(
                    [y_obs_block[z_block == 0], y_obs_block[z_block == arm]]
                )
                tau_hats[arm - 1] = diff_means_estimator.estimate(z_0_arm, y_0_arm)
        else:
            tau_hats = np.zeros(len(self.arm_compare_pairs))
            for pair_idx, pair in enumerate(self.arm_compare_pairs):
                n_pair_0 = np.sum(z_block == pair[0])
                n_pair_1 = np.sum(z_block == pair[1])
                z_pair = np.zeros(n_pair_0 + n_pair_1)
                z_pair[n_pair_0:] = 1
                y_pair = np.hstack(
                    [y_obs_block[z_block == pair[0]], y_obs_block[z_block == pair[1]]]
                )
                tau_hats[pair_idx] = diff_means_estimator.estimate(z_pair, y_pair)

        return tau_hats

    def estimate(self, z, y_obs):
        n_blocks = self.blocks.max() + 1

        # Default to comparing each arm to the control arm
        tau_hats_across_blocks = [
            self.within_block_estimate(
                z[self.blocks == block], y_obs[self.blocks == block]
            )
            for block in range(n_blocks)
        ]

        tau_hats_across_blocks = np.array(tau_hats_across_blocks)
        weighted_tau_hats_across_blocks = (
            self.weights[:, np.newaxis] * tau_hats_across_blocks
        )

        return np.sum(weighted_tau_hats_across_blocks, axis=0)

    def get_pval(self, z_pool, y_obs_pool, idx):
        t_obs = self.estimate(z_pool[idx, :], y_obs_pool[idx, :])
        t_null = np.array([self.estimate(z, y_obs_pool[idx, :]) for z in z_pool])

        return np.mean(abs(t_null) >= abs(t_obs), axis=0)
