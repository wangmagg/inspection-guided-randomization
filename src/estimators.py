import numpy as np
from sklearn.linear_model import LinearRegression

def get_tau_true(y_0, y_1, comps):
    y = np.vstack([y_0,  y_1])
    tau_true = np.array(
        [
            (y[pair[1]] - y[pair[0]]).mean()
            for pair in comps
        ]
    )
    return np.squeeze(tau_true)

def get_tau_true_composition(y, comps):
    tau_true = np.array(
        [
            (y[pair[1]] - y[pair[0]]).mean()
            for pair in comps
        ]
    )
    return np.squeeze(tau_true)

def diff_in_means(z, y_obs):
    z_2d = np.atleast_2d(z)
    z_comp_2d = 1 - z_2d
    
    if y_obs.ndim > 1:
        mean_1 = np.diag(np.matmul(z, y_obs.T)) / np.sum(z_2d, axis=1)
        mean_0 = np.diag(np.matmul(1 - z, y_obs.T)) / np.sum(z_comp_2d, axis=1)
    else:
        mean_1 = np.matmul(z, y_obs) / np.sum(z_2d, axis=1)
        mean_0 = np.matmul(1 - z, y_obs) / np.sum(z_comp_2d, axis=1)

    tau_hat = np.squeeze(mean_1 - mean_0)[()]

    return tau_hat


def diff_in_means_mult_arm(z, y_obs, comps):
    tau_hats = np.zeros(len(comps))
    for pair_idx, pair in enumerate(comps):
        mask_0 = (z == pair[0])
        mask_1 = (z == pair[1])

        n_pair_0 = np.sum(mask_0)
        n_pair_1 = np.sum(mask_1)

        z_pair = np.zeros(n_pair_0 + n_pair_1)
        z_pair[n_pair_0:] = 1

        y_pair = np.concatenate([y_obs[mask_0], y_obs[mask_1]])
        tau_hats[pair_idx] = diff_in_means(z_pair, y_pair)

    return tau_hats

def get_cluster_mask(z_cluster, mapping):
    cluster_mask = np.zeros((len(z_cluster), len(mapping)))
    for cluster_idx in range(cluster_mask.shape[0]):
        cluster_mask[cluster_idx, :] = mapping == cluster_idx
    return cluster_mask

def qb_diff_in_means_mult_arm(z, y_obs, blocks, n_blocks, weights, comps):
    # Default to comparing each arm to the control arm
    tau_hats_across_blocks = [
        diff_in_means_mult_arm(z[blocks == block], y_obs[blocks == block], comps)
        for block in range(n_blocks)
    ]

    tau_hats_across_blocks = np.array(tau_hats_across_blocks)
    weighted_tau_hats_across_blocks = weights[:, np.newaxis] * tau_hats_across_blocks

    return np.sum(weighted_tau_hats_across_blocks, axis=0)


def get_pval(z_pool, y_obs_pool, idx):
    t_obs = diff_in_means(z_pool[idx, :], y_obs_pool[idx, :])
    t_null = np.array([diff_in_means(z, y_obs_pool[idx, :]) for z in z_pool])

    return np.mean(abs(t_null) >= abs(t_obs), axis=0)


def get_pval_mult_arm(z_pool, y_obs_pool, idx, comps):
    t_obs = diff_in_means_mult_arm(
        z_pool[idx, :], y_obs_pool[idx, :], comps
    )
    t_null = np.array(
        [
            diff_in_means_mult_arm(z, y_obs_pool[idx, :], comps)
            for z in z_pool
        ]
    )

    return np.mean(abs(t_null) >= abs(t_obs), axis=0)


def get_pval_qb(z_pool, y_obs_pool, idx, blocks, n_blocks, weights, comps):
    t_obs = qb_diff_in_means_mult_arm(
        z_pool[idx, :], y_obs_pool[idx, :], blocks, n_blocks, weights, comps
    )
    t_null = np.array(
        [
            qb_diff_in_means_mult_arm(
                z, y_obs_pool[idx, :], blocks, n_blocks, weights, comps
            )
            for z in z_pool
        ]
    )

    return np.mean(abs(t_null) >= abs(t_obs), axis=0)