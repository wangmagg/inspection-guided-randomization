import numpy as np
from joblib import Parallel, delayed
import scipy.linalg
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import time

def _smd(z, n_arms, comps, X):
    # calculate mean and standard deviation of each covariate for each arm
    arm_means = np.vstack([np.mean(X[z == arm, :], axis=0) for arm in range(n_arms)])
    arm_stds = np.vstack([np.std(X[z == arm, :], axis=0) for arm in range(n_arms)])

    # calculate pairwise differences in means and standard deviations
    all_smd = np.zeros((comps.shape[0], X.shape[1]))
    for i, pair in enumerate(comps):
        mean_diff = arm_means[pair[0], :] - arm_means[pair[1], :]
        std_sum = np.sqrt(arm_stds[pair[0], :] ** 2 + arm_stds[pair[1], :] ** 2)
        smd = np.divide(mean_diff, std_sum, out=mean_diff, where=std_sum != 0)
        all_smd[i, :] = smd

    return all_smd

def SMD(z_pool, n_arms, comps, X, n_jobs=-2):
    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_smd)(z, n_arms, comps, X) for z in tqdm(z_pool)
    )
    vals = np.array(vals)
    return vals

def _signed_max_abs_smd(z, n_arms, comps, X):
    all_smd = _smd(z, n_arms, comps, X)
    where_max_abs_smd = np.argmax(np.abs(all_smd), axis=0)
    signed_max_abs_smd = all_smd[where_max_abs_smd, np.arange(all_smd.shape[1])]

    return signed_max_abs_smd


def SignedMaxAbsSMD(z_pool, n_arms, comps, X, n_jobs=-2):
    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_signed_max_abs_smd)(z, n_arms, comps, X) for z in z_pool
    )
    return np.array(vals)


def _sum_max_abs_smd(z, n_arms, comps, X):
    signed_max_abs_smds = _signed_max_abs_smd(z, n_arms, comps, X)
    return np.sum(np.abs(signed_max_abs_smds))


def SumMaxAbsSMD(z_pool, n_arms, comps, X, n_jobs=-2):
    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_sum_max_abs_smd)(z, n_arms, comps, X) for z in z_pool
    )
    return np.array(vals)

def _max_mahalanobis(z, n_arms, comps, X, S_hat_inv):
    # calculate Mahalanobis distance between means of each pair of arms
    arm_means = np.vstack([np.mean(X[z == arm, :], axis=0) for arm in range(n_arms)])
    dists = [mahalanobis(arm_means[p[0], :], arm_means[p[1], :], S_hat_inv) for p in comps]

    return np.max(dists)


def MaxMahalanobis(z_pool, n_arms, comps, X, cluster_lbls=None, n_jobs=-2):
    if cluster_lbls is not None:
        z_pool = np.vstack([z[cluster_lbls] for z in z_pool])

    S_hat = np.cov(X.T) 
    S_hat_inv = np.linalg.pinv(S_hat) 

    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_max_mahalanobis)(z, n_arms, comps, X, S_hat_inv) for z in z_pool
    )

    # vals = [_max_mahalanobis(z, n_arms, comps, X, S_hat_inv) for z in tqdm(z_pool, total=z_pool.shape[0])]

    return np.array(vals)

def FracExpo(z_pool, A, q, cluster_lbls=None):
    if cluster_lbls is not None:
        z_pool = np.vstack([z[cluster_lbls] for z in z_pool])

    n_z1_nbrs = np.dot(A.T, z_pool.T).T
    n_nbrs = np.sum(A, axis=0)
    is_expo_z0 = (n_z1_nbrs >= (q * n_nbrs)) & (z_pool == 0)

    n_0 = np.sum(1 - z_pool, axis=1)
    frac_expo = np.divide(np.sum(is_expo_z0, axis=1), n_0, where=n_0 != 0)

    return frac_expo

def InvMinEuclidDist(z_pool, D):
    min_D = np.zeros(z_pool.shape[0])
    for i, z in tqdm(enumerate(z_pool), total=z_pool.shape[0]):
        z = np.expand_dims(z, axis=0)

        # get mask of units assigned to different arms
        Z_diff_mask = np.matmul(z.T, 1 - z) + np.matmul((1 - z).T, z)

        # calculate minimum pairwise Euclidean distance between units assigned to different arms
        D_diff_z = D[Z_diff_mask.astype(bool)]

        # take inverse
        min_D[i] = 1 / np.min(D_diff_z)

    return min_D


def get_metric(metric_name):
    if metric_name == "SumMaxAbsSMD":
        metric_fn = SumMaxAbsSMD
    elif metric_name == "MaxMahalanobis":
        metric_fn = MaxMahalanobis
    elif metric_name == "FracExpo":
        metric_fn = FracExpo
    elif metric_name == "InvMinEuclidDist":
        metric_fn = InvMinEuclidDist
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metric_fn