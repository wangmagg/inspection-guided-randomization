import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm

def _smd(z: np.ndarray, n_arms: int, comps: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Calculate standardized mean differences (SMD) for each covariate between pairs of arms
    Args:
        - z: treatment assignment vector
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
    """
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

def SMD(z_pool: np.ndarray, n_arms: int, comps: np.ndarray, X: np.ndarray, n_jobs:int=-2) -> np.ndarray:
    """
    Calculate standardized mean differences (SMD) for each covariate between pairs of arms
    for each candidate treatment allocation in pool
    Args:
        - z_pool: candidate treatment allocation pool
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
        - n_jobs: number of parallel jobs to run; defaults to number of cores minus 2
    """
    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_smd)(z, n_arms, comps, X) for z in tqdm(z_pool)
    )
    vals = np.array(vals)
    return vals

def _signed_max_abs_smd(z: np.ndarray, n_arms: int, comps: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Calculate the signed maximum absolute standardized mean difference (SMD) for each covariate
    between pairs of arms
    Args:
        - z: treatment assignment vector
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
    """
    all_smd = _smd(z, n_arms, comps, X)
    where_max_abs_smd = np.argmax(np.abs(all_smd), axis=0)
    signed_max_abs_smd = all_smd[where_max_abs_smd, np.arange(all_smd.shape[1])]

    return signed_max_abs_smd


def SignedMaxAbsSMD(z_pool: np.ndarray, n_arms: int, comps: np.ndarray, X: np.ndarray, n_jobs=-2) -> np.ndarray:
    """
    Calculate the signed maximum absolute standardized mean difference (SMD) for each covariate
    between pairs of arms for each candidate treatment allocation in pool
    Args:
        - z_pool: candidate treatment allocation pool
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
        - n_jobs: number of parallel jobs to run; defaults to number of cores minus 2
    """
    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_signed_max_abs_smd)(z, n_arms, comps, X) for z in z_pool
    )
    return np.array(vals)


def _sum_max_abs_smd(z: np.ndarray, n_arms: np.ndarray, comps: np.ndarray, X: np.ndarray) -> float:
    """
    Calculate the sum of the signed maximum absolute standardized mean differences (SMD) for each covariate
    between pairs of arms
    Args:
        - z: treatment assignment vector
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
    """
    signed_max_abs_smds = _signed_max_abs_smd(z, n_arms, comps, X)
    return np.sum(np.abs(signed_max_abs_smds))


def SumMaxAbsSMD(z_pool: np.ndarray, n_arms: np.ndarray, comps: np.ndarray, X: np.ndarray, n_jobs: int=-2) -> np.ndarray:
    """
    Calculate the sum of the signed maximum absolute standardized mean differences (SMD) for each covariate
    between pairs of arms for each candidate treatment allocation in pool
    Args:
        - z_pool: candidate treatment allocation pool
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
        - n_jobs: number of parallel jobs to run; defaults to number of cores minus 2
    """
    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_sum_max_abs_smd)(z, n_arms, comps, X) for z in z_pool
    )
    return np.array(vals)

def _max_mahalanobis(z: np.ndarray, n_arms: int, comps: np.ndarray, X: np.ndarray, S_hat_inv: np.ndarray) -> float:
    """
    Calculate the maximum Mahalanobis distance between means of pairs of arms
    Args:
        - z: treatment assignment vector
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
        - S_hat_inv: inverse of the covariance matrix
    """
    arm_means = np.vstack([np.mean(X[z == arm, :], axis=0) for arm in range(n_arms)])
    dists = [mahalanobis(arm_means[p[0], :], arm_means[p[1], :], S_hat_inv) for p in comps]

    return np.max(dists)


def MaxMahalanobis(z_pool, n_arms, comps, X, cluster_lbls=None, n_jobs=-2) -> np.ndarray:
    """
    Calculate the maximum Mahalanobis distance between means of pairs of arms for each candidate treatment allocation in pool
    Args:
        - z_pool: candidate treatment allocation pool
        - n_arms: number of arms
        - comps: list of pairs of arms to compare
        - X: covariate matrix
        - cluster_lbls: cluster labels for units; if provided, maps cluster-level treatment assignment vector to unit-level
        - n_jobs: number of parallel jobs to run; defaults to number of cores minus 2
    """
    if cluster_lbls is not None:
        z_pool = np.vstack([z[cluster_lbls] for z in z_pool])

    # calculate covariance matrix and its inverse
    S_hat = np.cov(X.T) 
    S_hat_inv = np.linalg.pinv(S_hat) 

    vals = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_max_mahalanobis)(z, n_arms, comps, X, S_hat_inv) for z in z_pool
    )
    return np.array(vals)

def FracExpo(z_pool: np.ndarray, A: np.ndarray, q: float, cluster_lbls:np.ndarray=None) -> np.ndarray:
    """
    Calculate the fraction of control units that are exposed to treatment via their neighbors
    under a q-fraction exposure model for each candidate allocation in pool

    Args:
        - z_pool: candidate treatment allocation pool
        - A: adjacency matrix
        - q: fraction threshold in neighborhood exposure model
        - cluster_lbls: cluster labels for units; if provided, maps cluster-level treatment assignment vector to unit-level
    """
    if cluster_lbls is not None:
        z_pool = np.vstack([z[cluster_lbls] for z in z_pool])

    # Get the number of treated and the number of total neighbors that each individual has
    n_z1_nbrs = np.dot(A.T, z_pool.T).T
    n_nbrs = np.sum(A, axis=0)

    # Calculate the fraction of control units that are exposed to treatment via their neighbors
    is_expo_z0 = (n_z1_nbrs >= (q * n_nbrs)) & (z_pool == 0)
    n_0 = np.sum(1 - z_pool, axis=1)
    frac_expo = np.divide(np.sum(is_expo_z0, axis=1), n_0, where=n_0 != 0)

    return frac_expo

def InvMinEuclidDist(z_pool: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of the minimum Euclidean distance between units assigned to different arms
    for each candidate allocation in pool

    Args:
        - z_pool: candidate treatment allocation pool
        - D: pairwise Euclidean distance matrix
    """
    min_D = np.zeros(z_pool.shape[0])
    for i, z in tqdm(enumerate(z_pool), total=z_pool.shape[0]):
        z = np.expand_dims(z, axis=0)

        # get mask of units assigned to different arms
        Z_diff_mask = np.matmul(z.T, 1 - z) + np.matmul((1 - z).T, z)

        # calculate minimum Euclidean distance between units assigned to different arms
        # and take the inverse
        D_diff_z = D[Z_diff_mask.astype(bool)]
        min_D[i] = 1 / np.min(D_diff_z)

    return min_D


def get_metric(metric_name: str) -> callable:
    """
    Get metric function based on metric name
    Args:
        - metric_name: name of metric
    """
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