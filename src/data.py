import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.spatial.distance import pdist, squareform


def gen_multarm_data(n_students, tau_sizes, sigma, seed=42):
    rng = np.random.default_rng(seed)

    tau_sizes = np.array(tau_sizes)
    gender = rng.binomial(1, 0.7, n_students)

    age = rng.uniform(19, 25, n_students)
    major = rng.choice(3, n_students, p=[0.5, 0.3, 0.2])
    ability = (age - np.mean(age)) / np.std(age) - (major == 1) - gender + rng.normal(0, sigma, n_students)
    confidence = gender + (major == 2) + rng.normal(0, sigma, n_students)
    hw = ability + rng.normal(0, sigma, n_students)

    y_0 = ability + confidence + rng.normal(0, sigma, n_students)
    y_arms = y_0 + tau_sizes[:, np.newaxis] * np.std(y_0)

    X_df = pd.DataFrame.from_dict(
        {"gender": gender, "age": age, "major": major, "ability": ability, "hw": hw}
    )

    return y_0, y_arms, X_df

def get_multarm_y_obs(y_0, y_arms, z):
    y_obs = (z == 0) * y_0
    for i in range(len(y_arms)):
        y_obs += (z == (i + 1)) * y_arms[i]
    return y_obs

def gen_composition_data(n_students, prop_men, tau_sizes, sigma, seed=42):
    rng = np.random.default_rng(seed)

    tau_sizes = np.array(tau_sizes)

    gender = np.zeros(n_students)
    gender[: int(prop_men * n_students)] = 1
    rng.shuffle(gender)

    age = rng.uniform(19, 25, n_students)
    major = rng.choice(3, n_students, p=[0.5, 0.3, 0.2])
    ability = (age - np.mean(age)) / np.std(age) - (major == 1) - gender + rng.normal(0, sigma, n_students)
    confidence = gender + (major == 2) + rng.normal(0, sigma, n_students)
    hw = ability + rng.normal(0, sigma, n_students)

    y_0 = ability + confidence + rng.normal(0, sigma, n_students)
    y = y_0 + tau_sizes[:, np.newaxis] * np.std(y_0)

    X_df = pd.DataFrame.from_dict(
        {"gender": gender, "age": age, "major": major, "ability": ability, "hw": hw}
    )

    return y, X_df

def get_composition_y_obs(y, z):
    y_obs = np.zeros_like(z)
    for i, y_i in enumerate(y):
        y_obs += (z == i) * y_i
    return y_obs

def gen_kenya_network(
    set_coords,
    coords_range,
    n_sch_per_set,
    gamma,
    p_same_sch,
    p_same_set_diff_sch,
    p_diff_set_diff_sch,
    block_sizes,
    seed=42
):
    rng = np.random.default_rng(seed)
    all_sch_coords = []
    for coords in set_coords:
        sch_coords_0 = rng.uniform(coords[0] - coords_range, coords[0] + coords_range, n_sch_per_set)
        sch_coords_1 = rng.uniform(coords[1] - coords_range, coords[1] + coords_range, n_sch_per_set)
        sch_coords = np.vstack((sch_coords_0, sch_coords_1)).T
        all_sch_coords.append(sch_coords)
    all_sch_coords = np.vstack(all_sch_coords)

    n_sch = n_sch_per_set * len(set_coords)
    w_mat = np.ones((n_sch, n_sch))
    p_mat = np.ones((n_sch, n_sch))
    sch_to_set = np.repeat(np.arange(len(set_coords)), n_sch_per_set)
    diff_sch = ~np.eye(n_sch, dtype=bool)    

    pdists = squareform(pdist(all_sch_coords, metric="euclidean"))
    pdists_normed = pdists / np.max(pdists)
    intxn_scores = pdists_normed[diff_sch] ** -gamma
    w_mat[diff_sch] = intxn_scores / np.max(intxn_scores)
    
    same_set = sch_to_set[:, np.newaxis] == sch_to_set[np.newaxis, :]
    p_mat[~diff_sch & same_set] = p_same_sch
    p_mat[diff_sch & same_set] = p_same_set_diff_sch * w_mat[diff_sch & same_set]
    p_mat[diff_sch & ~same_set] = p_diff_set_diff_sch * w_mat[diff_sch & ~same_set]

    G = nx.stochastic_block_model(block_sizes, p_mat, seed=seed)
    A = nx.to_numpy_array(G, dtype=np.float32)

    return A, pdists

def gen_kenya_data(
    set_mus,
    n_sch_per_set,
    n_stu_per_sch,
    sigma_sch_in_set,
    sigma_stu_in_sch,
    A,
    beta,
    tau_size,
    sigma,
    seed
):
    rng = np.random.default_rng(seed)
    n_set = set_mus.shape[0]
    set_mus_sd = np.std(set_mus, axis=0)

    X = []
    for set in range(n_set):
        for _ in range(n_sch_per_set):
            sch_mu = rng.normal(set_mus[set], sigma_sch_in_set * set_mus_sd)
            sch_Xs = [rng.normal(sch_mu, sigma_stu_in_sch * set_mus_sd) for _ in range(n_stu_per_sch)]
            X.append(sch_Xs)
    X = np.vstack(X)
    X = np.matmul(A, X)

    eps = rng.normal(0, sigma, X.shape[0])
    y_0 = np.matmul(X, beta.transpose()).squeeze() + eps
    y_1 = y_0 + tau_size * np.std(y_0)

    X_df = pd.DataFrame(X)
    set_lbl = np.repeat(np.arange(n_set), n_sch_per_set * n_stu_per_sch)
    sch_in_set_lbl = np.tile(np.repeat(np.arange(n_sch_per_set), n_stu_per_sch), n_set)
    sch_lbl = np.repeat(np.arange(n_sch_per_set * n_set), n_stu_per_sch)
    X_df["set"] = set_lbl
    X_df["sch_in_set"] = sch_in_set_lbl
    X_df["sch"] = sch_lbl

    return y_0, y_1, X_df


def get_kenya_y_obs(y_0, y_1, z_accepted, A, q, cluster_lbls):
    z_accepted_stu = np.vstack([z[cluster_lbls] for z in z_accepted])

    n_z1_nbrs = np.dot(A.T, z_accepted_stu.T).T
    n_nbrs = np.sum(A, axis=0)
    is_expo = (n_z1_nbrs >= (q * n_nbrs)) | (z_accepted_stu == 1)
    y_obs = y_0 * (1 - is_expo) + y_1 * is_expo

    return y_obs

def get_kenya_sch_y_obs(y_0, y_1, z_accepted, A, q, cluster_lbls):
    y_obs = get_kenya_y_obs(y_0, y_1, z_accepted, A, q, cluster_lbls)

    cluster_range = np.arange(z_accepted.shape[1])
    cluster_mask = (cluster_lbls == cluster_range[:, None])
    sch_y_obs = np.matmul(y_obs, cluster_mask.transpose()) / np.sum(cluster_mask, axis=1)[None, :]

    return sch_y_obs
    

