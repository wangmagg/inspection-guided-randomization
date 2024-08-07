import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from matplotlib import pyplot as plt


def gen_multarm_data(
    n_students: int, tau_sizes: np.ndarray, sigma: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate covariates and potential outcomes for multi-arm experiment simulation
    Outcome is a linear function of ability and confidence
    Confidence is a linear function of gender and major
    Ability is a linear function of age, major, and gender
    Args:
        - n_students: number of students
        - tau_sizes: effect sizes for each arm
        - sigma: standard deviation of noise
        - seed: random seed
    """
    rng = np.random.default_rng(seed)

    tau_sizes = np.array(tau_sizes)
    gender = rng.binomial(1, 0.7, n_students)

    age = rng.uniform(19, 25, n_students)
    major = rng.choice(3, n_students, p=[0.5, 0.3, 0.2])
    ability = (
        (age - np.mean(age)) / np.std(age)
        - (major == 1)
        - 0.5 * gender
        + rng.normal(0, sigma, n_students)
    )
    confidence = gender + (major == 2) + rng.normal(0, sigma, n_students)
    hw = ability + rng.normal(0, sigma, n_students)

    y_0 = ability + confidence + rng.normal(0, sigma, n_students)
    y_arms = y_0 + tau_sizes[:, np.newaxis] * np.std(y_0)

    X_df = pd.DataFrame.from_dict(
        {"gender": gender, "age": age, "major": major, "hw": hw, "ability": ability, "confidence": confidence}
    )

    return y_0, y_arms, X_df


def get_multarm_y_obs(y_0: np.ndarray, y_arms: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Generate observed outcomes for multi-arm experiment
    from the potential outcomes and the treatment assignments
    """
    y_obs = (z == 0) * y_0
    for i in range(len(y_arms)):
        y_obs += (z == (i + 1)) * y_arms[i]
    return y_obs


def gen_composition_data(
    n_students: int,
    prop_men: float,
    tau_sizes: np.ndarray,
    sigma: float,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate covariates and potential outcomes for group formation experiment simulation

    Similar DGP to the multi-arm experiment, but gender is
    in a fixed proportion and the effect sizes correspond
    to each composition and treatment pair.

    Args:
        - n_students: number of students
        - prop_men: proportion of the students that are men
        - tau_sizes: effect sizes for each composition and treatment pair
        - sigma: standard deviation of noise
        - seed: random seed
    """
    rng = np.random.default_rng(seed)

    tau_sizes = np.array(tau_sizes)

    gender = np.zeros(n_students)
    gender[: int(prop_men * n_students)] = 1
    rng.shuffle(gender)

    age = rng.uniform(19, 25, n_students)
    major = rng.choice(3, n_students, p=[0.5, 0.3, 0.2])
    ability = (
        (age - np.mean(age)) / np.std(age)
        - (major == 1)
        - 0.5 * gender
        + rng.normal(0, sigma, n_students)
    )
    confidence = gender + (major == 2) + rng.normal(0, sigma, n_students)
    hw = ability + rng.normal(0, sigma, n_students)

    y_0 = ability + confidence + rng.normal(0, sigma, n_students)
    y = y_0 + tau_sizes[:, np.newaxis] * np.std(y_0)

    X_df = pd.DataFrame.from_dict(
        {"gender": gender, "age": age, "major": major, "hw": hw, "ability": ability, "confidence": confidence}
    )

    return y, X_df


def get_composition_y_obs(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Generate observed outcomes for group formation experiment
    from the potential outcomes and the treatment assignments

    Args:
        - y: potential outcomes
        - z: treatment assignment vector
    """
    y_obs = np.zeros_like(z)
    for i, y_i in enumerate(y):
        y_obs += (z == i) * y_i
    return y_obs


def gen_kenya_network(
    set_coords: np.ndarray,
    coords_range: float,
    n_sch_per_set: int,
    gamma: float,
    p_same_sch: float,
    p_same_set_diff_sch: float,
    p_diff_set_diff_sch: float,
    block_sizes: np.ndarray[int],
    seed: int=42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate schools and individual-level interaction network for simulated experiment with interference

    Use stochastic block model with different edge probabilities
    for individuals in the same school, in the same settlement but different schools,
    and individuals in different settlements

    Args:
        - set_coords: coordinates of the centroids of the settlements
        - coords_range: range of the coordinates used to generate school locations
        - n_sch_per_set: number of schools per settlement
        - gamma: distance decay rate for the interaction scores
        - p_same_sch: edge probability for individuals in the same school
        - p_same_set_diff_sch: edge probability factor for individuals in the same settlement but different schools
        - p_diff_set_diff_sch: edge probability factor for individuals in different settlements
        - block_sizes: number of individuals in each school
        - seed: random seed

    """
    rng = np.random.default_rng(seed)
    all_sch_coords = []

    # Generate lat/long coordinates for each school
    for coords in set_coords:
        sch_coords_0 = rng.choice(
            np.linspace(coords[0] - coords_range, coords[0] + coords_range, 200),
            n_sch_per_set,
            replace=False,
        )
        sch_coords_1 = rng.choice(
            np.linspace(coords[1] - coords_range, coords[1] + coords_range, 200),
            n_sch_per_set,
            replace=False,
        )
        sch_coords = np.vstack((sch_coords_0, sch_coords_1)).T
        all_sch_coords.append(sch_coords)
    all_sch_coords = np.vstack(all_sch_coords)

    n_sch = n_sch_per_set * len(set_coords)
    w_mat = np.ones((n_sch, n_sch)) # stores normalized interaction scores
    p_mat = np.ones((n_sch, n_sch)) # stores edge probabilities
    sch_to_set = np.repeat(np.arange(len(set_coords)), n_sch_per_set) # maps schools to settlements
    diff_sch = ~np.eye(n_sch, dtype=bool) # mask of pairs of schools in different settlements

    # Calculate interaction scores as the 
    # inverse of the gamma^th power of the distance between schools
    pdists = squareform(pdist(all_sch_coords, metric="euclidean"))
    pdists_normed = pdists / np.max(pdists)
    intxn_scores = pdists_normed[diff_sch] ** -gamma
    w_mat[diff_sch] = intxn_scores / np.max(intxn_scores)

    # Calculate edge probabilities based on the interaction scores
    # and whether individuals are in the same school, same settlement, or different settlements
    same_set = sch_to_set[:, np.newaxis] == sch_to_set[np.newaxis, :]
    p_mat[~diff_sch & same_set] = p_same_sch
    p_mat[diff_sch & same_set] = p_same_set_diff_sch * w_mat[diff_sch & same_set]
    p_mat[diff_sch & ~same_set] = p_diff_set_diff_sch * w_mat[diff_sch & ~same_set]

    # Generate the network 
    G = nx.stochastic_block_model(block_sizes, p_mat, seed=seed)
    A = nx.to_numpy_array(G, dtype=np.float32)

    return A, pdists


def gen_kenya_data(
    set_mus: np.ndarray,
    n_sch_per_set: int,
    n_stu_per_sch: int,
    sigma_sch_in_set: float,
    sigma_stu_in_sch: float,
    A: np.ndarray,
    beta: np.ndarray,
    tau_size: float,
    sigma: float,
    seed: int=42
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate covariates and potential outcomes for the simulated experiment with interference

    Sample individual-level covariates from a nested model with settlement- and school-specific means
    Outcomes are generated as a linear function of covariates with an additive treatment effect

    Args:
        - set_mus: mean of the random effects for each settlement
        - n_sch_per_set: number of schools per settlement
        - n_stu_per_sch: number of students per school
        - sigma_sch_in_set: standard deviation of the random effects for schools within a settlement
        - sigma_stu_in_sch: standard deviation of the random effects for students within a school
        - A: adjacency matrix of the network
        - beta: coefficients on the covariates in the outcome model
        - tau_size: effect size
        - sigma: standard deviation of noise
        - seed: random seed

    """
    rng = np.random.default_rng(seed)
    n_set = set_mus.shape[0]
    set_mus_sd = np.std(set_mus, axis=0)

    # Generate individual-level covariates
    X = []
    for set in range(n_set):
        for _ in range(n_sch_per_set):
            # Sample school-specific mean (within the settlement)
            sch_mu = rng.normal(set_mus[set], sigma_sch_in_set * set_mus_sd)
            # Sample individual-specific covariates (within the school)
            sch_Xs = [
                rng.normal(sch_mu, sigma_stu_in_sch * set_mus_sd)
                for _ in range(n_stu_per_sch)
            ]
            X.append(sch_Xs)
    X = np.vstack(X)
    X = np.matmul(A, X)

    # Generate potential outcomes
    eps = rng.normal(0, sigma, X.shape[0])
    y_0 = np.matmul(X, beta.transpose()).squeeze() + eps
    y_1 = y_0 + tau_size * np.std(y_0)

    # Store covariates and school and settlement labels
    X_df = pd.DataFrame(X)
    set_lbl = np.repeat(np.arange(n_set), n_sch_per_set * n_stu_per_sch)
    sch_in_set_lbl = np.tile(np.repeat(np.arange(n_sch_per_set), n_stu_per_sch), n_set)
    sch_lbl = np.repeat(np.arange(n_sch_per_set * n_set), n_stu_per_sch)
    X_df["set"] = set_lbl
    X_df["sch_in_set"] = sch_in_set_lbl
    X_df["sch"] = sch_lbl

    return y_0, y_1, X_df


def get_kenya_y_obs(
    y_0: np.ndarray,
    y_1: np.ndarray,
    z_accepted: np.ndarray,
    A: np.ndarray,
    q: float,
    cluster_lbls: np.ndarray,
) -> np.ndarray:
    """
    Generate observed outcomes for the simulated experiment with interference
    from the potential outcomes and the treatment assignments

    Applies interference through a fraction-neighborhood exposure model,
    where control individuals are exposed to treatment if at least a fraction q
    of their neighbors are treated

    Args:
        - y_0: potential outcomes under all-control
        - y_1: potential outcomes under all-treatment
        - z_accepted: binary treatment assignments
        - A: adjacency matrix of the network
        - q: threshold for exposure
        - cluster_lbls: cluster (school) labels for individuals
    """
    z_accepted_stu = np.vstack([z[cluster_lbls] for z in z_accepted])

    n_z1_nbrs = np.dot(A.T, z_accepted_stu.T).T
    n_nbrs = np.sum(A, axis=0)
    is_expo = (n_z1_nbrs >= (q * n_nbrs)) | (z_accepted_stu == 1)
    y_obs = y_0 * (1 - is_expo) + y_1 * is_expo

    return y_obs


def get_kenya_sch_y_obs(
    y_0: np.ndarray,
    y_1: np.ndarray,
    z_accepted: np.ndarray,
    A: np.ndarray,
    q: np.ndarray,
    cluster_lbls: np.ndarray,
) -> np.ndarray:
    """
    Generate school-level average observed outcomes for the simulated experiment with interference
    from the potential outcomes and the treatment assignments

    Args:
        - y_0: potential outcomes under all-control
        - y_1: potential outcomes under all-treatment
        - z_accepted: binary treatment assignments
        - A: adjacency matrix of the network
        - q: threshold for exposure
        - cluster_lbls: cluster (school) labels for individuals
    """

    # Get individual-level observed outcomes
    y_obs = get_kenya_y_obs(y_0, y_1, z_accepted, A, q, cluster_lbls)

    # Calculate the mean of the individual-level observed outcomes for each school
    cluster_range = np.arange(z_accepted.shape[1])
    cluster_mask = cluster_lbls == cluster_range[:, None]
    sch_y_obs = (
        np.matmul(y_obs, cluster_mask.transpose())
        / np.sum(cluster_mask, axis=1)[None, :]
    )

    return sch_y_obs
