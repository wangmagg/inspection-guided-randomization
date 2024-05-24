from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_kenya_adjacency(X, X_school, A, fig_dir):
    n = X_school.shape[0]
    mapping = X["school_id"].values

    W = np.zeros((n, n))
    for schl in range(n):
        indivs_in_schl_mask = mapping == schl
        indivs_in_schl_adj = np.sum(A[indivs_in_schl_mask,], axis=0)
        schl_adj = np.bincount(mapping, weights=indivs_in_schl_adj)
        W[schl,] = schl_adj

    W_mask = np.triu(np.ones_like(W, dtype=bool))
    W_mask[W.shape[0] :, :] = True

    save_path = fig_dir / f"school_adjacency.png"
    print(save_path)
    # plt.imsave(save_path, W)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(W, cmap=sns.color_palette("crest", as_cmap=True), mask=W_mask, ax=ax)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kenya_pairwise_dists(X_school, fig_dir):
    n = X_school.shape[0]
    pairwise_dists = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        i_coords = X_school.iloc[i][["coord_lat", "coord_lon"]]
        j_coords = X_school.iloc[j][["coord_lat", "coord_lon"]]
        dist = np.linalg.norm(i_coords - j_coords)
        pairwise_dists[i, j] = dist
        pairwise_dists[j, i] = dist

    # Set off diagonal to inverse of pairwise distances
    inv_pairwise_dists = np.zeros_like(pairwise_dists)
    off_diag_mask = ~np.eye(pairwise_dists.shape[0], dtype=bool)
    diag_mask = np.eye(pairwise_dists.shape[0], dtype=bool)
    inv_pairwise_dists[off_diag_mask] = 1 / pairwise_dists[off_diag_mask]
    inv_pairwise_dists[diag_mask] = np.max(inv_pairwise_dists)

    save_path = fig_dir / f"inverse_pairwise_dists.png"
    print(save_path)
    plt.imsave(save_path, inv_pairwise_dists)
    plt.close()


# def plot_cos_sim_v_dist(X, X_school, fig_dir):
#     n = X_school.shape[0]
#     pairwise_dists = np.zeros((n, n))
#     for i, j in combinations(range(n), 2):
#         i_coords = X_school.iloc[i][["coord_lat", "coord_lon"]]
#         j_coords = X_school.iloc[j][["coord_lat", "coord_lon"]]
#         dist = np.linalg.norm(i_coords - j_coords)
#         pairwise_dists[i, j] = dist
#         pairwise_dists[j, i] = dist
#     X = X.drop(
#         ["school", "settlement", "settlement_id", "coord_lat", "coord_lon"], axis=1
#     )
#     X = X.groupby("school_id").mean().to_numpy()
#     X_normed = X / np.linalg.norm(X, axis=1)[:, None]
#     X_sim = np.matmul(X_normed, X_normed.T)
#     pairwise_dists_off_diag = pairwise_dists[~np.eye(n, dtype=bool)]
#     X_sim_off_diag = X_sim[~np.eye(n, dtype=bool)]

#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     sns.scatterplot(x=pairwise_dists_off_diag, y=X_sim_off_diag, ax=ax, alpha=0.5, s=5)
#     ax.set_xlabel("Pairwise Distance", fontsize=16)
#     ax.set_ylabel("Cosine Similarity", fontsize=16)
#     ax.tick_params(axis="x", labelsize=14)
#     ax.tick_params(axis="y", labelsize=14)

#     save_path = fig_dir / f"cos_sim_v_dist.png"
#     print(save_path)
#     fig.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()


def plot_kenya_adj_v_dist(X, X_school, A, fig_dir):
    n = X_school.shape[0]
    pairwise_dists = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        i_coords = X_school.iloc[i][["coord_lat", "coord_lon"]]
        j_coords = X_school.iloc[j][["coord_lat", "coord_lon"]]
        dist = np.linalg.norm(i_coords - j_coords)
        pairwise_dists[i, j] = dist
        pairwise_dists[j, i] = dist

    mapping = X["school_id"].values
    W = np.zeros((n, n))
    for schl in range(n):
        indivs_in_schl_mask = mapping == schl
        indivs_in_schl_adj = np.sum(A[indivs_in_schl_mask,], axis=0)
        schl_adj = np.bincount(mapping, weights=indivs_in_schl_adj)
        W[schl,] = schl_adj

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(x=pairwise_dists.flatten(), y=W.flatten(), ax=ax)
    ax.set_xlabel("Pairwise Distance", fontsize=16)
    ax.set_ylabel("Adjacency", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    save_path = fig_dir / f"adj_v_dist.png"
    print(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
