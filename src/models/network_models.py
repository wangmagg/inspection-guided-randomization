import numpy as np
import networkx as nx

from typing import List, Union

class EuclideanDistPowerDecayIntxn:
    """
    Power decay interaction model based on the Euclidean distance between units

    Args:
        gamma: power decay parameter
    """
    def __init__(self, gamma):
        self.name = f'euclidean-dist-power-decay-gamma-{gamma}'
        self.gamma = gamma

    def __call__(self, 
                 coords: np.ndarray, 
                 pairwise_dists: np.ndarray) -> np.ndarray:
        """
        Get pairwise interaction scores between units

        Args:
            - coords: unit latitude, longitude coordinates (n_units, 2)
            - pairwise_dists: pairwise distances between units (n_units, n_units)

        Returns:
            Matrix of interaction scores (n_units, n_units)
        """
        dists = pairwise_dists / pairwise_dists.max()
        n = coords.shape[0]

        W = np.ones((n, n))
        off_diag_mask = ~np.eye(n, dtype=bool)

        # Compute interaction "score" as inverse power of distance
        intxn_score = dists[off_diag_mask] ** -self.gamma

        # Normalize interaction scores with respect to the maximum score
        W[off_diag_mask] = intxn_score / intxn_score.max()

        return W

class StochasticBlock:
    """
    Stochastic block model for network between individuals

    Args:
        wi_p: within cluster edge probability
        bw_p: between cluster edge probability
        seed: random seed
    """
    def __init__(self, 
                 wi_p: float, 
                 bw_p: float, 
                 seed: int):
        self.wi_p = wi_p
        self.bw_p = bw_p
        self.seed = seed
        self.name = f'sb_wip-{wi_p:.2f}_bwp-{bw_p:.2f}'

    def __call__(self, 
            n_clusters: int,
            cluster_sizes: Union[List, np.ndarray],
            intxn_mdl: callable,
            cluster_coords: np.ndarray,
            pairwise_dists: np.ndarray) -> tuple[nx.Graph, np.ndarray]:
        
        """
        Generate a network using the stochastic block model

        Args:
            n_clusters: number of clusters
            cluster_sizes: size of each cluster
            intxn_mdl: interaction model for determining between cluster edge probabilities
            cluster_coords: cluster coordinates
            pairwise_dists: pairwise distances between clusters

        Returns:
            - G: networkx graph
            - A: adjacency matrix
        """
        
        # Initialize the density matrix
        # Diagonal elements are the within cluster edge probabilities
        density = np.zeros((n_clusters, n_clusters))
        np.fill_diagonal(density, self.wi_p)

        # Calculate between cluster edge probabilities using interaction model
        bw_p = self.bw_p * intxn_mdl(cluster_coords, pairwise_dists)

        # Off-diagonal elements are the between cluster edge probabilities
        off_diag_mask = ~np.eye(n_clusters, dtype=bool)
        density[off_diag_mask] = bw_p[off_diag_mask]

        # Generate the network
        G = nx.stochastic_block_model(cluster_sizes, density, seed=self.seed)
        A = nx.to_numpy_array(G, dtype=np.float32)

        return G, A