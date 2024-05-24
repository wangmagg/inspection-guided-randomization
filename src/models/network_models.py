import numpy as np
import networkx as nx
import pandas as pd

from typing import List, Union, Optional


class EuclideanDistPowerDecayIntxn:
    """
    Power decay interaction model based on the Euclidean distance between units

    Args:
        gamma: power decay parameter
    """
    def __init__(self, gamma):
        self.name = f'euclidean-dist-power-decay-gamma-{gamma}'
        self.gamma = gamma
        self.plotting_name = f'gamma={gamma}'

    def __call__(self, 
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
        n = dists.shape[0]

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
            cluster_sizes: Optional[Union[List, np.ndarray]]=None,
            n: Optional[int]=None,
            intxn_mdl: Optional[callable]=None,
            pairwise_dists: Optional[np.ndarray]=None) -> tuple[nx.Graph, np.ndarray]:
        
        """
        Generate a network using the stochastic block model

        Args:
            n_clusters: number of clusters
            cluster_sizes: size of each cluster
            intxn_mdl: interaction model for determining between cluster edge probabilities
            pairwise_dists: pairwise distances between clusters

        Returns:
            - G: networkx graph
            - A: adjacency matrix
        """

        # Initialize the density matrix
        # Diagonal elements are the within cluster edge probabilities
        density = np.zeros((n_clusters, n_clusters))
        np.fill_diagonal(density, self.wi_p)

        # Off-diagonal elements are the between cluster edge probabilities
        off_diag_mask = ~np.eye(n_clusters, dtype=bool)

        # Calculate between cluster edge probabilities using interaction model if provided
        if intxn_mdl is not None:
            bw_p = self.bw_p * intxn_mdl(pairwise_dists)
            density[off_diag_mask] = bw_p[off_diag_mask]
        else:
            density[off_diag_mask] = self.bw_p

        if cluster_sizes is None:
            if n is None:
                raise ValueError('Either cluster_sizes or n must be provided')
            
            cluster_size = n // n_clusters
            rem_size = n % n_clusters
            cluster_sizes = np.array([cluster_size for _ in range(n_clusters)])
            cluster_sizes[n_clusters-1] += rem_size

        # Generate the network
        G = nx.stochastic_block_model(cluster_sizes, density, seed=self.seed)
        A = nx.to_numpy_array(G, dtype=np.float32)

        return G, A
    
class TwoLevelNestedStochasticBlock:
    """
    Stochastic block model for network between individuals
    with nested clusters

    Args:
        p_arr: edge probabilities for each level of the nested clusters 
        seed: random seed
    """

    def __init__(self, 
                 p_same_in: List[float], 
                 p_diff_in_same_out: List[float],
                 p_diff_in_diff_out: List[float],
                 inner_to_outer_map: np.ndarray,
                 seed: int):
        self.p_same_in = p_same_in
        self.p_diff_in_same_out = p_diff_in_same_out
        self.p_diff_in_diff_out = p_diff_in_diff_out
        self.inner_to_outer_map = inner_to_outer_map
        self.seed = seed

        if p_diff_in_diff_out < 0.01:
            self.name = f'nested-2lvl-sb_psi-{p_same_in:.2f}_pdiso-{p_diff_in_same_out:.2f}_pdido-{p_diff_in_diff_out:.3f}'
            self.plotting_name = f'Nested SBM\np=({p_same_in:.2f}, {p_diff_in_same_out:.2f}, {p_diff_in_diff_out:.3f})'
        else:
            self.name = f'nested-2lvl-sb_psi-{p_same_in:.2f}_pdiso-{p_diff_in_same_out:.2f}_pdido-{p_diff_in_diff_out:.2f}'
            self.plotting_name = f'Nested SBM\np=({p_same_in:.2f}, {p_diff_in_same_out:.2f}, {p_diff_in_diff_out:.2f})'

    def __call__(self, 
            n_clusters: int,
            cluster_sizes: Union[List, np.ndarray],
            intxn_mdl: callable,
            pairwise_dists: np.ndarray) -> tuple[nx.Graph, np.ndarray]:
        
        """
        Generate a network using the stochastic block model

        Args:
            n_clusters: number of clusters at each level
            cluster_sizes: size of each cluster at each level
            intxn_mdl: interaction model for determining between cluster edge probabilities
            pairwise_dists: pairwise distances between clusters at each level

        Returns:
            - G: networkx graph
            - A: adjacency matrix
        """
        
        # Initialize the density matrix
        # Diagonal elements are the within cluster edge probabilities
        density = np.zeros((n_clusters, n_clusters))
        np.fill_diagonal(density, self.p_same_in)

        # Get mask of inner clusters that belong to the same outer clusters
        same_outer_mask = self.inner_to_outer_map[:, None] == self.inner_to_outer_map[None, :]

        # Calculate between cluster edge probabilities using interaction model
        intxn_scores = intxn_mdl(pairwise_dists)

        # Off-diagonal elements are the between cluster edge probabilities
        off_diag_mask = ~np.eye(n_clusters, dtype=bool)
        density[off_diag_mask & same_outer_mask] = self.p_diff_in_same_out * intxn_scores[off_diag_mask & same_outer_mask]
        density[off_diag_mask & ~same_outer_mask] = self.p_diff_in_diff_out * intxn_scores[off_diag_mask & ~same_outer_mask]

        # Generate the network
        G = nx.stochastic_block_model(cluster_sizes, density, seed=self.seed)
        A = nx.to_numpy_array(G, dtype=np.float32)

        return G, A

class ErdosRenyi:
    def __init__(self, p_er, seed):
        self.p = p_er
        self.seed = seed
        self.name = f'er_p-{p_er:.2f}'
    def __call__(self, n):
        G = nx.erdos_renyi_graph(n=n, p=self.p, seed=self.seed)
        A = nx.to_numpy_array(G)
        return G, A

class WattsStrogatz:
    def __init__(self, k, p_ws, seed):
        self.k = k
        self.p_ws = p_ws
        self.seed = seed
        self.name = f'ws_k-{k}_p-{p_ws:.2f}'
    def __call__(self, n):
        G = nx.watts_strogatz_graph(n=n, k=self.k, p=self.p_ws, seed=self.seed)
        A = nx.to_numpy_array(G)
        return G, A
    
class BarabasiAlbert:
    def __init__(self, m, seed):
        self.m = m
        self.seed = seed
        self.name = f'ba_m-{m}'
    def __call__(self, n):
        G = nx.barabasi_albert_graph(n=n, m=self.m, seed=self.seed)
        A = nx.to_numpy_array(G)
        return G, A