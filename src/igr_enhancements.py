import numpy as np
from tqdm import tqdm
from typing import Optional
import argparse

def crossover(
    z_par1: np.ndarray, z_par2: np.ndarray, cross_k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    k-point crossover for a single pair of parent allocations

    Args:
        - z_par1: parent 1 treatment allocation
        - z_par2: parent 2 treatment allocation

    Returns:
        tuple of children treatment allocations with crossover applied
    """
    for _ in range(cross_k):
        cross_loc = rng.integers(len(z_par1))

        z_par1_rhs = z_par1[cross_loc:].copy()
        z_par2_rhs = z_par2[cross_loc:].copy()

        # swap right-hand sides
        z_par1[cross_loc:] = z_par2_rhs
        z_par2[cross_loc:] = z_par1_rhs

    return z_par1, z_par2

def crossover_mult(
    z_par1_pool: np.ndarray, z_par2_pool: np.ndarray, cross_k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    
    for i, (z_par1, z_par2) in enumerate(zip(z_par1_pool, z_par2_pool)):
        z_par1, z_par2 = crossover(z_par1, z_par2, cross_k, rng)
        z_par1_pool[i] = z_par1
        z_par2_pool[i] = z_par2
    
    return z_par1_pool, z_par2_pool

def mutation(z_pool: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """
    Introduce mutations in pool of allocations at specified rate

    Args:
        - z_pool: pool of treatment allocations
        - rate: mutation rate

    Returns:
        pool of treatment allocations with mutations applied
    """
    max_z = np.max(z_pool)
    mut_loc = rng.binomial(1, rate, size=z_pool.shape)
    z_pool[mut_loc == 1] = (
        z_pool[mut_loc == 1] + rng.integers(0, max_z, size=np.sum(mut_loc == 1))
    ) % (max_z + 1)

    return z_pool


def tournament(scores: np.ndarray, tourn_size: int, rng: np.random.Generator) -> int:
    """
    Use a tournament to choose a parent to mate

    Args:
        - scores: fitness scores for each allocation
        - tourn_size: number of allocations to consider in each tournament

    Returns:
        index of winning allocation
    """
    tourn_block_idxs = rng.choice(np.arange(len(scores)), tourn_size, replace=False)
    winner = tourn_block_idxs[np.argmin(scores[tourn_block_idxs])]

    return winner

def make_new_generation(
        scores: np.ndarray, 
        z_pool: np.ndarray, 
        z_bincounts: np.ndarray, 
        tourn_size: int, 
        cross_rate: float, 
        cross_k: int, 
        mut_rate: float,
        eps: float, 
        rng: np.random.Generator) -> np.ndarray:
    """
    Create a new generation of treatment allocations using genetic algorithms
    Args:
        - scores: fitness scores for each allocation
        - z_pool: pool of treatment allocations
        - z_bincounts: distribution of individuals to each arm in the original pool of allocations
        - tourn_size: number of allocations to consider in each tournament
        - cross_rate: probability of crossover
        - cross_k: number of crossover points
        - mut_rate: probability of mutation
        - eps: tolerance for deviation from original distribution of individuals to treatments (z_bincounts)
        - rng: random number generator
    """

    # use tournament selection to make mating pool
    winners = np.array(
        [tournament(scores, tourn_size, rng) for _ in range(len(scores))]
    )
    z_pool_mate = z_pool[winners, :]
    rng.shuffle(z_pool_mate)

    # split mating pool into two sets of parents
    mate_split = z_pool_mate.shape[0] // 2
    z_pool_par1 = z_pool_mate[:mate_split, :]
    z_pool_par2 = z_pool_mate[mate_split:, :]

    # identify which mating pairs will have crossover
    which_cross = rng.binomial(1, cross_rate, z_pool_par1.shape[0])
    z_pool_chil1 = z_pool_par1.copy()
    z_pool_chil2 = z_pool_par2.copy()

    # if a pair is designated as a crossover pair, make children with crossover
    # otherwise, children are exact copies of parents
    z_pool_chil1_cross, z_pool_chil2_cross = crossover_mult(z_pool_par1[which_cross == 1, :], z_pool_par2[which_cross == 1, :], cross_k, rng)
    z_pool_chil1[which_cross == 1, :] = z_pool_chil1_cross
    z_pool_chil2[which_cross == 1, :] = z_pool_chil2_cross

    # introduce mutations in children
    z_pool_chil = np.vstack((z_pool_chil1, z_pool_chil2))
    z_pool_chil_mut = mutation(z_pool_chil, mut_rate, rng)

    # remove children where distribution of treatment assigments is too dissimilar to original pool
    z_pool_chil_mut_bin_counts = np.vstack(
        [np.bincount(z.astype(int)) / z.shape[0] for z  in z_pool_chil_mut]
    )
    z_pool_chil_mut_within_eps = (
        np.all(abs(z_pool_chil_mut_bin_counts / z_bincounts - 1) <= eps, axis=1)
    )

    # combine parent and child generations
    z_pool_new = np.vstack((z_pool, z_pool_chil_mut[z_pool_chil_mut_within_eps, :]))
    z_pool_new = np.unique(z_pool_new, axis=0)

    return z_pool_new

def run_genetic_alg(
    z_pool: np.ndarray,
    scores_pool_1: np.ndarray,
    metric_1: callable,
    tourn_size: int,
    cross_k: int,
    cross_rate: float,
    mut_rate: float,
    genetic_iters: int,
    scores_pool_2: Optional[np.ndarray] = None,
    metric_2: Optional[callable] = None,
    agg_fn: Optional[callable] = None,
    seed: int = 42,
    eps: float = 0.05,
    metric_1_kwargs = None,
    metric_2_kwargs = None,
    agg_kwargs = None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Run genetic algorithm to enumerate a better candidate pool of treatment allocations
    Args:
        - z_pool: initial enumerated pool of treatment allocations
        - scores_pool_1: scores for initial pool from first inspection metric
        - metric_1: first inspection metric 
        - tourn_size: number of allocations to consider in each tournament
        - cross_k: number of crossover points
        - cross_rate: probability of crossover
        - mut_rate: probability of mutation
        - genetic_iters: number of iterations to run genetic algorithm
        - scores_pool_2: scores for initial pool from second inspection metric
        - metric_2: second inspection metric
        - agg_fn: function to aggregate scores from two metrics
        - seed: random seed
        - eps: tolerance for deviation from original distribution of individuals to treatments
        - metric_1_kwargs: additional keyword arguments for metric_1
        - metric_2_kwargs: additional keyword arguments for metric_2
        - agg_kwargs: additional keyword arguments for agg_fn
    """
    rng = np.random.default_rng(seed)

    init_pool_size = z_pool.shape[0]

    # Aggregate scores from two metrics if second metric is provided
    scores_pool = scores_pool_1
    if metric_2 is not None:
        scores_pool = agg_fn(scores_pool_1, scores_pool_2, **agg_kwargs)

    # Calculate distribution of individuals to treatments in original pool
    z_bincounts = np.mean(
        np.vstack([np.bincount(z.astype(int)) / z.shape[0] for z in z_pool]), axis=0
    )

    # Run genetic algorithm
    for _ in tqdm(range(genetic_iters)):
        z_pool_new = make_new_generation(
            scores_pool,
            z_pool,
            z_bincounts,
            tourn_size,
            cross_rate,
            cross_k,
            mut_rate,
            eps,
            rng,
        )

        # Get scores for all allocations in parent and child generations
        scores_pool_1 = metric_1(z_pool=z_pool_new, **metric_1_kwargs)
        scores_pool = scores_pool_1
        if metric_2 is not None:
            scores_pool_2 = metric_2(z_pool=z_pool_new, **metric_2_kwargs)
            scores_pool = agg_fn(scores_pool, scores_pool_2, **agg_kwargs)

        # Keep the best-scoring allocations
        keep = np.argsort(scores_pool)[:init_pool_size]
        z_pool = z_pool_new[keep, :]
        scores_pool = scores_pool[keep]
        scores_pool_1 = scores_pool_1[keep]
        if metric_2 is not None:
            scores_pool_2 = scores_pool_2[keep]

    if metric_2 is not None:
        return z_pool, (scores_pool_1, scores_pool_2), scores_pool
    else:
        return z_pool, (scores_pool_1, None), scores_pool
    
def get_genetic_kwargs(args: argparse.Namespace) -> dict:
    """
    Get keyword arguments for genetic algorithm from argparse arguments
    Args:
        - args: argparse arguments
    """
    genetic_args = {
        "genetic_iters": args.genetic_iters,
        "tourn_size": args.tourn_size,
        "cross_k": args.cross_k,
        "cross_rate": args.cross_rate,
        "mut_rate": args.mut_rate,
        "eps": args.eps,
        "seed": args.seed
    }
    return genetic_args

def _get_mirrors(z: np.ndarray, n_trt_arms: int) -> np.ndarray:
    """
    Get mirrors of treatment allocations
    Args:
        - z: treatment allocation
        - n_trt_arms: number of treated arms
    """
    mirrors = np.zeros((n_trt_arms, len(z)), dtype=np.float32)
    for i in range(n_trt_arms):
        mirrors[i] = z.copy()
        mirrors[i][z == (i + 1)] = 0
        mirrors[i][z == 0] = i + 1
    return mirrors

def _get_gfr_mirrors(z: np.ndarray, same_rho_pairs: np.ndarray) -> np.ndarray:
    """
    Get mirrors of treatment allocations for group formation randomization
    Args:
        - z: treatment allocation
        - same_rho_pairs: pairs of treatment arms with the same rho (salient attribute composition)
            but different z
    """
    n_trt_arms = same_rho_pairs.shape[1] - 1
    mirrors = np.zeros((n_trt_arms, len(z)), dtype=np.float32)
    for i in range(n_trt_arms):
        mirrors[i] = z.copy()
        for pair in same_rho_pairs:
            mirrors[i][z == pair[1]] = pair[0]
            mirrors[i][z == pair[0]] = pair[1]
    return mirrors
    
def get_mirrors(z_pool_accepted: np.ndarray, gfr:bool=False, same_rho_pairs:np.ndarray=None) -> np.ndarray:
    """
    Get mirrors of treatment allocations for all accepted allocations
    Args:
        - z_pool_accepted: accepted treatment allocations
        - gfr: whether group formation randomization is used
        - same_rho_pairs: pairs of treatment arms with the same rho (salient attribute composition)
            but different z
    """
    if gfr:
        return np.vstack([_get_gfr_mirrors(z, same_rho_pairs) for z in z_pool_accepted])
    else:
        n_trt_arms = int(z_pool_accepted.max())
        return np.vstack([_get_mirrors(z, n_trt_arms) for z in z_pool_accepted])
        
def add_all_mirrors(
    z_pool_accepted: np.ndarray, 
    z_pool_accepted_mirrors: np.ndarray, 
    scores_1:np.ndarray=None, 
    scores_mirrors_1:np.ndarray=None, 
    scores_2:np.ndarray=None,
    scores_mirrors_2:np.ndarray=None
)-> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:  
    """
    Add all mirrors of accepted allocations, regardless of their scores
    Args:
        - z_pool_accepted: accepted treatment allocations
        - z_pool_accepted_mirrors: mirrors of accepted allocations
        - scores_1: scores from first inspection metric
        - scores_mirrors_1: scores from first inspection metric for mirrors
        - scores_2: scores from second inspection metric
        - scores_mirrors_2: scores from second inspection metric for mirrors
    """
    # Determine how many original and mirror allocations to keep based on the 
    # total number that should be accepted
    n_mirr_per_orig = z_pool_accepted_mirrors.shape[0] // z_pool_accepted.shape[0]   
    cutoff_idx_for_orig = z_pool_accepted.shape[0] // (n_mirr_per_orig + 1)
    cutoff_idx_for_mirr = cutoff_idx_for_orig * n_mirr_per_orig
    z_pool_accepted = np.vstack(
        [
            z_pool_accepted[:cutoff_idx_for_orig],
            z_pool_accepted_mirrors[:cutoff_idx_for_mirr],
        ]
    )
    print(f"Original: {cutoff_idx_for_orig}, Mirrors: {cutoff_idx_for_mirr}")
    if scores_1 is not None:
        scores_1 = np.hstack([scores_1[:cutoff_idx_for_orig], scores_mirrors_1[:cutoff_idx_for_mirr]])
    if scores_2 is not None:
        scores_2 = np.hstack([scores_2[:cutoff_idx_for_orig], scores_mirrors_2[:cutoff_idx_for_mirr]])
    return z_pool_accepted, (scores_1, scores_2)

def add_good_mirrors(
    z_pool_accepted: np.ndarray,
    z_pool_accepted_mirrors: np.ndarray,
    scores_1: np.ndarray,
    scores_mirrors_1: np.ndarray,
    scores_2:np.ndarray=None,
    scores_mirrors_2:np.ndarray=None,
    agg_fn:np.ndarray=None,
    agg_kwargs:np.ndarray=None
):
    """
    Add mirrors of accepted allocations with scores that are not worse than the worse-scoring original allocation
    Args:
        - z_pool_accepted: accepted treatment allocations
        - z_pool_accepted_mirrors: mirrors of accepted allocations
        - scores_1: scores from first inspection metric
        - scores_mirrors_1: scores from first inspection metric for mirrors
        - scores_2: scores from second inspection metric
        - scores_mirrors_2: scores from second inspection metric for mirrors
        - agg_fn: function to aggregate scores from two metrics
        - agg_kwargs: additional keyword arguments for agg_fn
    """
    n_accepted = z_pool_accepted.shape[0]
    scores_all_1 = np.hstack([scores_1, scores_mirrors_1])
    if scores_2 is not None:
        scores_all_2 = np.hstack([scores_2, scores_mirrors_2])
        scores_all = agg_fn(scores_all_1, scores_all_2, **agg_kwargs)
    else:
        scores_all = scores_all_1

    # Determine the maximum (worst) score of the accepted allocations
    # and keep mirrors with scores that are not worse
    max_accepted_score = np.max(scores_all[:n_accepted])
    mirrors_good_mask = scores_all[n_accepted:] <= max_accepted_score

    # Make a map of indices of mirrors to indices of original allocations
    map_mirr_to_orig_idx = np.repeat(np.arange(n_accepted), int(z_pool_accepted.max()))
    if sum(mirrors_good_mask) > 0:
        # Determine how many original and mirror allocations to keep based on total
        # number that should be accepted
        z_pool_accepted_mirrors_good = z_pool_accepted_mirrors[mirrors_good_mask]
        map_mirr_to_orig_idx_good = map_mirr_to_orig_idx[mirrors_good_mask]
        bin_map = np.bincount(map_mirr_to_orig_idx_good)
        bin_map_csum = np.cumsum(bin_map)
        bin_map_add_orig_csum = np.cumsum(bin_map + 1)
        where_cutoff = np.where(bin_map_add_orig_csum >= n_accepted)

        if len(where_cutoff[0]) == 0:
            # all kept mirrors can be added
            cutoff_idx_for_mirr = bin_map_csum[-1]
            cutoff_idx_for_orig = n_accepted - cutoff_idx_for_mirr
        else:
            # only add mirrors up to the cutoff
            cutoff_idx_for_orig = where_cutoff[0][0]
            cutoff_idx_for_mirr = bin_map_csum[cutoff_idx_for_orig]

        # Combine original and mirror allocations
        z_pool_accepted = np.vstack(
            [
                z_pool_accepted[:cutoff_idx_for_orig],
                z_pool_accepted_mirrors_good[:cutoff_idx_for_mirr],
            ]
        )
        # Combine original and mirror scores
        scores_1 = np.hstack([scores_1[:cutoff_idx_for_orig], scores_mirrors_1[:cutoff_idx_for_mirr]])
        if scores_2 is not None:
            scores_2 = np.hstack([scores_2[:cutoff_idx_for_orig], scores_mirrors_2[:cutoff_idx_for_mirr]])
        print(f"Original: {cutoff_idx_for_orig}, Mirrors: {cutoff_idx_for_mirr}")
    else:
        print("No accepted mirrors")

    return z_pool_accepted, (scores_1, scores_2)


def add_mirrors(
    mirror_type: str,
    z_pool_accepted: np.ndarray,
    metric_1:callable=None,
    scores_1:np.ndarray=None,
    metric_2:callable=None,
    scores_2:np.ndarray=None,
    agg_fn:callable=None,
    metric_1_kwargs:dict=None,
    metric_2_kwargs:dict=None,
    agg_kwargs:dict=None,
    mirror_kwargs:dict=None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Add mirrors of accepted allocations
    Args:
        - mirror_type: which type of addition to use ("all", "good", or "none")
        - z_pool_accepted: accepted treatment allocations
        - metric_1: first inspection metric
        - scores_1: scores from first inspection metric
        - metric_2: second inspection metric
        - scores_2: scores from second inspection metric
        - agg_fn: function to aggregate scores from two metrics
        - metric_1_kwargs: additional keyword arguments for metric_1
        - metric_2_kwargs: additional keyword arguments for metric_2
        - agg_kwargs: additional keyword arguments for agg_fn
        - mirror_kwargs: additional keyword arguments for getting mirrors
    """

    # Get mirrors and their inspection metric scores
    if mirror_kwargs is None:
        z_pool_accepted_mirrors = get_mirrors(z_pool_accepted)
    else:
        z_pool_accepted_mirrors = get_mirrors(z_pool_accepted, **mirror_kwargs)

    scores_mirrors_1 = None
    scores_mirrors_2 = None
    
    if metric_1 is not None:
        scores_mirrors_1 = metric_1(z_pool=z_pool_accepted_mirrors, **metric_1_kwargs)
    if metric_2 is not None:
        scores_mirrors_2 = metric_2(z_pool=z_pool_accepted_mirrors, **metric_2_kwargs)

    # Add mirrors based on the specified type
    if mirror_type == "all":
        z_pool_accepted, (scores_1, scores_2) = add_all_mirrors(
            z_pool_accepted,
            z_pool_accepted_mirrors,
            scores_1,
            scores_mirrors_1,
            scores_2,
            scores_mirrors_2
        )
    elif mirror_type == "good":
        z_pool_accepted, (scores_1, scores_2) = add_good_mirrors(
            z_pool_accepted,
            z_pool_accepted_mirrors,
            scores_1,
            scores_mirrors_1,
            scores_2,
            scores_mirrors_2,
            agg_fn=agg_fn,
            agg_kwargs=agg_kwargs
        )
    elif mirror_type == "none":
        pass
    else:
        raise ValueError(f"Invalid mirror type: {mirror_type}. Must be 'all' or 'good'")
    
    return z_pool_accepted, (scores_1, scores_2)