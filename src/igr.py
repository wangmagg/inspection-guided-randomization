import numpy as np
import time
from tqdm import tqdm

from src.igr_enhancements import run_genetic_alg, add_mirrors


def igr_enumeration(n: int, n_arms: int, n_enum: int, seed: int = 42) -> np.ndarray:
    """
    Enumerate treatment allocations with an equal number of individuals per arm
    (i.e. sample from complete randomization assignment mechanism)

    Args:
        - n: number of individuals
        - n_arms: number of arms
        - n_enum: number of enumerated treatment allocations
        - seed: random seed
    """
    rng = np.random.default_rng(seed)

    # Calculate number of individuals per arm
    n_per_arm = n // n_arms
    n_rem = n % n_arms
    # If n is not divisible by n_arms, add the remainder to a random arm
    if n_rem > 0:
        n_per_arm = np.repeat(n_per_arm, n_arms)
        which_arm_to_add = rng.choice(n_arms, size=1)[0]
        n_per_arm[which_arm_to_add] += n_rem

    # Enumerate a pool of candidate treatment allocations
    z_pool = np.zeros((n_enum, n), dtype=np.float32)
    for i in tqdm(range(n_enum)):
        z = np.repeat(np.arange(n_arms, dtype=np.float32), n_per_arm)
        rng.shuffle(z)
        z_pool[i] = z

    return z_pool


def igr_paired_gfr_enumeration(
    n: int,
    n_arms: int,
    n_enum: int,
    rhos: np.ndarray,
    attr_arr: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Enumerate treatment allocations in a paired group formation randomization (GFR) design
    Args:
        - n: number of individuals
        - n_arms: number of arms
        - n_enum: number of enumerated treatment allocations
        - rhos: proportion of individuals with salient attribute in each group
        - attr_arr: boolean array indicating which individuals have salient attribute
        - seed: random seed
    """
    rng = np.random.default_rng(seed)

    # Number of groups is number of arms times number of compositions
    n_groups = n_arms * len(rhos)

    # Calculate number of individuals per group
    # If n is not divisible by n_groups, add the remainder to a random group
    n_per_group = n // n_groups
    n_rem = n % n_groups
    if n_rem > 0:
        n_per_group = np.repeat(n_per_group, n_groups)
        which_group_to_add = rng.choice(n_groups, size=1)[0]
        n_per_group[which_group_to_add] += n_rem
    else:
        n_per_group = np.repeat(n_per_group, n_groups)

    # Calculate number of individuals with/without salient attribute that should be in each group
    rho_arr = np.repeat(rhos, n_arms)
    n_attr_on_per_group = np.array(
        [int(rho * n_per_group) for rho, n_per_group in zip(rho_arr, n_per_group)]
    )
    n_attr_off_per_group = n_per_group - n_attr_on_per_group

    # Create group assignments for individuals with/without salient attribute
    z_X_on = np.repeat(np.arange(n_groups, dtype=np.float32), n_attr_on_per_group)
    z_X_off = np.repeat(np.arange(n_groups, dtype=np.float32), n_attr_off_per_group)
    if np.sum(attr_arr) < np.sum(n_attr_on_per_group):
        z_X_on = z_X_on[: np.sum(attr_arr)]
        z_X_off = np.concatenate((z_X_off, z_X_on[np.sum(attr_arr) :]))
    elif np.sum(attr_arr) > np.sum(n_attr_on_per_group):
        z_X_on = np.concatenate((z_X_on, z_X_off[np.sum(attr_arr) :]))
        z_X_off = z_X_off[: np.sum(attr_arr)]

    z_pool = np.zeros((n_enum, n), dtype=np.float32)
    for i in range(n_enum):
        rng.shuffle(z_X_on)
        rng.shuffle(z_X_off)

        # Map group assignments to individuals
        z = np.zeros(n)
        z[attr_arr] = z_X_on
        z[~attr_arr] = z_X_off

        z_pool[i] = z

    return z_pool


def igr_restriction(
    z_pool,
    n_accept,
    random=False,
    metric_1=None,
    metric_2=None,
    scores_pool_1=None,
    scores_pool_2=None,
    agg_fn=None,
    mirror_type="all",
    genetic=False,
    metric_1_kwargs=None,
    metric_2_kwargs=None,
    agg_kwargs=None,
    genetic_kwargs=None,
    mirror_kwargs=None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Restrict the pool of treatment allocations to the top scoring allocations based on the fitness scores
    Args:
        - z_pool: pool of candidate treatment allocations from enumeration step
        - n_accept: number of allocations to accept
        - random: whether to randomly select the accepted allocations (used for benchmarking)
        - metric_1: first inspection metric
        - metric_2: second inspection metric
        - scores_pool_1: scores for metric_1
        - scores_pool_2: scores for metric_2
        - agg_fn: aggregation function for combining scores from metric_1 and metric_2
        - mirror_type: type of mirror to add to the accepted allocations
        - genetic: whether to run genetic algorithm to refine the pool of enumerated allocations before restriction
        - metric_1_kwargs: additional keyword arguments for metric_1
        - metric_2_kwargs: additional keyword arguments for metric_2
        - agg_kwargs: additional keyword arguments for agg_fn
        - genetic_kwargs: additional keyword arguments for genetic algorithm
        - mirror_kwargs: additional keyword arguments for mirror addition
    """

    # Calculate inspection metric scores for each candidate allocation
    # if not already provided
    if scores_pool_1 is None and metric_1 is not None:
        print(f"Evaluating {metric_1.__name__}...")
        t_1_start = time.time()
        scores_pool_1 = metric_1(z_pool=z_pool, **metric_1_kwargs)
        t_1_end = time.time()
        print(f"{metric_1.__name__}: {t_1_end - t_1_start:.2f} s")

    if scores_pool_2 is None and metric_2 is not None:
        print(f"Evaluating {metric_2.__name__}...")
        t_2_start = time.time()
        scores_pool_2 = metric_2(z_pool=z_pool, **metric_2_kwargs)
        t_2_end = time.time()
        print(f"{metric_2.__name__}: {t_2_end - t_2_start:.2f} s")

    # Aggregate scores from metric_1 and metric_2, if applicable
    if agg_fn is not None:
        scores_pool_agg = agg_fn(scores_pool_1, scores_pool_2, **agg_kwargs)
    else:
        scores_pool_agg = scores_pool_1

    # Run genetic algorithm to get a better pool of enumerated allocations
    if genetic:
        print("Running genetic algorithm...")
        z_pool, (scores_pool_1, scores_pool_2), scores_pool_agg = run_genetic_alg(
            z_pool=z_pool,
            metric_1=metric_1,
            scores_pool_1=scores_pool_1,
            metric_2=metric_2,
            scores_pool_2=scores_pool_2,
            agg_fn=agg_fn,
            metric_1_kwargs=metric_1_kwargs,
            metric_2_kwargs=metric_2_kwargs,
            agg_kwargs=agg_kwargs,
            **genetic_kwargs,
        )

    # Select the allocations to accept
    if random:
        accept_indices = np.arange(n_accept)
    else:
        accept_indices = np.argsort(scores_pool_agg)[:n_accept]

    z_pool_accepted = z_pool[accept_indices]
    scores_accepted_1 = scores_pool_1[accept_indices]
    if metric_2 is not None:
        scores_accepted_2 = scores_pool_2[accept_indices]
    else:
        scores_accepted_2 = None

    # Add mirrors to the accepted allocations
    print(f"Adding mirrors...")
    z_pool_accepted, (scores_accepted_1, scores_accepted_2) = add_mirrors(
        mirror_type=mirror_type,
        z_pool_accepted=z_pool_accepted,
        metric_1=metric_1,
        scores_1=scores_accepted_1,
        metric_2=metric_2,
        scores_2=scores_accepted_2,
        agg_fn=agg_fn,
        metric_1_kwargs=metric_1_kwargs,
        metric_2_kwargs=metric_2_kwargs,
        agg_kwargs=agg_kwargs,
        mirror_kwargs=mirror_kwargs,
    )
    
    # Return the accepted allocations and their scores
    scores_pool = (scores_pool_1, scores_pool_2)
    scores_accepted = (scores_accepted_1, scores_accepted_2)

    return z_pool_accepted, scores_pool, scores_accepted


def igr_randomization(z_pool_accepted: np.ndarray, seed=42):
    """
    Randomly select an allocation from the set of accepted allocations
    Args:
        - z_pool_accepted: accepted allocations
    """
    rng = np.random.default_rng(seed)
    obs_idx = rng.choice(np.arange(z_pool_accepted.shape[0]), size=1)

    return z_pool_accepted[obs_idx]
