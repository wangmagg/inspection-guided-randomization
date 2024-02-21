import numpy as np

def crossover(z_par1: np.ndarray, 
              z_par2: np.ndarray, 
              cross_k: int, 
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
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

        # get right-hand side of crossover point in both parents
        z_par1_rhs = np.copy(z_par1[cross_loc:]) 
        z_par2_rhs = np.copy(z_par2[cross_loc:]) 

        # swap right-hand sides 
        z_par1[cross_loc:] = z_par2_rhs 
        z_par2[cross_loc:] = z_par1_rhs

    return z_par1, z_par2

def mutation(z_pool: np.ndarray, 
             rate: float, 
             rng: np.random.Generator) -> np.ndarray:
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
    z_pool[mut_loc == 1] = (z_pool[mut_loc == 1] + rng.integers(0, max_z, size=np.sum(mut_loc == 1))) % (max_z + 1)

    return z_pool

def tournament(scores: np.ndarray, 
               tourn_size: int, 
               rng: np.random.Generator) -> int:
    """
    Use a tournament to choose a parent to mate

    Args:
        - scores: fitness scores for each allocation
        - tourn_size: number of allocations to consider in each tournament

    Returns:
        index of winning allocation
    """
    tourn_block_idxs = rng.choice(np.arange(len(scores)), tourn_size, replace=False)
    tourn_block_scores = scores[tourn_block_idxs]
    winner = tourn_block_idxs[np.argmin(tourn_block_scores)]

    return winner

def run_genetic_alg(z_pool: np.ndarray, 
                    fitness_fn, 
                    tourn_size: int, 
                    cross_k: int, 
                    cross_rate: float, 
                    mut_rate: float, 
                    genetic_iters: int, 
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Run genetic algorithm to optimize treatment allocation pool

    Args:
        - z_pool: initial pool of treatment allocations
        - fitness_fn: function to evaluate fitness of each allocation
        - tourn_size: number of allocations to consider in each tournament
        - cross_k: number of crossover points
        - cross_rate: probability of crossover
        - mut_rate: probability of mutation
        - genetic_iters: number of iterations to run genetic algorithm

    Returns:
        optimized pool of treatment allocations and their fitness scores
    """
    init_pool_size = z_pool.shape[0]

    for _ in range(genetic_iters):
        scores = fitness_fn(z_pool) + 2 * np.abs(0.5 - np.mean(z_pool, axis=1))

        # use tournament selection to make mating pool
        winners = np.array([tournament(scores, tourn_size, rng) for _ in range(len(scores))])
        z_pool_mate = z_pool[winners, :]
        rng.shuffle(z_pool_mate)

        # split mating pool into two sets of parents
        mate_split = z_pool_mate.shape[0] // 2 
        z_pool_par1 = z_pool_mate[ :mate_split, :]
        z_pool_par2 = z_pool_mate[mate_split:, :]
        
        # identify which mating pairs will have crossover
        which_cross = rng.binomial(1, cross_rate, z_pool_par1.shape[0])
        z_pool_par1_cross = z_pool_par1[which_cross == 1, :]
        z_pool_par2_cross = z_pool_par2[which_cross == 1, :]

        # if a pair is designated as a crossover pair, make children with crossover
        # otherwise, children are exact copies of parents
        z_pool_chil_cross = np.vstack([crossover(z_par1, z_par2, cross_k, rng) for 
                                       (z_par1, z_par2) in zip(z_pool_par1_cross, z_pool_par2_cross)])
        z_pool_chil_copy = np.vstack((z_pool_par1[which_cross == 0, :], z_pool_par2[which_cross == 0, :])) 
        z_pool_chil = np.vstack((z_pool_chil_copy, z_pool_chil_cross))
        
        # introduce mutations in children
        z_pool_chil_mut = mutation(z_pool_chil, mut_rate, rng)

        # combine parent and child generations
        z_pool_new = np.vstack((z_pool, z_pool_chil_mut))
        new_scores = fitness_fn(z_pool_new) + 2 * np.abs(0.5 - np.mean(z_pool_new, axis=1))
        keep = np.argsort(new_scores)[:init_pool_size]
        z_pool = z_pool_new[keep, :]
    
    return z_pool, new_scores[keep]