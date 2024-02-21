import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.design.genetic_algorithms import *

class CompleteRandomization():
    """
    Complete randomization design

    Args:
        - n: number of units
        - n_z: number of treatment allocations to sample
        - n_cutoff: number of treatment allocations to accept
        - seed: random seed
        - n_arms: number of treatment arms
    """
    def __init__(self, 
                 n: int, 
                 n_z: int, 
                 n_cutoff: int, 
                 seed=42,
                 n_arms=2):
        self.name = 'complete'
        self.n = n
        self.n_z = n_z
        self.n_cutoff = n_cutoff
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        
    
    def sample_z(self) -> np.ndarray:
        """
        Sample single treatment allocation (n,)
        """
        n_per_arm = self.n // self.n_arms
        n_rem = self.n % self.n_arms
        if n_rem > 0:
            n_per_arm = np.repeat(n_per_arm, self.n_arms)
            which_arm_to_add = self.rng.choice(self.n_arms, size=1)[0]
            n_per_arm[which_arm_to_add] += n_rem
        z = np.repeat(np.arange(self.n_arms, dtype=np.float32), n_per_arm)
        self.rng.shuffle(z)

        return z
    
    def sample_mult_z(self) -> np.ndarray:
        """
        Sample a pool of candidate treatment allocations (n_z x n)
        """
        print('Sampling z pool')
        return np.vstack([self.sample_z() for _ in tqdm(range(self.n_z))])
    
    def sample_accepted_idxs(self) -> np.ndarray:
        """
        Sample indices of accepted treatment allocations (n_cutoff,)
        """
        print('Filtering z pool')
        return self.rng.choice(np.arange(self.n_z), size=self.n_cutoff, replace=False)

    def sample_chosen_idx(self) -> int:
        """
        Sample index of chosen treatment allocation 
        """
        return self.rng.integers(self.n_cutoff)
    
    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under complete randomization and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        z_pool = self.sample_mult_z()
        accepted_idxs = self.sample_accepted_idxs()
        z_accepted = z_pool[accepted_idxs, :]
        z_accepted = np.unique(z_accepted, axis=0)
        self.n_cutoff = z_accepted.shape[0]
        chosen_idx = self.sample_chosen_idx()

        return z_accepted, chosen_idx

class RestrictedRandomization(CompleteRandomization):
    """
    Restricted randomization design

    Args:
        - n: number of units
        - n_z: number of treatment allocations to sample
        - n_cutoff: number of treatment allocations to accept
        - fitness_fn: fitness function
        - seed: random seed
        - n_arms: number of treatment arms
    """
    def __init__(self,
                 n: int, 
                 n_z: int, 
                 n_cutoff: int, 
                 fitness_fn: callable, 
                 seed=42, 
                 n_arms=2):
        super().__init__(n, n_z, n_cutoff, n_arms=n_arms)
        self.name = f'restricted_{fitness_fn.name}'
        self.fitness_fn = fitness_fn
        self.rng = np.random.default_rng(seed)

    def sample_accepted_idxs(self, z_pool) -> np.ndarray:
        """
        Sample indices of accepted treatment allocations (n_cutoff,)

        Args:
            - z_pool: pool of candidate treatment allocations (n_z x n)
        """
                
        print('Filtering z pool')

        # Compute fitness scores
        scores = self.fitness_fn(z_pool)
        self.scores = scores
        sorted_idx = np.argsort(scores)

        # Store accepted scores
        self.accepted_scores = scores[sorted_idx][:self.n_cutoff]
        
        # Return indices of accepted treatment allocations
        return sorted_idx[:self.n_cutoff]
    
    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under restricted randomization and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        z_pool = self.sample_mult_z()
        accepted_idxs = self.sample_accepted_idxs(z_pool)
        z_accepted = z_pool[accepted_idxs, :]
        z_accepted = np.unique(z_accepted, axis=0)
        self.n_cutoff = z_accepted.shape[0]
        chosen_idx = self.sample_chosen_idx()

        return z_accepted, chosen_idx
    
class RestrictedRandomizationGenetic(RestrictedRandomization):
    """
    Restricted randomization design using genetic algorithms to optimize pool of treatment allocations

    Args:
        - n: number of units
        - n_z: number of treatment allocations to sample
        - n_cutoff: number of treatment allocations to accept
        - fitness_fn: fitness function to evaluate treatment allocations
        - tourn_size: tournament size
        - cross_k: number of crossover points
        - cross_rate: crossover rate
        - mut_rate: mutation rate
        - genetic_iters: number of genetic iterations
        - seed: random seed
        - n_arms: number of treatment arms
    """
    def __init__(self, 
                 n: int, 
                 n_z: int, 
                 n_cutoff: int, 
                 fitness_fn: callable, 
                 tourn_size: int, 
                 cross_k: int, 
                 cross_rate: float, 
                 mut_rate: float, 
                 genetic_iters: int,
                 seed=42, 
                 n_arms=2):
        super().__init__(n, n_z, n_cutoff, fitness_fn, seed=seed, n_arms=n_arms)
        self.name = f'restricted-genetic_{fitness_fn.name}'

        self.tourn_size = tourn_size
        self.cross_k = cross_k
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.genetic_iters = genetic_iters
    
    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under restricted randomization with genetic algorithm optimization
        and choose one as the official observed allocation
        """
        z_pool = self.sample_mult_z()
        z_pool, _ = run_genetic_alg(z_pool, self.fitness_fn,
                                    self.tourn_size, self.cross_k, self.cross_rate, 
                                    self.mut_rate, self.genetic_iters, self.rng)
        accepted_idxs = self.sample_accepted_idxs(z_pool)
        z_accepted = z_pool[accepted_idxs, :]
        z_accepted = np.unique(z_accepted, axis=0)
        self.n_cutoff = z_accepted.shape[0]
        chosen_idx = self.sample_chosen_idx()

        return z_accepted, chosen_idx
    
class PairedGroupFormationRandomization(CompleteRandomization):
    """
    Paired group formation randomization design

    Args:
        - n: number of units
        - n_z: number of treatment allocations to sample
        - n_cutoff: number of treatment allocations to accept
        - X_on: mask of units with salient attribute
        - p_X_on_per_context: proportion of individuals with salient attribute per composition
        - n_arms: number of treatment arms
        - seed: random seed
    """
    def __init__(self, 
                 n: int, 
                 n_z: int, 
                 n_cutoff: int,  
                 X_on: np.ndarray, 
                 p_X_on_per: np.ndarray, 
                 n_arms=2, 
                 seed=42):
        self.name = f'group-formation'
        self.n = n
        self.n_z = n_z
        self.n_cutoff = n_cutoff
        self.n_arms = n_arms
        self.X_on = X_on
        self.p_X_on_per = p_X_on_per
        self.rng = np.random.default_rng(seed)
    
    def sample_z(self) -> np.ndarray:
        """
        Sample single treatment allocation (n,)
        """
        n_comps = len(self.p_X_on_per)

        # Number of groups is number of arms times number of compositions
        n_groups = self.n_arms * n_comps

        # Calculate number of individuals per group
        # If n is not divisible by n_groups, add the remainder to a random group
        n_per_group = self.n // n_groups
        n_rem = self.n % n_groups
        if n_rem > 0:
            n_per_group = np.repeat(n_per_group, self.n_arms)
            which_arm_to_add = self.rng.choice(self.n_arms, size=1)[0]
            n_per_group[which_arm_to_add] += n_rem

        # Calculate number of individuals with/without salient attribute 
        # that should be in each group
        n_X_on_per = np.array([int(p * n_per_group) for p in self.p_X_on_per])
        n_X_off_per = n_per_group - n_X_on_per
        n_X_on_per_group = np.repeat(n_X_on_per, self.n_arms)
        n_X_off_per_group = np.repeat(n_X_off_per, self.n_arms)

        # Create group assignments for individuals with/without salient attribute
        z_X_on = np.repeat(np.arange(n_groups, dtype=np.float32), n_X_on_per_group)
        z_X_off = np.repeat(np.arange(n_groups, dtype=np.float32), n_X_off_per_group)

        self.rng.shuffle(z_X_on)
        self.rng.shuffle(z_X_off)

        # Map group assignments to individuals
        z = np.zeros(self.n)
        z[self.X_on] = z_X_on
        z[~self.X_on] = z_X_off

        return z
    
class PairedGroupFormationRestrictedRandomization(PairedGroupFormationRandomization):
    """
    Restricted paired group formation randomization design

    Args:
        - n: number of units
        - n_z: number of treatment allocations to sample
        - n_cutoff: number of treatment allocations to accept
        - X_on: mask of units with salient attribute
        - p_X_on_per: proportion of individuals with salient attribute per composition
        - fitness_fn: fitness function
        - n_arms: number of treatment arms
        - seed: random seed
    """
    def __init__(self, 
                 n: int, 
                 n_z: int, 
                 n_cutoff: int,  
                 X_on: np.ndarray, 
                 p_X_on_per: np.ndarray, 
                 fitness_fn: callable, 
                 n_arms=2, 
                 seed=42):
         super().__init__(n, n_z, n_cutoff, X_on, p_X_on_per, n_arms=n_arms, seed=seed)
         self.name = f'group-formation-restricted_{fitness_fn.name}'
         self.fitness_fn = fitness_fn

    def sample_accepted_idxs(self, z_pool) -> np.ndarray:
        """
        Sample indices of accepted treatment allocations (n_cutoff,)

        Args:
            - z_pool: pool of candidate treatment allocations (n_z x n)
        """
        print('Filtering z pool')
        # Compute fitness scores
        scores = self.fitness_fn(z_pool)
        self.scores = scores
        sorted_idx = np.argsort(scores)

        # Store accepted scores
        self.accepted_scores = scores[sorted_idx][:self.n_cutoff]
        
        # Return indices of accepted treatment allocations
        return sorted_idx[:self.n_cutoff]
    
    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under paired group formation restricted randomization and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        z_pool = self.sample_mult_z()
        accepted_idxs = self.sample_accepted_idxs(z_pool)
        z_accepted = z_pool[accepted_idxs, :]
        z_accepted = np.unique(z_accepted, axis=0)
        self.n_cutoff = z_accepted.shape[0]
        chosen_idx = self.sample_chosen_idx()

        return z_accepted, chosen_idx 
    
class QuickBlockRandomization(CompleteRandomization):
    """
    Quick block randomization design
    Reads treatment allocations from pre-generated files

    Args:
        - n_arms: number of treatment arms
        - n_per_arm: number of units per arm
        - n_cutoff: number of treatment allocations to accept
        - min_block_factor: minimum size of block (relative to number of arms)
        - qb_dir: directory containing pre-generated treatment allocations
        - data_rep: data replication number
        - seed: random seed
    """
    def __init__(self, 
                 n_arms: int, 
                 n_per_arm: int, 
                 n_cutoff: int, 
                 min_block_factor: int, 
                 qb_dir: str, 
                 data_rep: int, 
                 seed=42):
        self.name = 'quick-block'
        self.n_arms = n_arms
        self.n_per_arm = n_per_arm
        self.n_cutoff = n_cutoff
        self.min_block_factor = min_block_factor
        self.qb_dir = qb_dir
        self.data_rep = data_rep
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Load quick-block treatment allocations from pre-generated files and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        arms_subdir = f'arms-{self.n_arms}'
        n_per_arm_subdir = f'n-per-arm-{self.n_per_arm}'
        n_cutoff_subdir = f'n-cutoff-{self.n_cutoff}'
        minblock_subdir = f'minblock-{self.min_block_factor}'

        z_dir = Path(self.qb_dir) / arms_subdir / n_per_arm_subdir / n_cutoff_subdir / minblock_subdir
        z_fname = f'{self.data_rep}.csv'
        z_accepted = pd.read_csv(z_dir / z_fname).to_numpy().transpose()
        chosen_idx = self.sample_chosen_idx()

        return z_accepted, chosen_idx