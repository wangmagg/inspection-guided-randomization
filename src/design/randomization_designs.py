import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time

from typing import Optional
from src.design.genetic_algorithms import *


class CompleteRandomization:
    """
    Complete randomization design

    Args:
        - n: number of units
        - n_z: number of treatment allocations to sample
        - n_cutoff: number of treatment allocations to accept
        - seed: random seed
        - n_arms: number of treatment arms
    """

    def __init__(self, n: int, n_z: int, n_cutoff: int, seed=42, n_arms=2):
        self.name = "complete"
        self.plotting_name = "CR"

        self.n = n
        self.n_z = n_z
        self.n_cutoff = n_cutoff
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

    def get_mirrors(self, z: np.ndarray) -> np.ndarray:
        """
        Get mirrors of treatment allocation

        Args:
            - z: treatment allocation (n,)
        """

        # Swap each arm with the control arm
        mirrors = np.zeros((self.n_arms - 1, self.n), dtype=np.float32)
        for i in range(self.n_arms - 1):
            mirrors[i] = z.copy()
            mirrors[i][z == (i + 1)] = 0
            mirrors[i][z == 0] = i + 1

        return mirrors

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
        print("Sampling z pool")
        return np.vstack([self.sample_z() for _ in tqdm(range(self.n_z))])

    def sample_accepted(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Sample indices of accepted treatment allocations (n_cutoff,)
        """
        print("Filtering z pool")

        # Sample indices of accepted treatment allocations
        n_cutoff_adj_for_mirrors = self.n_cutoff // self.n_arms
        accepted_idxs = self.rng.choice(
            np.arange(self.n_z), size=n_cutoff_adj_for_mirrors, replace=False
        )

        # Get accepted treatment allocations and remove duplicates
        z_accepted = z_pool[accepted_idxs]
        z_accepted_uniq = np.unique(z_accepted, axis=0)

        return z_accepted_uniq

    def add_mirror_allocations(self, z_accepted: np.ndarray) -> np.ndarray:
        """
        Add mirror allocations to pool of accepted treatment allocations

        Args:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
        """
        z_accepted_mirrors = np.vstack([self.get_mirrors(z) for z in z_accepted])
        z_accepted = np.vstack([z_accepted, z_accepted_mirrors])

        return z_accepted

    def sample_chosen_idx(self, z_accepted: np.ndarray) -> int:
        """
        Sample index of chosen treatment allocation

        Args:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
        """
        return self.rng.integers(z_accepted.shape[0])

    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under complete randomization and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        z_pool = self.sample_mult_z()
        z_accepted = self.sample_accepted(z_pool)
        z_accepted_with_mirrors = self.add_mirror_allocations(z_accepted)
        chosen_idx = self.sample_chosen_idx(z_accepted_with_mirrors)

        return z_accepted_with_mirrors, chosen_idx


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
        - add_all_mirrors: if True add all mirror allocations otherwise add only if they are not worse than the worst original allocation
        - n_batches: number of batches to split z_pool into
    """

    def __init__(
        self,
        n: int,
        n_z: int,
        n_cutoff: int,
        fitness_fn: callable,
        seed=42,
        n_arms=2,
        add_all_mirrors=False,
        n_batches=None,
    ):
        super().__init__(n, n_z, n_cutoff, n_arms=n_arms)
        if add_all_mirrors:
            self.name = f"restricted_{fitness_fn.name}-all-mirr"
        else:
            self.name = f"restricted_{fitness_fn.name}"

        self.plotting_name = f"IGR - {fitness_fn.plotting_name}"

        self.fitness_fn = fitness_fn
        self.add_all_mirrors = add_all_mirrors
        self.n_batches = n_batches

        self.scores = None
        self.z_pool = None

        self.rng = np.random.default_rng(seed)

    def sample_accepted(self, z_pool: np.ndarray) -> np.ndarray:
        """
        Sample indices of accepted treatment allocations (n_cutoff,)

        Args:
            - z_pool: pool of candidate treatment allocations (n_z x n)
        """

        print("Filtering z pool")

        # Compute fitness scores
        if self.scores is not None:
            print("Using cached scores")
            scores = self.scores
        else:
            scores = self.fitness_fn(z_pool)
            self.scores = scores

        sorted_idx = np.argsort(scores)

        # Get accepted treatment allocations and remove duplicates
        accepted_idxs = sorted_idx[: self.n_cutoff]
        z_accepted = z_pool[accepted_idxs, :]
        z_accepted_uniq = np.unique(z_accepted, axis=0)
        max_accepted_score = scores[accepted_idxs[-1]]

        return z_accepted_uniq, max_accepted_score

    def add_mirror_allocations(
        self, z_accepted: np.ndarray, max_accepted_score: np.ndarray, add_all=False
    ) -> np.ndarray:
        z_accepted_mirrors = np.vstack([self.get_mirrors(z) for z in z_accepted])

        if add_all:
            cutoff_adj_for_mirr = z_accepted.shape[0] // 2
            z_accepted = np.vstack(
                [
                    z_accepted[:cutoff_adj_for_mirr],
                    z_accepted_mirrors[:cutoff_adj_for_mirr],
                ]
            )

        else:
            # Get mirror allocations, keeping only those that are not worse than the worst accepted score of the original allocations
            map_mirr_to_orig_idx = np.repeat(
                np.arange(z_accepted.shape[0]), self.n_arms - 1
            )

            # only keep mirror allocations that aren't worse than the worst original allocation
            # across the two component fitness functions or figure out different way to standardize
            if "lin-comb" in self.fitness_fn.name:
                score_fit_fn_0 = self.fitness_fn.fitness_fns[0](z_accepted)
                score_fit_fn_1 = self.fitness_fn.fitness_fns[1](z_accepted)

                score_fit_fn_0_mirr = self.fitness_fn.fitness_fns[0](z_accepted_mirrors)
                score_fit_fn_1_mirr = self.fitness_fn.fitness_fns[1](z_accepted_mirrors)

                max_score_fit_fn_0 = np.max(score_fit_fn_0)
                max_score_fit_fn_1 = np.max(score_fit_fn_1)
                mirrors_not_worse_mask = (score_fit_fn_0_mirr <= max_score_fit_fn_0) & (
                    score_fit_fn_1_mirr <= max_score_fit_fn_1
                )
            # if fitness fn is not linear combination, check against max accepted score
            else:
                mirror_scores = self.fitness_fn(z_accepted_mirrors)
                mirrors_not_worse_mask = mirror_scores <= max_accepted_score

            if sum(mirrors_not_worse_mask) > 0:
                # Determine how many original and mirror allocations to keep based on n_cutoff
                z_accepted_mirrors_not_worse = z_accepted_mirrors[
                    mirrors_not_worse_mask
                ]
                map_mirr_to_orig_idx_not_worse = map_mirr_to_orig_idx[
                    mirrors_not_worse_mask
                ]
                binned_map = np.bincount(map_mirr_to_orig_idx_not_worse)
                binned_map_cum_sum = np.cumsum(binned_map)
                binned_map_add_orig_cumsum = np.cumsum(binned_map + 1)

                where_cutoff = np.where(
                    binned_map_add_orig_cumsum >= z_accepted.shape[0]
                )

                if len(where_cutoff[0]) == 0:
                    cutoff_idx_for_mirr = binned_map_cum_sum[-1]
                    cutoff_idx_for_orig = z_accepted.shape[0] - cutoff_idx_for_mirr
                else:
                    cutoff_idx_for_orig = where_cutoff[0][0]
                    cutoff_idx_for_mirr = binned_map_cum_sum[cutoff_idx_for_orig]

                z_accepted = np.vstack(
                    [
                        z_accepted[:cutoff_idx_for_orig],
                        z_accepted_mirrors_not_worse[:cutoff_idx_for_mirr],
                    ]
                )

        return z_accepted
    
    def _batched_filtering(self):
        print("Running batched version")
        batch_size = self.n_z // self.n_batches
        batch_cutoff = self.n_cutoff // self.n_batches

        # batched_z_pool = np.array_split(self.z_pool, self.n_batches, axis=0)
        # batch_cutoffs = [int(self.n_cutoff / self.n_z * batch.shape[0]) for batch in batched_z_pool]
        z_accepted_with_mirrors = None
        scores = None
        for _ in tqdm(range(self.n_batches)):
            self.n_z = batch_size
            self.n_cutoff = batch_cutoff

            # Sample batch of candidate treatment allocations
            z_pool_batch = self.sample_mult_z()

            # Get the batch's accepted treatment allocations and their scores
            z_batch_accepted, batch_max_accepted_score = self.sample_accepted(z_pool_batch)
            z_batch_accepted_with_mirrors = self.add_mirror_allocations(
                z_batch_accepted, batch_max_accepted_score, self.add_all_mirrors
            )
            if z_accepted_with_mirrors is None:
                z_accepted_with_mirrors = z_batch_accepted_with_mirrors
            else:
                z_accepted_with_mirrors = np.vstack(
                    [z_accepted_with_mirrors, z_batch_accepted_with_mirrors]
                )
            if scores is None:
                scores = self.scores
            else:
                scores = np.hstack([scores, self.scores])

            # Reset scores
            self.scores = None

        self.scores = scores

        return z_accepted_with_mirrors

    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under restricted randomization and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        
        if self.n_batches is None:
            if self.z_pool is not None:
                print("Using cached z pool")
            else:
                self.z_pool = self.sample_mult_z()

            z_accepted, max_accepted_score = self.sample_accepted(self.z_pool)
            z_accepted_with_mirrors = self.add_mirror_allocations(
                z_accepted, max_accepted_score, self.add_all_mirrors
            )
        else:
            z_accepted_with_mirrors = self._batched_filtering()

        chosen_idx = self.sample_chosen_idx(z_accepted_with_mirrors)

        return z_accepted_with_mirrors, chosen_idx


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
        - eps: tolerance for departing from original pool's treatment-to-control ratio
        - seed: random seed
        - n_arms: number of treatment arms
        - add_all_mirrors: if True add all mirror allocations otherwise add only if they are not worse than the worst original allocation
    """

    def __init__(
        self,
        n: int,
        n_z: int,
        n_cutoff: int,
        fitness_fn: callable,
        tourn_size: int,
        cross_k: int,
        cross_rate: float,
        mut_rate: float,
        genetic_iters: int,
        eps: Optional[float] = 0.05,
        seed: Optional[int] = 42,
        n_arms: Optional[int] = 2,
        add_all_mirrors=False,
        n_batches=None
    ):
        super().__init__(
            n,
            n_z,
            n_cutoff,
            fitness_fn,
            seed=seed,
            n_arms=n_arms,
            add_all_mirrors=add_all_mirrors,
            n_batches=n_batches
        )
        if add_all_mirrors:
            self.name = f"restricted-genetic_{fitness_fn.name}-all-mirr"
        else:
            self.name = f"restricted-genetic_{fitness_fn.name}"
        self.plotting_name = f"IGRg - {fitness_fn.plotting_name}"

        self.tourn_size = tourn_size
        self.cross_k = cross_k
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.genetic_iters = genetic_iters
        self.eps = eps

    def _batched_filtering(self):
        print("Running batched version")
        batch_size = self.n_z // self.n_batches
        batch_cutoff = self.n_cutoff // self.n_batches

        # batched_z_pool = np.array_split(self.z_pool, self.n_batches, axis=0)
        # batch_cutoffs = [int(self.n_cutoff / self.n_z * batch.shape[0]) for batch in batched_z_pool]
        z_accepted_with_mirrors = None
        scores = None
        for _ in tqdm(range(self.n_batches)):
            self.n_z = batch_size
            self.n_cutoff = batch_cutoff

            # Sample batch of candidate treatment allocations
            z_pool_batch = self.sample_mult_z()
            z_pool_batch, _ = run_genetic_alg(
                            z_pool_batch,
                            self.fitness_fn,
                            self.tourn_size,
                            self.cross_k,
                            self.cross_rate,
                            self.mut_rate,
                            self.genetic_iters,
                            self.rng,
                            self.eps,
                        )

            # Get the batch's accepted treatment allocations and their scores
            z_batch_accepted, batch_max_accepted_score = self.sample_accepted(z_pool_batch)
            z_batch_accepted_with_mirrors = self.add_mirror_allocations(
                z_batch_accepted, batch_max_accepted_score, self.add_all_mirrors
            )
            if z_accepted_with_mirrors is None:
                z_accepted_with_mirrors = z_batch_accepted_with_mirrors
            else:
                z_accepted_with_mirrors = np.vstack(
                    [z_accepted_with_mirrors, z_batch_accepted_with_mirrors]
                )
            if scores is None:
                scores = self.scores
            else:
                scores = np.hstack([scores, self.scores])

            # Reset scores
            self.scores = None

        self.scores = scores

        return z_accepted_with_mirrors

    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under restricted randomization with genetic algorithm optimization
        and choose one as the official observed allocation
        """
        if self.n_batches is None:
            z_pool = self.sample_mult_z()
            z_pool, _ = run_genetic_alg(
                z_pool,
                self.fitness_fn,
                self.tourn_size,
                self.cross_k,
                self.cross_rate,
                self.mut_rate,
                self.genetic_iters,
                self.rng,
                self.eps,
            )
            z_accepted, max_accepted_score = self.sample_accepted(z_pool)
            z_accepted_with_mirrors = self.add_mirror_allocations(
                z_accepted, max_accepted_score, self.add_all_mirrors
            )
        else:
            z_accepted_with_mirrors = self._batched_filtering()
            
        chosen_idx = self.sample_chosen_idx(z_accepted_with_mirrors)

        return z_accepted_with_mirrors, chosen_idx


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

    def __init__(
        self,
        n: int,
        n_z: int,
        n_cutoff: int,
        X_on: np.ndarray,
        p_X_on_per: np.ndarray,
        n_arms=2,
        seed=42,
    ):
        self.name = f"group-formation"
        self.plotting_name = "GFR"

        self.n = n
        self.n_z = n_z
        self.n_cutoff = n_cutoff
        self.n_arms = n_arms
        self.X_on = X_on
        self.p_X_on_per = p_X_on_per
        self.rng = np.random.default_rng(seed)

    def get_mirrors(self, z: np.ndarray) -> np.ndarray:
        """
        Get mirrors of treatment allocation

        Args:
            - z: treatment allocation (n,)
        """
        n_compositions = len(self.p_X_on_per)
        # Get labels for pairs of groups with same composition but different treatment
        same_comp_diff_trt_group_lbls = np.arange(self.n_arms * n_compositions).reshape(
            n_compositions, self.n_arms
        )

        # Swap treatment assignments within each composition
        mirrors = np.zeros((self.n_arms - 1, self.n), dtype=np.float32)
        for i in range(self.n_arms - 1):
            mirrors[i] = z.copy()
            for comp_group_lbls in same_comp_diff_trt_group_lbls:
                mirrors[i][z == comp_group_lbls[i + 1]] = comp_group_lbls[0]
                mirrors[i][z == comp_group_lbls[0]] = comp_group_lbls[i + 1]

        return mirrors

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
        - add_all_mirrors: if True add all mirror allocations otherwise add only if they are not worse than the worst original allocation
    """

    def __init__(
        self,
        n: int,
        n_z: int,
        n_cutoff: int,
        X_on: np.ndarray,
        p_X_on_per: np.ndarray,
        fitness_fn: callable,
        n_arms=2,
        seed=42,
        add_all_mirrors=False,
    ):
        super().__init__(
            n,
            n_z,
            n_cutoff,
            X_on,
            p_X_on_per,
            n_arms=n_arms,
            seed=seed,
            add_all_mirrors=add_all_mirrors,
        )
        if add_all_mirrors:
            self.name = f"group-formation-restricted_{fitness_fn.name}-all-mirr"
        else:
            self.name = f"group-formation-restricted_{fitness_fn.name}"

        self.plotting_name = f"IGR (GFR) - {fitness_fn.plotting_name}"

        self.fitness_fn = fitness_fn
        self.scores = None
        self.add_all_mirrors = add_all_mirrors

    def add_mirror_allocations(
        self, z_accepted: np.ndarray, max_accepted_score: np.ndarray, add_all=False
    ) -> np.ndarray:
        # Get mirror allocations, keeping only those that are not worse than the best accepted score of the original allocations
        z_accepted_mirrors = np.vstack([self.get_mirrors(z) for z in z_accepted])
        map_mirr_to_orig_idx = np.repeat(
            np.arange(z_accepted.shape[0]), self.n_arms - 1
        )
        mirror_scores = self.fitness_fn(z_accepted_mirrors)

        if add_all:
            z_accepted_mirrors_not_worse = z_accepted_mirrors
        else:
            z_accepted_mirrors_not_worse = z_accepted_mirrors[
                mirror_scores <= max_accepted_score
            ]

        if len(z_accepted_mirrors_not_worse) > 0:
            # Determine how many original and mirror allocations to keep based on n_cutoff
            map_mirr_to_orig_idx_not_worse = map_mirr_to_orig_idx[
                mirror_scores <= max_accepted_score
            ]
            binned_map = np.bincount(map_mirr_to_orig_idx_not_worse)
            binned_map_cum_sum = np.cumsum(binned_map)
            binned_map_add_orig_cumsum = np.cumsum(binned_map + 1)

            where_cutoff = np.where(binned_map_add_orig_cumsum >= self.n_cutoff)

            if len(where_cutoff[0]) == 0:
                cutoff_idx_for_mirr = binned_map_cum_sum[-1]
                cutoff_idx_for_orig = self.n_cutoff - cutoff_idx_for_mirr
            else:
                cutoff_idx_for_orig = where_cutoff[0][0]
                cutoff_idx_for_mirr = binned_map_cum_sum[cutoff_idx_for_orig]

            z_accepted = np.vstack(
                [
                    z_accepted[:cutoff_idx_for_orig],
                    z_accepted_mirrors_not_worse[:cutoff_idx_for_mirr],
                ]
            )

        return z_accepted

    def sample_accepted(self, z_pool) -> np.ndarray:
        """
        Sample indices of accepted treatment allocations (n_cutoff,)

        Args:
            - z_pool: pool of candidate treatment allocations (n_z x n)
        """
        print("Filtering z pool")
        # Compute fitness scores
        scores = self.fitness_fn(z_pool)
        sorted_idx = np.argsort(scores)

        # Get accepted treatment allocations and remove duplicates
        accepted_idxs = sorted_idx[: self.n_cutoff]
        z_accepted = z_pool[accepted_idxs, :]
        z_accepted_uniq = np.unique(z_accepted, axis=0)
        max_accepted_score = scores[accepted_idxs[-1]]

        self.scores = scores

        return z_accepted_uniq, max_accepted_score

    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under paired group formation restricted randomization and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        z_pool = self.sample_mult_z()
        z_accepted, max_accepted_score = self.sample_accepted(z_pool)
        z_accepted_with_mirrors = self.add_mirror_allocations(
            z_accepted, max_accepted_score, self.add_all_mirrors
        )
        chosen_idx = self.sample_chosen_idx(z_accepted_with_mirrors)

        return z_accepted_with_mirrors, chosen_idx


class PairedGroupFormationRestrictedRandomizationGenetic(
    PairedGroupFormationRestrictedRandomization
):
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
        - add_all_mirrors: if True add all mirror allocations otherwise add only if they are not worse than the worst original allocation
    """

    def __init__(
        self,
        n: int,
        n_z: int,
        n_cutoff: int,
        X_on: np.ndarray,
        p_X_on_per: np.ndarray,
        fitness_fn: callable,
        tourn_size: int,
        cross_k: int,
        cross_rate: float,
        mut_rate: float,
        genetic_iters: int,
        eps: Optional[float] = 0,
        seed: Optional[int] = 42,
        n_arms: Optional[int] = 2,
        add_all_mirrors=False,
    ):
        super().__init__(
            n,
            n_z,
            n_cutoff,
            X_on,
            p_X_on_per,
            fitness_fn,
            seed=seed,
            n_arms=n_arms,
            add_all_mirrors=add_all_mirrors,
        )
        if add_all_mirrors:
            self.name = f"group-formation-restricted-genetic_{fitness_fn.name}-all-mirr"
        else:
            self.name = f"group-formation-restricted-genetic_{fitness_fn.name}"

        self.plotting_name = f"IGRg (GFR) - {fitness_fn.plotting_name}"

        self.tourn_size = tourn_size
        self.cross_k = cross_k
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.genetic_iters = genetic_iters
        self.eps = eps

        self.same_comp_diff_trt = np.arange(n_arms * len(p_X_on_per)).reshape(
            len(p_X_on_per), n_arms
        )

    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under restricted randomization with genetic algorithm optimization
        and choose one as the official observed allocation
        """
        z_pool = self.sample_mult_z()
        z_pool, _ = run_genetic_alg(
            z_pool,
            self.fitness_fn,
            self.tourn_size,
            self.cross_k,
            self.cross_rate,
            self.mut_rate,
            self.genetic_iters,
            self.rng,
            self.eps,
        )
        z_accepted, max_accepted_score = self.sample_accepted(z_pool)
        z_accepted_with_mirrors = self.add_mirror_allocations(
            z_accepted, max_accepted_score, self.add_all_mirrors
        )
        chosen_idx = self.sample_chosen_idx(z_accepted_with_mirrors)

        return z_accepted_with_mirrors, chosen_idx


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

    def __init__(
        self,
        n_arms: int,
        n_per_arm: int,
        n_cutoff: int,
        min_block_factor: int,
        qb_dir: str,
        data_rep: int,
        seed=42,
    ):
        self.name = "quick-block"
        self.plotting_name = "QB"

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
        arms_subdir = f"arms-{self.n_arms}"
        n_per_arm_subdir = f"n-per-arm-{self.n_per_arm}"
        n_cutoff_subdir = f"n-cutoff-{self.n_cutoff}"
        minblock_subdir = f"minblock-{self.min_block_factor}"

        z_dir = (
            Path(self.qb_dir)
            / arms_subdir
            / n_per_arm_subdir
            / n_cutoff_subdir
            / minblock_subdir
        )
        z_fname = f"{self.data_rep}.csv"
        z_accepted = pd.read_csv(z_dir / z_fname).to_numpy().transpose()
        chosen_idx = self.sample_chosen_idx(z_accepted)

        return z_accepted, chosen_idx


class GraphRandomization(CompleteRandomization):
    def __init__(self, n, n_z, n_cutoff, dists, A, seed=42):
        super().__init__(n, n_z, n_cutoff)
        self.n_clusters = n
        self.name = "graph"
        self.plotting_name = "GR"

        self.dists = dists
        self.A = A
        self.rng = np.random.default_rng(seed)
        self.cached_B = False
        self.mapping = None

    def get_two_ball(self):
        # get two-ball for each vertex in graph
        B = []
        for i in range(self.A.shape[0]):
            r1_mask = self.A[i, :]
            r1_idxs = np.flatnonzero(r1_mask)  # all indices within radius 1
            r2_mask = np.sum(self.A[r1_idxs, :], axis=0)  # all indices within radius 2
            r2_idxs = np.flatnonzero(
                r1_mask + r2_mask
            )  # all indices within radius 1 or 2
            B.append(r2_idxs)
        return B

    def three_net(self, B):
        n = self.A.shape[0]
        visited = np.zeros(n)
        unvisited = np.flatnonzero(visited == 0)
        V = []  # store cluster centers

        while len(unvisited) > 0:
            # randomly choose an unvisited vertex
            v = self.rng.choice(unvisited)

            # mark the vertex and its two-ball as visited
            visited[v] = 1
            visited[B[v]] = 1

            # add vertex as a cluster center
            V.append(v)

            unvisited = np.flatnonzero(visited == 0)

        # assign vertices to closest clusters
        C = np.zeros(n)  # vertex assignments
        for i in range(n):
            dists_to_v = [self.dists[i][v] for v in V if v in self.dists[i]]
            if not dists_to_v:
                C[i] = V[self.rng.integers(len(V))]
            else:
                C[i] = V[np.argmin(dists_to_v)]

        # Drop elements from V if not in C
        V_rem_miss = [v for v in V if v in C]

        # recode V to be 0, 1, 2, ... n_clusters
        V_rec = np.arange(len(V_rem_miss))

        # update C to match the recoded V
        C_rec = np.zeros(n)
        for i, v in enumerate(V_rem_miss):
            C_rec[C == v] = i

        return V_rec, C_rec

    def sample_z(self, V):
        treated_clusters = self.rng.choice(V, size=len(V) // 2, replace=False)
        z = np.zeros(len(V))
        z[treated_clusters] = 1

        return z

    def sample_mult_z(self):
        if not self.cached_B:
            self.B = self.get_two_ball()
            self.cached_B = True
        V, C = self.three_net(self.B)
        self.mapping = C.astype(int)
        z_pool = [self.sample_z(V) for _ in range(self.n_z)]

        return np.array(z_pool)

    def __call__(self) -> tuple[np.ndarray, int]:
        """
        Sample treatment allocations under graph-clustered randomization and
        choose one as the official observed allocation

        Returns:
            - z_accepted: accepted treatment allocations (n_cutoff x n)
            - chosen_idx: index of chosen treatment allocation
        """
        z_pool = self.sample_mult_z()
        z_accepted = self.sample_accepted(z_pool)
        chosen_idx = self.sample_chosen_idx(z_accepted)

        return z_accepted, chosen_idx
