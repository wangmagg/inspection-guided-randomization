import pandas as pd
from pathlib import Path
from functools import partial

from src.models.outcome_models import (
    PotentialOutcomeDGP,
    KenyaPODGP,
    KenyaNbrSumPODGP,
    ClassroomPODGP,
    ClassroomFixedMalePODGP,
    NormSumPODGP,
    NormSumDegCorPODGP,
    NormSumDegCorClusterPODGP,
    AdditiveInterference,
    Consistent,
    AdditiveComposition,
)
from src.models.network_models import (
    StochasticBlock,
    TwoLevelNestedStochasticBlock,
    EuclideanDistPowerDecayIntxn,
    ErdosRenyi,
    WattsStrogatz,
    BarabasiAlbert,
)
from src.models.exposure_models import ExposureModel, FracNbr, OneNbr

from src.design.fitness_functions import (
    SMD,
    MaxAbsSMD,
    SignedMaxAbsSMD,
    SumMaxAbsSMD,
    SumAbsSMD,
    MaxMahalanobis,
    WeightedMaxMahalanobis,
    FracExposed,
    MinPairwiseEuclideanDist,
    SumMaxAbsSMDFracExpo,
    SumMaxAbsSMDMinPairwiseEuclideanDist,
    MaxMahalanobisFracExpo,
    MaxMahalanobisMinPairwiseEuclideanDist,
    WeightedMaxMahalanobisFracExpo,
    WeightedMaxMahalanobisMinPairwiseEuclideanDist,
)
from src.design.randomization_designs import (
    CompleteRandomization,
    RestrictedRandomization,
    RestrictedRandomizationGenetic,
    PairedGroupFormationRandomization,
    PairedGroupFormationRestrictedRandomization,
    PairedGroupFormationRestrictedRandomizationGenetic,
    QuickBlockRandomization,
    GraphRandomization,
)

from src.analysis.estimators import (
    diff_in_means,
    diff_in_means_mult_arm,
    clustered_diff_in_means,
    qb_diff_in_means_mult_arm,
    get_pval,
    get_pval_mult_arm,
    get_pval_clustered,
    get_pval_qb
)


"""
Loader functions for a simulated trial
"""


def get_potential_outcome_mdl(config) -> PotentialOutcomeDGP:
    """
    Get potential outcome model from name in configuration
    """
    if config.potential_outcome_mdl_name == "kenya-hierarchical":
        param = pd.read_csv(config.param_fname, index_col="settlement")

        return KenyaPODGP(
            param=param,
            tau_size=config.tau_size,
            sigma_sis_scale=config.sigma_sis_scale,
            sigma_iis_scale=config.sigma_iis_scale,
            beta=config.beta,
            sigma=config.sigma,
            seed=config.seed,
        )
    elif config.potential_outcome_mdl_name == "kenya-hierarchical-nbr-sum":
        param = pd.read_csv(config.param_fname, index_col="settlement")

        return KenyaNbrSumPODGP(
            param=param,
            tau_size=config.tau_size,
            sigma_sis_scale=config.sigma_sis_scale,
            sigma_iis_scale=config.sigma_iis_scale,
            beta=config.beta,
            sigma=config.sigma,
            seed=config.seed,
        )
    elif config.potential_outcome_mdl_name == "classroom":
        return ClassroomPODGP(
            n_students=config.n,
            n_arms=config.n_arms,
            sigma=config.sigma,
            tau_sizes=config.tau_sizes,
            seed=config.seed,
        )
    elif config.potential_outcome_mdl_name == "classroom-composition":
        return ClassroomFixedMalePODGP(
            n_students=config.n,
            sigma=config.sigma,
            tau_size=config.tau_size,
            tau_comp_sizes=config.tau_comp_sizes,
            seed=config.seed,
        )
    elif config.potential_outcome_mdl_name == "norm-sum":
        return NormSumPODGP(
            n=config.n,
            mu=config.mu,
            sigma=config.sigma,
            gamma=config.gamma,
            tau_size=config.tau_size,
            seed=config.seed,
        )
    elif config.potential_outcome_mdl_name == "norm-sum-deg-cor":
        return NormSumDegCorPODGP(
            n=config.n,
            mu=config.mu,
            sigma=config.sigma,
            gamma=config.gamma,
            tau_size=config.tau_size,
            seed=config.seed,
        )
    elif config.potential_outcome_mdl_name == "norm-sum-deg-cor-cluster":
        return NormSumDegCorClusterPODGP(
            n=config.n,
            mu=config.mu,
            sigma=config.sigma,
            gamma=config.gamma,
            tau_size=config.tau_size,
            seed=config.seed,
        )


def get_expo_mdl(config) -> ExposureModel:
    """
    Get exposure model from name in configuration
    """
    if config.expo_mdl_name == "frac-nbr-expo":
        return FracNbr(config.q)
    elif config.expo_mdl_name == "one-nbr-expo":
        return OneNbr()
    else:
        raise ValueError(f"Unrecognized exposure model: {config.expo_mdl_name}")


def get_spatial_intxn_mdl(config):
    """
    Get spatial interaction model from name in configuration
    """
    if config.intxn_mdl_name == "power-decay":
        return EuclideanDistPowerDecayIntxn(config.gamma)
    else:
        raise ValueError(
            f"Unrecognized spatial interaction model: {config.intxn_mdl_name}"
        )


def get_full_spatial_intxn_mdl_name(config):
    """
    Get full name of spatial interaction model from name in configuration
    """
    if config.intxn_mdl_name == "power-decay":
        return f"euclidean-dist-power-decay-gamma-{config.gamma}"
    else:
        raise ValueError(
            f"Unrecognized spatial interaction model: {config.intxn_mdl_name}"
        )


def get_net_mdl(config, inner_to_outer_mapping=None):
    """
    Get network model from name in configuration
    """
    if "sb" in config.net_mdl_name and "nested" not in config.net_mdl_name:
        return StochasticBlock(config.wi_p, config.bw_p, config.seed)
    if "nested-2lvl-sb" in config.net_mdl_name:
        return TwoLevelNestedStochasticBlock(
            config.p_same_in,
            config.p_diff_in_same_out,
            config.p_diff_in_diff_out,
            inner_to_outer_mapping,
            config.seed,
        )
    if "er" in config.net_mdl_name:
        return ErdosRenyi(config.p_er, config.seed)
    if "ws" in config.net_mdl_name:
        return WattsStrogatz(config.k, config.p_ws, config.seed)
    if "ba" in config.net_mdl_name:
        return BarabasiAlbert(config.m, config.seed)
    else:
        raise ValueError(f"Unrecognized network model: {config.net_mdl_name}")


def get_full_net_mdl_name(config) -> str:
    """
    Get full name of network model from name in configuration
    """
    if "sb" in config.net_mdl_name and "nested" not in config.net_mdl_name:
        return f"sb_wip-{config.wi_p:.2f}_bwp-{config.bw_p:.2f}"
    if "nested-2lvl-sb" in config.net_mdl_name:
        if config.p_diff_in_diff_out < 0.01:
            return f"nested-2lvl-sb_psi-{config.p_same_in:.2f}_pdiso-{config.p_diff_in_same_out:.2f}_pdido-{config.p_diff_in_diff_out:.3f}"
        else:
            return f"nested-2lvl-sb_psi-{config.p_same_in:.2f}_pdiso-{config.p_diff_in_same_out:.2f}_pdido-{config.p_diff_in_diff_out:.2f}"
    if "er" in config.net_mdl_name:
        return f"er_p-{config.p_er:.2f}"
    if "ws" in config.net_mdl_name:
        return f"ws_k-{config.k}_p-{config.p_ws:.2f}"
    if "ba" in config.net_mdl_name:
        return f"ba_m-{config.m}"
    else:
        raise ValueError(f"Unrecognized network model: {config.net_mdl_name}")


def get_observed_outcome_mdl(trial):
    """
    Get observed outcome model from name in configuration

    Args:
        - trial (SimulatedTrial): trial to get observed outcome model for
    """
    if trial.config.observed_outcome_mdl_name == "additive-interference":
        return AdditiveInterference(
            trial.config.delta_size, trial.expo_mdl, trial.A, trial.mapping
        )
    elif trial.config.observed_outcome_mdl_name == "consistent":
        return Consistent()
    elif trial.config.observed_outcome_mdl_name == "additive-composition":
        return AdditiveComposition(trial.same_comp_diff_trt)
    else:
        raise ValueError(
            f"Unrecognized outcome model: {trial.config.observed_outcome_mdl_name}"
        )


def get_fitness_fn(trial):
    """
    Get fitness function from name in configuration

    Args:
        - trial (SimulatedTrial): trial to get fitness function for
    """
    if (
        trial.config.fitness_fn_name == "smd"
        or trial.config.fitness_fn_name == "weighted-smd"
    ):
        return SMD.from_trial(trial)
    elif trial.config.fitness_fn_name == "max-abs-smd":
        return MaxAbsSMD.from_trial(trial)
    elif trial.config.fitness_fn_name == "signed-max-abs-smd":
        return SignedMaxAbsSMD.from_trial(trial)
    elif trial.config.fitness_fn_name == "sum-max-abs-smd":
        return SumMaxAbsSMD.from_trial(trial)
    elif trial.config.fitness_fn_name == "weighted-sum-abs-smd":
        return SumAbsSMD.from_trial(trial)
    elif trial.config.fitness_fn_name == "max-mahalanobis":
        return MaxMahalanobis.from_trial(trial)
    elif trial.config.fitness_fn_name == "weighted-max-mahalanobis":
        return WeightedMaxMahalanobis.from_trial(trial)
    elif trial.config.fitness_fn_name == "frac-expo":
        return FracExposed.from_trial(trial)
    elif trial.config.fitness_fn_name == "min-pairwise-euclidean-dist":
        return MinPairwiseEuclideanDist.from_trial(trial)
    elif trial.config.fitness_fn_name == "lin-comb_sum-max-abs-smd_frac-expo":
        return SumMaxAbsSMDFracExpo.from_trial(trial)
    elif trial.config.fitness_fn_name == "lin-comb_sum-max-abs-smd_euclidean-dist":
        return SumMaxAbsSMDMinPairwiseEuclideanDist.from_trial(trial)
    elif trial.config.fitness_fn_name == "lin-comb_max-mahalanobis_frac-expo":
        return MaxMahalanobisFracExpo.from_trial(trial)
    elif trial.config.fitness_fn_name == "lin-comb_max-mahalanobis_euclidean-dist":
        return MaxMahalanobisMinPairwiseEuclideanDist.from_trial(trial)
    elif trial.config.fitness_fn_name == "lin-comb_weighted-max-mahalanobis_frac-expo":
        return WeightedMaxMahalanobisFracExpo.from_trial(trial)
    elif (
        trial.config.fitness_fn_name
        == "lin-comb_weighted-max-mahalanobis_euclidean-dist"
    ):
        return WeightedMaxMahalanobisMinPairwiseEuclideanDist.from_trial(trial)
    else:
        raise ValueError(
            f"Unrecognized fitness function: {trial.config.fitness_fn_name}"
        )


def get_design(trial):
    """
    Get randomization design from name in configuration

    Args:
        - trial (SimulatedTrial): trial to get randomization design for
    """
    if trial.config.rand_mdl_name == "complete":
        return CompleteRandomization(
            trial.config.n,
            trial.config.n_z,
            trial.config.n_cutoff,
            n_arms=trial.config.n_arms,
            seed=trial.config.seed,
        )
    elif trial.config.rand_mdl_name == "restricted":
        fitness_fn = get_fitness_fn(trial)
        return RestrictedRandomization(
            trial.config.n,
            trial.config.n_z,
            trial.config.n_cutoff,
            fitness_fn,
            seed=trial.config.seed,
            n_arms=trial.config.n_arms,
            add_all_mirrors=trial.config.add_all_mirrors,
            n_batches=trial.config.n_batches
        )
    elif trial.config.rand_mdl_name == "restricted-genetic":
        fitness_fn = get_fitness_fn(trial)
        return RestrictedRandomizationGenetic(
            trial.config.n,
            trial.config.n_z,
            trial.config.n_cutoff,
            fitness_fn,
            trial.config.tourn_size,
            trial.config.cross_k,
            trial.config.cross_rate,
            trial.config.mut_rate,
            trial.config.genetic_iters,
            eps=trial.config.eps,
            seed=trial.config.seed,
            n_arms=trial.config.n_arms,
            add_all_mirrors=trial.config.add_all_mirrors,
            n_batches=trial.config.n_batches
        )
    elif trial.config.rand_mdl_name == "group-formation":
        return PairedGroupFormationRandomization(
            trial.config.n,
            trial.config.n_z,
            trial.config.n_cutoff,
            trial.X_on,
            trial.config.p_comps,
            seed=trial.config.seed,
            n_arms=trial.config.n_arms,
        )
    elif trial.config.rand_mdl_name == "group-formation-restricted":
        fitness_fn = get_fitness_fn(trial)
        return PairedGroupFormationRestrictedRandomization(
            trial.config.n,
            trial.config.n_z,
            trial.config.n_cutoff,
            trial.X_on,
            trial.config.p_comps,
            fitness_fn,
            seed=trial.config.seed,
            n_arms=trial.config.n_arms,
            add_mirrors=trial.config.add_all_mirrors,
        )
    elif trial.config.rand_mdl_name == "group-formation-restricted-genetic":
        fitness_fn = get_fitness_fn(trial)
        return PairedGroupFormationRestrictedRandomizationGenetic(
            trial.config.n,
            trial.config.n_z,
            trial.config.n_cutoff,
            trial.X_on,
            trial.config.p_comps,
            fitness_fn,
            trial.config.tourn_size,
            trial.config.cross_k,
            trial.config.cross_rate,
            trial.config.mut_rate,
            trial.config.genetic_iters,
            eps=trial.config.eps,
            seed=trial.config.seed,
            n_arms=trial.config.n_arms,
            add_all_mirrors=trial.config.add_all_mirrors,
        )
    elif trial.config.rand_mdl_name == "quick-block":
        return QuickBlockRandomization(
            trial.config.n_arms,
            trial.config.n_per_arm,
            trial.config.n_cutoff,
            trial.config.min_block_factor,
            trial.config.qb_dir,
            trial.config.rep_to_run,
        )

    elif trial.config.rand_mdl_name == "graph":
        return GraphRandomization(
            trial.config.n,
            trial.config.n_z,
            trial.config.n_cutoff,
            trial.network_dists,
            trial.A,
            seed=trial.config.seed,
        )


def get_estimator(trial):
    """
    Get estimator from name in configuration

    Args:
        - trial (SimulatedTrial)
    """
    if trial.config.estimator_name == "diff-in-means":
        return diff_in_means, get_pval
    elif trial.config.estimator_name == "diff-in-means-mult-arm":
        return (
            partial(
                diff_in_means_mult_arm,
                n_arms=trial.config.n_arms,
                arm_compare_pairs=trial.arm_compare_pairs,
            ),
            partial(
                get_pval_mult_arm,
                n_arms=trial.config.n_arms,
                arm_compare_pairs=trial.arm_compare_pairs,
            ),
        )
    elif trial.config.estimator_name == "clustered-diff-in-means":
        return (
            partial(clustered_diff_in_means, mapping=trial.mapping),
            partial(get_pval_clustered, mapping=trial.mapping),
        )
    elif trial.config.estimator_name == "qb-diff-in-means":
        arms_subdir = f"arms-{trial.config.n_arms}"
        n_per_arm_subdir = f"n-per-arm-{trial.config.n_per_arm}"
        n_cutoff_subdir = f"n-cutoff-{trial.config.n_cutoff}"
        minblock_subdir = f"minblock-{trial.config.min_block_factor}"

        path = (
            Path(trial.config.qb_dir)
            / arms_subdir
            / n_per_arm_subdir
            / n_cutoff_subdir
            / minblock_subdir
        )
        fname = f"{trial.config.rep_to_run}-blocks.csv"
        blocks = (
            pd.read_csv(path / fname).to_numpy()[:, 1].astype(int)
        )  # first column is index, second column is block assignment

        return (
            partial(
                qb_diff_in_means_mult_arm,
                blocks=blocks,
                n_arms=trial.config.n_arms,
                arm_compare_pairs=trial.arm_compare_pairs,
            ),
            partial(
                get_pval_qb,
                blocks=blocks,
                n_arms=trial.config.n_arms,
                arm_compare_pairs=trial.arm_compare_pairs,
            )
        )

        # return QBDiffMeansMultArm(blocks, trial.config.n_arms, trial.arm_compare_pairs)
    else:
        raise ValueError(f"Unrecognized estimator: {trial.config.estimator_name}")
