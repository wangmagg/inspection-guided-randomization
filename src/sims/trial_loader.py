import pandas as pd

from src.models.exposure_models import *
from src.models.outcome_models import *
from src.models.network_models import *

from src.design.fitness_functions import *
from src.design.genetic_algorithms import *
from src.design.randomization_designs import *

from src.analysis.estimators import *

class SimulatedTrialLoader:
    """
    Loader for a simulated trial
    """
    def __init__(self, config):
        self.config = config
        return
        
    def get_potential_outcome_mdl(self) -> PotentialOutcomeDGP:
        """
        Get potential outcome model from name in configuration
        """
        if self.config.potential_outcome_mdl_name == "kenya-hierarchical":
            params = pd.read_csv(self.config.params_fname)
            settlement_info = pd.read_csv(self.config.settlement_info_fname, index_col = 'settlement')

            return KenyaPODGP(
                params = params,
                settlement_info = settlement_info,
                coords_range = self.config.coords_range,
                n_per_schl = self.config.n_per_schl,
                tau_size = self.config.tau_size,
                sigma = self.config.sigma,
                seed = self.config.seed
            )
        elif self.config.potential_outcome_mdl_name == "classroom":
            return ClassroomPODGP(
                n_students = self.config.n,
                n_arms = self.config.n_arms,
                sigma = self.config.sigma,
                tau_sizes = self.config.tau_sizes,
                seed = self.config.seed
            )
        elif self.config.potential_outcome_mdl_name == "classroom-composition":
            return ClassroomFixedMalePODGP(
                n_students = self.config.n, 
                sigma = self.config.sigma,
                tau_size = self.config.tau_size, 
                tau_comp_sizes = self.config.tau_comp_sizes,
                seed = self.config.seed
            )
        
    def get_full_potential_outcome_mdl_name(self) -> str:
        """
        Get full name of potential outcome model from name in configuration
        """
        if self.config.potential_outcome_mdl_name == "kenya-hierarchical":
            return f'kenya-hierarchical_tau-{self.config.tau_size:.2f}'
        elif self.config.potential_outcome_mdl_name == "classroom":
            return f'classroom_tau-{self.config.tau_size:.2f}'
        else:
            raise ValueError(f"Unrecognized potential outcome model: {self.config.potential_outcome_mdl_name}")
    
    def get_expo_mdl(self) -> ExposureModel:
        """
        Get exposure model from name in configuration
        """
        if self.config.expo_mdl_name == "frac-nbr-expo":
            return FracNbr(self.config.q)
        elif self.config.expo_mdl_name == "one-nbr-expo":
            return OneNbr()
        else:
            raise ValueError(f"Unrecognized exposure model: {self.config.expo_mdl_name}")
    
    def get_spatial_intxn_mdl(self):
        """
        Get spatial interaction model from name in configuration
        """
        if self.config.intxn_mdl_name == "power-decay":
            return EuclideanDistPowerDecayIntxn(self.config.gamma)
        else:
            raise ValueError(f"Unrecognized spatial interaction model: {self.config.intxn_mdl_name}")
    
    def get_full_spatial_intxn_mdl_name(self):
        """
        Get full name of spatial interaction model from name in configuration
        """
        if self.config.intxn_mdl_name == "power-decay":
            return f'euclidean-dist-power-decay-gamma-{self.config.gamma}'
        else:
            raise ValueError(f"Unrecognized spatial interaction model: {self.config.intxn_mdl_name}")
    
    def get_net_mdl(self):
        """
        Get network model from name in configuration
        """
        if "sb" in self.config.net_mdl_name:
            return StochasticBlock(self.config.wi_p, self.config.bw_p, self.config.seed)
        else:
            raise ValueError(f"Unrecognized network model: {self.config.net_mdl_name}")
    
    def get_full_net_mdl_name(self) -> str:
        """
        Get full name of network model from name in configuration
        """
        if "sb" in self.config.net_mdl_name:
            return f'sb_wip-{self.config.wi_p:.2f}_bwp-{self.config.bw_p:.2f}'
        else:
            raise ValueError(f"Unrecognized network model: {self.config.net_mdl_name}")
        
    def get_observed_outcome_mdl(self, trial):
        """
        Get observed outcome model from name in configuration

        Args:
            - trial (SimulatedTrial): trial to get observed outcome model for 
        """
        if self.config.observed_outcome_mdl_name == "additive-interference":
            if hasattr(trial, 'mapping'):
                return AdditiveInterference(self.config.delta_size, trial.expo_mdl, trial.A, trial.mapping)
            return AdditiveInterference(self.config.delta_size, trial.expo_mdl, trial.A)
        elif self.config.observed_outcome_mdl_name == "consistent":
            return Consistent()
        elif self.config.observed_outcome_mdl_name == "additive-composition":
            return AdditiveComposition(trial.arm_pairs)
        else:
            raise ValueError(f"Unrecognized outcome model: {self.config.observed_outcome_mdl_name}")
        
    def get_fitness_fn(self, trial):
        """
        Get fitness function from name in configuration

        Args:
            - trial (SimulatedTrial): trial to get fitness function for
        """
        if self.config.fitness_fn_name == "smd":
            return SMD(trial.X_fit)
        elif self.config.fitness_fn_name == "weighted-smd":
            return SMD(trial.X_fit, weights=self.config.fitness_fn_weights, covar_to_weight=trial.covar_to_weight)
        elif self.config.fitness_fn_name == "max-abs-smd":
            return MaxAbsSMD(trial.X_fit)
        elif self.config.fitness_fn_name == "signed-max-abs-smd":
            return SignedMaxAbsSMD(trial.X_fit)
        elif self.config.fitness_fn_name == "weighted-max-abs-smd":
            return MaxAbsSMD(trial.X_fit, weights=self.config.fitness_fn_weights, covar_to_weight=trial.covar_to_weight)
        elif self.config.fitness_fn_name == "weighted-sum-smd":
            return SumSMD(trial.X_fit, mapping=trial.mapping,
                          weights=self.config.fitness_fn_weights, covar_to_weight=trial.covar_to_weight)
        elif self.config.fitness_fn_name == "sum-max-smd":
            return SumMaxSMD(trial.X_fit, mapping=trial.mapping)
        elif self.config.fitness_fn_name == "weighted-sum-max-smd":
            return SumMaxSMD(trial.X_fit, mapping=trial.mapping, 
                             weights=self.config.fitness_fn_weights, covar_to_weight=trial.covar_to_weight)
        elif self.config.fitness_fn_name == "max-mahalanobis":
            return MaxMahalanobis(trial.X_fit, trial.mapping)
        elif self.config.fitness_fn_name == 'weighted-max-mahalanobis':
            return WeightedMaxMahalanobis(trial.X_fit, trial.beta, trial.mapping)
        elif self.config.fitness_fn_name == "frac-expo":
            return FracExposed(trial.expo_mdl, trial.A, trial.mapping)
        elif self.config.fitness_fn_name == "min-pairwise-euclidean-dist":
            return MinPairwiseEuclideanDist(trial.pairwise_dists)
        elif self.config.fitness_fn_name == "lin-comb_max-abs-dist_frac-expo":
            return SumMaxSMDFracExpo(trial.X_fit, trial.expo_mdl, trial.A, trial.mapping, 
                                      *self.config.fitness_fn_weights)
        elif self.config.fitness_fn_name == "lin-comb_max-abs-dist_euclidean-dist":
            return SumMaxSMDMinPairwiseEuclideanDist(trial.X_fit, trial.pairwise_dists, trial.mapping,
                                                      *self.config.fitness_fn_weights)
        elif self.config.fitness_fn_name == "lin-comb_max-mahalanobis_frac-expo":
            return MaxMahalanobisFracExpo(trial.X_fit, trial.expo_mdl, trial.A, trial.mapping, 
                                          *self.config.fitness_fn_weights)
        elif self.config.fitness_fn_name == "lin-comb_max-mahalanobis_euclidean-dist":
            return MaxMahalanobisMinPairwiseEuclideanDist(trial.X_fit, trial.pairwise_dists, trial.mapping, 
                                                          *self.config.fitness_fn_weights)
        elif self.config.fitness_fn_name == "lin-comb_weighted-max-mahalanobis_frac-expo":
            return WeightedMaxMahalanobisFracExpo(trial.X_fit, trial.beta, trial.expo_mdl, trial.A, trial.mapping, 
                                          *self.config.fitness_fn_weights)
        elif self.config.fitness_fn_name == "lin-comb_weighted-max-mahalanobis_euclidean-dist":
            return WeightedMaxMahalanobisMinPairwiseEuclideanDist(trial.X_fit, trial.beta, trial.pairwise_dists, trial.mapping, 
                                                                  *self.config.fitness_fn_weights)
        else:
            raise ValueError(f"Unrecognized fitness function: {self.config.fitness_fn_name}")
    
    def get_design(self, trial):
        """
        Get randomization design from name in configuration

        Args:
            - trial (SimulatedTrial): trial to get randomization design for
        """
        if self.config.rand_mdl_name == "complete":
            return CompleteRandomization(trial.config.n, self.config.n_z, self.config.n_cutoff, n_arms=self.config.n_arms)
        elif self.config.rand_mdl_name == "restricted":
            fitness_fn = self.get_fitness_fn(trial)
            return RestrictedRandomization(trial.config.n, self.config.n_z, self.config.n_cutoff, fitness_fn, seed=self.config.seed, n_arms=self.config.n_arms)
        elif self.config.rand_mdl_name == "restricted-genetic":
            fitness_fn = self.get_fitness_fn(trial)
            return RestrictedRandomizationGenetic(trial.config.n, self.config.n_z, self.config.n_cutoff, fitness_fn, 
                                                  self.config.tourn_size, self.config.cross_k, self.config.cross_rate, 
                                                  self.config.mut_rate, self.config.genetic_iters, seed=self.config.seed, n_arms=self.config.n_arms)
        elif self.config.rand_mdl_name == "group-formation":
            return PairedGroupFormationRandomization(trial.config.n, self.config.n_z, self.config.n_cutoff,
                                                     trial.X_on, self.config.p_comps, seed=self.config.seed, n_arms=self.config.n_arms)
        elif self.config.rand_mdl_name == "group-formation-restricted":
            fitness_fn = self.get_fitness_fn(trial)
            return PairedGroupFormationRestrictedRandomization(trial.config.n, self.config.n_z, self.config.n_cutoff,  
                                                               trial.X_on, self.config.p_comps, fitness_fn, n_arms=2, seed=42)
        elif self.config.rand_mdl_name == "quick-block":
            return QuickBlockRandomization(self.config.n_arms, 
                                           self.config.n_per_arm, 
                                           self.config.n_cutoff, 
                                           self.config.min_block_factor,
                                           self.config.qb_dir, 
                                           self.config.rep_to_run)

    def get_estimator(self, trial):
        """
        Get estimator from name in configuration

        Args:
            - trial (SimulatedTrial): trial to get estimator for
        """
        if self.config.estimator_name == "diff-in-means":
            return DiffMeans(trial.mapping)
        elif self.config.estimator_name == "diff-in-means-mult-arm":
            return DiffMeansMultArm(self.config.n_arms, trial.arm_pairs)
        elif self.config.estimator_name == "clustered-diff-in-means":
            estimator = ClusteredDiffMeans(trial.mapping)
            estimator.cache_cluster_masks(trial.z_pool)
            return estimator
        else:
            raise ValueError(f"Unrecognized estimator: {self.config.estimator_name}")
        
    
