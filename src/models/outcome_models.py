import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional

class PotentialOutcomeDGP(ABC):
    """
    Abstract class for covariate and potential outcome data generating processes.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class ClassroomPODGP(PotentialOutcomeDGP):
    """
    Data generating process for simulated classroom experiment

    Args:
        - n_students: number of students
        - sigma: standard deviation of noise
        - n_arms: number of treatment arms
        - tau_sizes: sizes of treatment effects
        - seed: random seed
    """
    def __init__(self, 
                 n_students: int, 
                 sigma: float, 
                 n_arms: int, 
                 tau_sizes: Union[List[float], np.ndarray[float]], 
                 seed): 
        self.name = "classroom"
        self.n_students = n_students
        self.sigma = sigma
        self.n_arms = n_arms
        self.tau_sizes = tau_sizes
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Simulate covariates and potential outcomes

        Returns:
            - y_0: potential outcomes for control
            - y_arms: potential outcomes for treatment arms
            - X: covariates
        """
        male = self.rng.binomial(1, 0.7, self.n_students)

        age = self.rng.uniform(19, 25, self.n_students)
        age_z = (age - np.mean(age)) / np.std(age)

        major = self.rng.choice(3, self.n_students, p=[0.5, 0.3, 0.2])

        ability = (
            age_z
            - (major == 1)
            - male
            + self.rng.normal(0, self.sigma, self.n_students)
        )
        confidence = (
            male + (major == 2) + self.rng.normal(0, self.sigma, self.n_students)
        )

        hw = ability + self.rng.normal(0, self.sigma, self.n_students)

        X = pd.DataFrame.from_dict({'male': male,
                                    'age': age,
                                    'major': major,
                                    'ability': ability,
                                    'hw': hw})
        
        # Simulate potential outcomes for control
        y_0 = ability + confidence + self.rng.normal(0, self.sigma, self.n_students)

        # Simulate potential outcomes for treatment arms
        y_arms = []
        for tau_size in self.tau_sizes:
            y_arm = y_0 + tau_size * np.std(y_0)
            y_arms.append(y_arm)
        y_arms = np.vstack(y_arms)

        return y_0, y_arms, X
    
class ClassroomFixedMalePODGP(PotentialOutcomeDGP):
    """
    Data generating process for simulated classroom experiment with fixed total
    number of male students (used for simulated group formation experiment)

    Args:
        - n_students: number of students
        - sigma: standard deviation of noise
        - n_arms: number of treatment arms
        - tau_size: size of treatment effect
        - tau_comp_sizes: sizes of treatment effects in different group compositions
        - seed: random seed
    """
    def __init__(self, 
                 n_students: int, 
                 sigma: float, 
                 tau_size: float, 
                 tau_comp_sizes: Union[List[float], np.ndarray[float]],
                 seed: int):
        self.name = "classroom-composition"
        self.n_students = n_students
        self.sigma = sigma
        self.tau_size = tau_size
        self.tau_comp_sizes = tau_comp_sizes
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Simulate covariates and potential outcomes

        Returns:
            - y_0: potential outcomes for control
            - y_1_comps: potential outcomes for treatment arms in different group compositions
            - X: covariates
        """
        male = np.zeros(self.n_students)
        male[:self.n_students // 2] = 1
        self.rng.shuffle(male)

        age = self.rng.uniform(19, 25, self.n_students)
        age_z = (age - np.mean(age)) / np.std(age)

        major = self.rng.choice(3, self.n_students, p=[0.5, 0.3, 0.2])

        ability = (
            age_z
            - (major == 1)
            - male
            + self.rng.normal(0, self.sigma, self.n_students)
        )
        confidence = (
            male + (major == 2) + self.rng.normal(0, self.sigma, self.n_students)
        )

        hw = ability + self.rng.normal(0, self.sigma, self.n_students)

        X = pd.DataFrame.from_dict({'male': male,
                                    'age': age,
                                    'major': major,
                                    'ability': ability,
                                    'hw': hw})
        
        # Simulate potential outcomes for control
        y_0 = ability + confidence + self.rng.normal(0, self.sigma, self.n_students)

        # Simulate potential outcomes for treatment arms, for each group composition
        y_1_comps = []
        for tau_comp_size in self.tau_comp_sizes:
            y_1_comp = y_0 + (self.tau_size + tau_comp_size) * np.std(y_0)
            y_1_comps.append(y_1_comp)
        y_1_comps = np.vstack(y_1_comps)

        return y_0, y_1_comps, X

class KenyaPODGP(PotentialOutcomeDGP):
    """
    Data generating process for simulated Kenyan school experiment

    Args:
        - params: parameters for covariate distributions
        - settlement_info: information about settlements
        - coords_range: range of school coordinates within each settlement
        - n_per_sch: number of students per school
        - n_lower_clip: lower clip for number of students per school
        - n_upper_clip: upper clip for number of students per school
        - tau_size: size of treatment effect
        - sigma: standard deviation of noise
        - seed: random seed
    """
    def __init__(
        self, 
        params: pd.DataFrame,
        settlement_info: pd.DataFrame,
        coords_range: float,
        n_per_schl: int,
        tau_size: float,
        sigma: float,
        seed: int
    ):
        self.name = f"kenya-hierarchical_tau-{tau_size:.2f}"
        self.params = params
        self.settlement_info = settlement_info
        self.settlement_coords_range = coords_range
        self.n_per_schl = n_per_schl
        self.tau_size = tau_size
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def _sample_mus(self, 
                    mus: np.ndarray, 
                    sigmas: np.ndarray, 
                    rng: np.random.Generator) -> np.ndarray:
        """
        Sample normally distributed means

        Args:
            - mus: means 
            - sigmas: standard deviations
            - rng: random number generator

        Returns:
            - mus: sampled means
        """
        return rng.normal(loc = mus, scale = sigmas)

    def _sample_nested_covariates(self, 
                                  mus: np.ndarray, 
                                  sigmas: np.ndarray, 
                                  n: int, 
                                  covariate_names: Union[List[str], np.ndarray[str]], 
                                  rng: np.random.Generator) -> pd.DataFrame:
        """
        Sample covariates 

        Args:
            - mus: means
            - sigmas: standard deviations
            - n: number of covariates
            - covariate_names: names of covariates
            - rng: random number generator

        Returns:
            - covariate_df: dataframe of covariates
        """
        covariates = [rng.normal(loc = mus, scale = sigmas) for _ in range(n)]
        covariates = np.vstack(covariates)
        covariate_df = pd.DataFrame(covariates, columns=covariate_names)

        return covariate_df
    
    def sample_school_coords(self) -> pd.DataFrame:
        """
        Sample school coordinates

        Returns:
            dataframe of school coordinates
        """
        all_sch_coords = []

        # Sample coordinates of schools for each settlement
        for i, (settlement, data) in enumerate(self.settlement_info.iterrows()):
            n_sch = data['n_sch'].astype(int)

            sch_coords = {}
            
            # Define settlement bounds for school coordinates
            set_coords_lon_bounds = data['coord_lon'] + np.array([-1, 1]) * self.settlement_coords_range
            set_coords_lat_bounds = data['coord_lat'] + np.array([-1, 1]) * self.settlement_coords_range

            # Sample school coordinates in a grid without replacement
            sch_coords["coord_lon"] = self.rng.choice(np.linspace(set_coords_lon_bounds[0], set_coords_lon_bounds[1], 200), 
                                                          size=n_sch, 
                                                          replace=False)
            sch_coords["coord_lat"] = self.rng.choice(np.linspace(set_coords_lat_bounds[0], set_coords_lat_bounds[1], 200), 
                                                          size=n_sch, 
                                                          replace=False)
            
            # Add settlement id and within-settlement school id
            sch_coords["settlement"] = settlement
            sch_coords["settlement_id"] = i
            sch_coords["school"] = range(n_sch)
            sch_coords_df = pd.DataFrame.from_dict(sch_coords)

            all_sch_coords.append(sch_coords_df)

        all_sch_coords = pd.concat(all_sch_coords)

        # Add global school id
        all_sch_coords["school_id"] = np.arange(all_sch_coords.shape[0])

        return all_sch_coords
    
    def sample_individual_covariates(self) -> pd.DataFrame:
        """
        Sample individual-level covariates in a hierarchical structure

        Returns:
            dataframe of individual-level covariates
        """

        all_ind_covariates = []

        # Sample individual-level covariates for each school in each settlement
        for i, (settlement, data) in enumerate(self.settlement_info.iterrows()):

            # Sample school-level means 
            sch_mus = self._sample_mus(
                mus = self.params.loc[self.params["level"] == "ind"]["mu"],
                sigmas = self.params.loc[self.params["level"] == "ind"]["sigma_set"],
                rng = self.rng
            )
            n_sch = data['n_sch'].astype(int)
            for school in range(n_sch):

                # Sample individual-level means, centered around school-level means
                ind_mus = self._sample_mus(
                    mus = sch_mus,
                    sigmas = self.params.loc[self.params["level"] == "ind"]["sigma_sch_in_set"],
                    rng = self.rng
                )

                # Sample individual-level covariates, centered around individual-level means
                ind_covariates = self._sample_nested_covariates(
                    mus = ind_mus,
                    sigmas = self.params.loc[self.params["level"] == "ind"]["sigma_ind_in_sch"],
                    n = self.n_per_schl,
                    covariate_names = self.params.loc[self.params["level"] == "ind"]["var_name"].values,
                    rng = self.rng
                )

                # Add settlement, settlement id, and school id
                ind_covariates["settlement"] = settlement
                ind_covariates["settlement_id"] = i
                ind_covariates["school"] = school
                all_ind_covariates.append(ind_covariates)
                
        all_ind_covariates = pd.concat(all_ind_covariates)
        return all_ind_covariates
    
    def __call__(self):
        """
        Simulate covariates and potential outcomes

        Returns:
            - y_0: potential outcomes for control
            - y_1: potential outcomes for treatment
            - X: covariates
            - beta_shared: coefficients
        """

        # Sample individual-level covariates and school coordinates
        all_ind = self.sample_individual_covariates()
        all_school_coords = self.sample_school_coords()

        # Merge individual-level covariates and school coordinates
        # to create a dataframe of covariates
        X = pd.merge(
            all_ind,
            all_school_coords,
            on = ["settlement", "settlement_id", "school"]
        )

        # Get coefficients for covariates that exist in X
        beta = self.params[['var_name', 'beta']].set_index('var_name').transpose()
        shared = X.columns.intersection(beta.columns)
        beta_shared = beta[shared].to_numpy().squeeze()
        X_shared_centered = (X[shared] - X[shared].mean()).to_numpy()

        # Simulate potential outcomes for control and treatment using linear model
        eps = self.rng.normal(0, self.sigma, X_shared_centered.shape[0])
        y_0 = np.matmul(X_shared_centered, beta_shared.transpose()).squeeze() + eps
        y_1 = y_0 + self.tau_size * np.std(y_0)

        return y_0, y_1, X, beta_shared

class ObservedOutcomeDGP(ABC):
    """
    Abstract class for observed outcome data generating processes.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class Consistent(ObservedOutcomeDGP):
    """
    Data generating process for observed outcomes consistent with generated potential outcomes
    """
    def __init__(self):
        self.name = "consistent"
    def __call__(self, 
                 z: np.ndarray, 
                 y0: np.ndarray, 
                 y_arms: np.ndarray) -> np.ndarray:
        """
        Generate observed outcomes

        Args:
            - z: treatment assignment
            - y0: potential outcomes for control
            - y_arms: potential outcomes for treatment arms
        """
        y_obs = (z == 0) * y0
        for i in range(len(y_arms)):
            y_obs += (z == (i+1)) * y_arms[i]
        return y_obs
    
class AdditiveComposition(ObservedOutcomeDGP):
    """
    Data generating process for observed outcomes with additive composition effects

    Args:
        - arm_pairs: pairs of groups with control and treatment labels
    """
    def __init__(self, 
                 arm_pairs: np.ndarray):
        self.name = "additive-composition"
        self.arm_pairs = arm_pairs
    def __call__(self, 
                 z: np.ndarray, 
                 y0: np.ndarray, 
                 y_1_comps: np.ndarray) -> np.ndarray:
        """
        Generate observed outcomes

        Args:
            - z: treatment assignment
            - y0: potential outcomes for control
            - y_1_comps: potential outcomes for treatment arms in different group compositions
        """
        y_obs = np.zeros((z.shape[0],  z.shape[1]))

        # Generate observed outcomes for each composition type
        for comp_lbl, (ctrl_grp_lbl, trt_grp_lbl) in enumerate(self.arm_pairs):
            y_obs += (z == ctrl_grp_lbl) * y0 + (z == trt_grp_lbl) * (y_1_comps[comp_lbl])
        return y_obs

class AdditiveInterference(ObservedOutcomeDGP):
    """
    Data generating process for observed outcomes with additive interference effects

    Args:
        - delta_size: size of spillover effect
        - expo_mdl (ExposureModel): instance of exposure model
        - A: adjacency matrix
        - mapping: mapping from individual-level to group-level treatment assignment
    """
    def __init__(self, 
                 delta_size: float, 
                 expo_mdl: callable, 
                 A: np.ndarray,
                 mapping: Optional[np.ndarray]=None):
        self.name = f"additive-{delta_size:.2f}"
        self.delta_size = delta_size
        self.expo_mdl = expo_mdl
        self.A = A
        self.mapping = mapping

    def __call__(self, 
                 z: np.ndarray, 
                 y_0: np.ndarray, 
                 y_1: np.ndarray) -> np.ndarray:
        """
        Generate observed outcomes

        Args:
            - z: treatment assignment
            - y_0: potential outcomes for control
            - y_1: potential outcomes for treatment
        """
        if self.mapping is not None:
            if z.ndim == 2:
                z = np.vstack([z_single[self.mapping] for z_single in z])
            else:
                z = z[self.mapping]

        # Generate observed outcomes with additive effect for interference
        y_obs = z * y_1 + (1 - z) * y_0 + (1 - z) * self.delta_size * np.std(y_0) * self.expo_mdl(z, self.A)

        return y_obs