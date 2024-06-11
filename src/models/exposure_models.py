import numpy as np
from abc import ABC, abstractmethod
import time
from tqdm import tqdm

try:
    import cupy as cp
    USE_GPU=True
except ModuleNotFoundError:
    USE_GPU=False

class ExposureModel(ABC):
    """
    Abstract class for exposure models.
    """
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class OneNbr(ExposureModel):
    """
    Exposure model where a unit is exposed if it has at least 1 treated neighbor
    Assumes binary treatment assignment
    """
    def __init__(self):
        self.name = 'one-nbr-expo'
    def __call__(self, 
                 z: np.ndarray, 
                 A: np.ndarray) -> np.ndarray:
        """
        Get the exposure of each unit

        Args:
            z: binary treatment assignment
            A: adjacency matrix
        """
        return np.dot(z, A) >= 1 

class FracNbr(ExposureModel):
    """
    Exposure model where a unit is exposed if at least q fraction of its neighbors are treated
    Assumes binary treatment assignment

    Args:
        q: fraction of treated neighbors required for a unit to be exposed
    """
    def __init__(self, q: float):
        self.q = q
        self.name = f'frac-nbr-expo-{q:.2f}'
    def __call__(self, 
                 z: np.ndarray, 
                 A: np.ndarray) -> np.ndarray:
        """
        Get the exposure of each unit

        Args:
            z: binary treatment assignment
            A: adjacency matrix
        """
        time_start = time.time()
        if USE_GPU:
           A = cp.array(A)
           z = cp.array(z)
           n_z1_nbrs = cp.dot(A.T, z.T).T
        else:
           n_z1_nbrs = np.dot(A.T, z.T).T
        time_end = time.time()

        time_start = time.time()
        n_nbrs = np.sum(A, axis=0)
        time_end = time.time()

        time_start = time.time()
        is_expo = n_z1_nbrs >= (self.q * n_nbrs)
        time_end = time.time()

        return is_expo
