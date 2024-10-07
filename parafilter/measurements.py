from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .entities import Landmark
    from .states import StateVector

import numpy as np


class Measurement:
    def __init__(self, t: float, z: np.ndarray[float], R: np.ndarray[float]):
        self.t = t
        self.z = z
        self.R = R
    
    def __len__(self) -> int:
        return len(self.z)

    def h(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def H(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


"""
The following are measurements that assume the state is a 2D cartesian state

e.g. to be used for David's ground vehicle
"""

class Position2DFor2D(Measurement):
    """Used in the update equation"""
    def h(self, X: "StateVector", *args, **kwargs) -> np.ndarray:
        return X.position[:2]
    
    def H(self, X: "StateVector", *args, **kwargs) -> np.ndarray:
        H = np.zeros((len(self), len(X)))
        np.fill_diagonal(H, 1)
        return H


class RangeToLandmarkFor2D(Measurement):
    """Used in the update equation
    
    defined as a measurement from the platform body to a landmark
    """
    def h(self, X: "StateVector", landmark: "Landmark") -> float:
        return np.linalg.norm(landmark.position[:2] - X.position[:2])
    
    def H(self, X: "StateVector", landmark: "Landmark") -> np.ndarray:
        H = np.zeros((len(self), len(X)))
        dx = landmark.position[:2] - X.position[:2]
        H[0,:] = dx / np.linalg.norm(dx)
        return H
    

class AzimuthToLandmarkFor2D(Measurement):
    """Used in the update equation
    
    defined as a measurement from the platform body to a landmark
    azimuth = 0 defined along x-axis
    """
    def h(self, X: "StateVector", landmark: "Landmark") -> float:
        dx_local = X.R_g2b @ (landmark.position[:2] - X.position[:2])
        return np.arctan2(dx_local[1], dx_local[0])
    
    def H(self, X: "StateVector", landmark: "Landmark") -> np.ndarray:
        H = np.zeros((len(self), len(X)))
        dx_local = X.R_g2b @ (landmark.position[:2] - X.position[:2])
        r2d = np.linalg.norm(dx_local[:2])
        H[1, 0] = -dx_local[1] / r2d**2
        H[1, 1] =  dx_local[0] / r2d**2
        return H


class GyroFor2D:
    """Used in the propagation equation"""
    pass