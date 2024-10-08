from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .measurements import Measurement

import numpy as np


class BaseNavigator:
    def __init__(self, t0: float, x0: np.ndarray, P0: np.ndarray, *args, **kwargs):
        # timing
        self.t0 = t0
        self.t = t0
        self._t_last_update = -np.inf

        # states
        self.x = x0
        self.P = P0

    def propagate(self, t: float):
        """Propagate state to a particular time"""
        dt = t - self.t
        if dt < 0:
            raise RuntimeError("Cannot propagate to a time in the past")
        self._propagate(dt)
        self.t = t

    def _propagate(self, dt: float):
        raise NotImplementedError
    
    def update(self, msmt: "Measurement"):
        """Perform state update with new measurement"""
        if not np.isclose(msmt.t, self.t):
            raise RuntimeError("Propagation and measurement time must be synched")


class KalmanNavigator:
    """Interface for Kalman filter navigation algorithms"""
    def _propagate(self, ):
        pass

    def _update(self, ):
        pass