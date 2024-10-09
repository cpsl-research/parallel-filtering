from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .measurements import Measurement

import numpy as np

from estimators import inputs, measurements
from estimators.filters import modular, states


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


class GroundVehicleNavigator:
    def __init__(
        self,
        t0: float,
        x0: List[float],
        s0: float,
        theta0: float,
        gyrob0: float,
        xp0: List[float],
        sp0: float,
        tp0: float,
        gyrop0: float,
        use_imu: bool,
    ):
        """Initialize the ground vehicle's navigator
        
        x0: initial position
        s0: initial speed
        theta0: initial heading angle
        gyrob0: initial gyro bias
        xp0: initial position variance
        sp0: initial speed variance
        tp0: initial heading variance
        gyrop0: initial gyro bias variance
        """
        self.use_imu = use_imu

        # states are constant for all filters
        st_list = [
            states.Position_XY_StateBlock(x=x0[0], y=x0[1], p0=xp0),
            states.SpeedHeading_StateBlock(x=s0, yaw=theta0, p0=[sp0, tp0]),
            states.FixedSensorBias_StateBlock(sensor_ID="gyro", b_init=[gyrob0], p0=[gyrop0])
        ]

        # set up the filter
        self.filter = modular.ModularExtendedKalmanFilter(
            states=states.FilterStateArray(st_list),
            t0=t0,
        )