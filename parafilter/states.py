import numpy as np


class StateVector:
    def __init__(self, t0: float, x0: np.ndarray[float], P0: np.ndarray[float]):
        self.t = t0
        self.x = x0
        self.P = P0

    def __len__(self):
        return len(self.x)
    
    @property
    def x(self) -> np.ndarray:
        return self._x
    
    @x.setter
    def x(self, x: np.ndarray):
        if len(x) != self.n_states:
            raise ValueError(f"Passed x of len {len(x)} but needed {self.n_states} states")
        self._x = x
    
    @property
    def P(self) -> np.ndarray:
        return self._P
    
    @P.setter
    def P(self, P: np.ndarray):
        if (P.shape[0] != self.n_states) or (P.shape[1] != self.n_states):
            raise ValueError(f"Passed P of shape {P.shape} but needed {self.n_states} states")
        self._P = P

    @property
    def position(self) -> np.ndarray:
        return self.x[self.idx_pos]
    
    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.idx_vel]
    
    @property
    def attitude(self) -> np.ndarray:
        return self.x[self.idx_att]
    
    @property
    def R_g2b(self) -> np.ndarray:
        """Get rotation from global to body frame"""
        raise NotImplementedError


class Standard2DState(StateVector):
    n_states: int = 7
    
    def __init__(self, t0: float, x0: np.ndarray[float], P0: np.ndarray[float]):
        """State vector is position, velocity, orientation, bias, scale factor
        
        [x, y, vx, vy, theta, gyro_bias, gyro_sf]
        """
        super().__init__(t0, x0, P0)
        self.idx_pos = np.array([0, 1])
        self.idx_vel = np.array([2, 3])
        self.idx_att = np.array([4])
        self.idx_gerr = np.array([5, 6])

    @property
    def R_g2b(self) -> np.ndarray:
        """Get matrix to transform orientation from global to body frame"""
        th = -self.attitude[0]
        return np.array(
            [
                [np.cos(th), -np.sin(th)],
                [np.sin(th),  np.cos(th)]
            ]
        )