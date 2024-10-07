import numpy as np
from .measurements import Measurement


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
    
    def update(self, msmt: Measurement):
        """Perform state update with new measurement"""
        if not np.isclose(msmt.t, self.t):
            raise RuntimeError("Propagation and measurement time must be synched")


class KalmanNavigator:
    """Interface for Kalman filter navigation algorithms"""
    def _propagate(self, dt: float):
        pass

    def _update(self, ):
        pass




def PhiFromIMU(dt, dv, dth, rE, qB2E):
    """
    Get state transition and process noise matrices from IMU data
    """
    grav_grad = np.zeros((3, 3))
    fRaw_i_i = dv / dt
    OmRaw_b_b = dth / dt

    # Frame transformations
    ce2i = np.eye(3)
    ci2b = quaternion.as_rotation_matrix(qB2E).T @ ce2i.T

    # State transitions
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)  # kinematics
    A[3:6, 0:3] = grav_grad  # gravity gradient
    A[3:6, 6:9] = -skew(fRaw_i_i)

    # -----------------
    # Accel/Gyro state transitions
    # TODO
    # -----------------

    # Get Phi and Q matrices
    Phi = A_to_STM(A, dt)
    return Phi


def QFromImu(dt, R):
    # Process noise
    B = np.zeros((9, 6))
    B[3:6, 0:3] = np.eye(3)
    B[6:9, 3:6] = np.eye(3)
    Q = B @ (R * dt) @ B.T
    return Q


def A_to_STM(A, dt, order=2):
    """Get state transition matrix from dynamics matrix, A"""
    assert order <= 2
    assert order >= 0

    STM = np.eye(A.shape[0])

    if order >= 1:
        STM += 0.5 * np.linalg.matrix_power(A, 2) * dt**2

    if order >= 2:
        STM += (1 / 6) * np.linalg.matrix_power(A, 3) * dt**3

    return STM


def skew(v):
    """Make skew-symmetric matrix from vector"""
    v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return v
