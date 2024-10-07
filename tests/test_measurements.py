import numpy as np

from parafilter.entities import Landmark
from parafilter.measurements import AzimuthToLandmarkFor2D, RangeToLandmarkFor2D
from parafilter.states import Standard2DState


def test_range_to_landmark():
    lm = Landmark(x=np.array([10, 12]))
    x0 = np.array([5, 6, 0, 0, 0.1, 0, 0])
    X = Standard2DState(t0=0, x0=x0, P0=np.diag(np.ones(len(x0))))
    dx = lm.position - X.position
    M = RangeToLandmarkFor2D(t=0.0, z=np.linalg.norm(dx), R=np.diag([3]))
    assert np.isclose(M.h(X, lm), M.z)


def test_azimuth_to_landmark_no_angle():
    lm = Landmark(x=np.array([10, 12]))
    x0 = np.array([5, 6, 0, 0, 0, 0, 0])
    X = Standard2DState(t0=0, x0=x0, P0=np.diag(np.ones(len(x0))))
    dx = lm.position - X.position
    M = AzimuthToLandmarkFor2D(t=0.0, z=np.arctan2(dx[1], dx[0]), R=np.diag([1]))
    assert np.isclose(M.h(X, lm), M.z)


def test_azimuth_to_landmark_angle():
    lm = Landmark(x=np.array([10, 12]))
    x0 = np.array([5, 6, 0, 0, 0.1, 0, 0])
    X = Standard2DState(t0=0, x0=x0, P0=np.diag(np.ones(len(x0))))
    dx = lm.position - X.position
    az = np.arctan2(dx[1], dx[0]) - X.attitude[0]
    M = AzimuthToLandmarkFor2D(t=0.0, z=az, R=np.diag([1]))
    assert np.isclose(M.h(X, lm), M.z)
