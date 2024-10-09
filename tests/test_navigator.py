import numpy as np

from estimators.inputs import ConstantVelocityInput
from estimators.measurements import PositionMeasurement_2D_XY
from parafilter.navigator import GroundVehicleNavigator


def test_ground_vehicle_navigator_no_imu():
    # spin up the navigator
    nav = GroundVehicleNavigator(
        t0=0.0,
        x0=[0, 0],
        s0=0.0,
        theta0=0.0,
        gyrob0=0.0,
        xp0=[2.0, 2.0],
        sp0=2.0,
        tp0=0.5,
        gyrop0=1e-4,
        use_imu=False
    )

    # generate some position measurements
    dt = 0.1
    for i in range(1, 10):
        t = i * dt

        # propagate filter
        u = ConstantVelocityInput(dt=dt, sigma_m=2, tau_m=3)
        nav.filter.process_msmts(u=u, msmts=[])

        # make measurement
        x = np.random.randn(2)
        r = np.array([2, 2])
        msmt1 = PositionMeasurement_2D_XY(
            source_ID=0, t=t, r=r, x=x[0], y=x[1]
        )
        msmt1.add_gaussian_noise(r)

        # update filter
        nav.filter.process_msmts(u=None, msmts=[msmt1])