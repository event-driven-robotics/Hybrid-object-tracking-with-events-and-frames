import numpy as np
import Extended_Kalman_Filter


def create_ekf(config):
    """
    Create the EKF used for event-based velocity tracking.

    State:
        [vx, vy, vz, wx, wy, wz]
    """

    # Initial state
    initial_state = Extended_Kalman_Filter.Gaussian(config.initial_state["mean"],
                                                     0.01 * np.diag(config.initial_state["cov"]))

    # process noise omega~N(0,Q)
    process_noise = Extended_Kalman_Filter.Gaussian(config.noise["mean"],
                                                     0.001 * np.diag(config.noise["cov"]))
    # EKF measurement noise~N(0,R)
    sigma_mu = config.sigma_mu

    # x, y, z, roll, yaw and pitch
    state_dimension = 6

    return Extended_Kalman_Filter.EKF(
        initial_state,
        process_noise,
        sigma_mu,
        state_dimension,
    )
