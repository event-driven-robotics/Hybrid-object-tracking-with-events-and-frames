"""
The file is the implement of Extended kalman filter using in
'Event-based camera pose tracking using a generative event model'
"""
import numpy as np


class Gaussian:
    def __init__(self, mean, cov):
        self.mean = np.array(mean).reshape((6, 1))
        self.cov = np.matrix(cov)

    def read_mean(self):
        return self.mean

    def read_cov(self):
        return self.cov


class EKF():
    def __init__(self, prior_state, noise_state, sigma_mu, dimension):
        self.state_estimated = prior_state
        self.state_noise = noise_state
        self.sigma_mu = sigma_mu
        self.dimension = dimension

        # EKF self.F jacobi matrix
        self.F = np.eye(6)

        # KF  H matrix(measurement) & R matrix(measurement noise)
        self.H = np.eye(6)

        self.R = 0.01 * noise_state.read_cov()

    def forward_kinematic_KF(self):
        state_mean_cur = self.state_estimated.read_mean()
        state_cov_cur = self.state_estimated.read_cov()

        # state transition(constant velocity model)
        # 1. mean state(predicted)
        index = np.arange(0, 6, 1)
        noise_cov = self.state_noise.read_cov()
        state_mean_pred = state_mean_cur + np.random.randn(6, 1) * np.array(
            [np.sqrt(noise_cov[i, i]) for i in index]).reshape((6, 1))

        # 2. error cov(predicted)

        # 2.1 predict error cov
        state_cov_pred = np.dot(np.dot(self.F, state_cov_cur), self.F.transpose()) + self.state_noise.read_cov()

        return state_mean_pred.reshape((6, -1)), state_cov_pred

    def forward_kinematic(self):
        state_mean_cur = self.state_estimated.read_mean()
        state_cov_cur = self.state_estimated.read_cov()

        # state transition(constant velocity model)
        # 1. mean state(predicted)
        index = np.arange(0, 6, 1)
        noise_cov = self.state_noise.read_cov()
        state_mean_pred = state_mean_cur*0.5 + np.array(
            [np.random.normal(loc=0.0, scale=np.sqrt(noise_cov[i, i]), size=None) for i in index]).reshape((6, 1))
        # 2. error cov(predicted)

        # 2.1 calculate Jacobi matrix
        F_n = 0.5*self.jacobi_f(self.dimension)
        L_n = self.jacobi_l(self.dimension)

        # 2.2 predict error cov
        state_cov_pred = np.dot(np.dot(F_n, state_cov_cur), F_n.transpose()) + np.dot(
            np.dot(L_n, self.state_noise.read_cov()), L_n.transpose())

        return state_mean_pred.reshape((6, -1)), state_cov_pred

    def implicit_measurement(self, q_xn, h_n):
        # event_measurements
        # from 3D motion to 2D pixel motion field
        # event_measurement_n = Events_generating_v2.event_generative_model_(state_mean_pred)
        # return event_measurement_n
        return q_xn, h_n

    def update_model_state_KF(self, model_state, measurement):
        # integrate measurement in the filter
        P = model_state[1]
        S = np.dot(np.dot(self.H, P), self.H.transpose()) + self.R
        K = np.dot(P, self.H.transpose()).dot(np.linalg.inv(S))
        y = measurement - np.dot(self.H, model_state[0])
        model_state[0] += np.array(np.dot(K, y))
        model_state[1] += model_state[1] - np.dot(np.dot(K, self.H), P)
        self.state_estimated = Gaussian(model_state[0], model_state[1])
        return self.state_estimated

    def update_model_state(self, model_state, event_measurement_n, sigma_mu):
        """
        :param self:
        :param model_state: 6D velocity
        :param event_measurement_n: event measurement q
        :param sigma_mu: measurement noise covariance
        :return:
        """

        # integrate events measurement in the filter

        q_n = event_measurement_n[0]
        h_n = event_measurement_n[1]
        state_mean_pred = model_state[0].reshape((6, -1))
        state_cov_pred = model_state[1]

        # self.dimension = 6(v_x,v_y,v_z,angle x,angle_y,angle_z)

        # update model with each event
        # integrate events measure in error cov
        h_n = h_n.reshape((-1, self.dimension))
        Q_n_mu = sigma_mu
        D_n = np.eye(1)
        R_n = np.dot(np.dot(D_n, Q_n_mu), D_n.transpose())
        S_n = np.dot(np.dot(h_n, state_cov_pred), h_n.transpose()) + R_n

        # Kalman gain
        K_n = np.dot(state_cov_pred, h_n.transpose()) / S_n

        # posterior estimation(state correction)
        state_mean_pred = state_mean_pred + np.dot(K_n, q_n).reshape(self.dimension, 1)
        state_cov_pred = np.dot(
            np.dot((np.identity(self.dimension) - np.dot(K_n, h_n.reshape(1, self.dimension))), state_cov_pred),
            (np.identity(self.dimension) - np.dot(K_n, h_n.reshape(1, self.dimension))).transpose()) + np.dot(np.dot(K_n, R_n),
                                                                                            K_n.transpose())

        self.state_estimated = Gaussian(state_mean_pred, state_cov_pred)
        return self.state_estimated

    @staticmethod
    def jacobi_f(dimension):
        return np.matrix(np.identity(dimension))

    @staticmethod
    def jacobi_l(dimension):
        return np.matrix(np.identity(dimension))