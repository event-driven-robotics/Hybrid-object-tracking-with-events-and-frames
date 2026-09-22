"""A testable Unscented Kalman Filter for position and orientation.

The filter state is ``[x, y, z, qw, qx, qy, qz]``.  Its covariance lives in
the 6-D tangent space: position (3) plus rotation-vector error (3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DIM = 6
POSE_DIM = 7


@dataclass(frozen=True)
class PoseUKFConfig:
    """Numerical and noise parameters for :class:`PoseUKF`."""

    process_noise: np.ndarray
    measurement_noise: np.ndarray
    dt: float
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        for name, matrix in (
            ("process_noise", self.process_noise),
            ("measurement_noise", self.measurement_noise),
        ):
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (STATE_DIM, STATE_DIM):
                raise ValueError(f"{name} must have shape ({STATE_DIM}, {STATE_DIM}).")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")


class PoseUKF:
    """UKF that predicts a 6-DoF pose from linear and angular velocity.

    ``predict`` accepts ``[vx, vy, vz, wx, wy, wz]``.  ``update`` accepts a
    pose in ``[x, y, z, qw, qx, qy, qz]`` order.  No dataset I/O, plotting, or
    rendering code belongs in this class.
    """

    def __init__(self, initial_pose: np.ndarray, initial_covariance: np.ndarray, config: PoseUKFConfig):
        self.config = config
        self.mean = self._as_pose(initial_pose)
        self.covariance = self._as_covariance(initial_covariance)

    def predict(self, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the posterior pose using a body-frame velocity measurement."""
        velocity = np.asarray(velocity, dtype=float).reshape(6)
        sigma, wm, wc = self._augmented_sigma_points(self.mean, self.covariance, self.config.process_noise)
        propagated = np.array(
            [self._motion_model(point[:POSE_DIM], velocity, point[POSE_DIM:]) for point in sigma]
        )
        self.mean = self._pose_mean(propagated, wm)
        self.covariance = self._pose_covariance(propagated, self.mean, wc)
        return self.mean.copy(), self.covariance.copy()

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Correct the current prediction with a pose measurement."""
        measurement = self._as_pose(measurement)
        sigma, wm, wc = self._augmented_sigma_points(
            self.mean, self.covariance, self.config.measurement_noise
        )
        state_sigma = sigma[:, :POSE_DIM]
        measured_sigma = np.array(
            [self._measurement_model(point[:POSE_DIM], point[POSE_DIM:]) for point in sigma]
        )
        measured_mean = self._pose_mean(measured_sigma, wm)
        innovation_covariance = self._pose_covariance(measured_sigma, measured_mean, wc)

        cross_covariance = np.zeros((STATE_DIM, STATE_DIM))
        for state, predicted_measurement, weight in zip(state_sigma, measured_sigma, wc):
            cross_covariance += weight * np.outer(
                self._pose_error(state, self.mean),
                self._pose_error(predicted_measurement, measured_mean),
            )

        gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
        correction = gain @ self._pose_error(measurement, measured_mean)
        self.mean = self._apply_error(self.mean, correction)
        self.covariance = self._stabilize(self.covariance - gain @ innovation_covariance @ gain.T)
        return self.mean.copy(), self.covariance.copy()

    def _augmented_sigma_points(
        self, mean: np.ndarray, covariance: np.ndarray, noise_covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        augmented_covariance = np.zeros((2 * STATE_DIM, 2 * STATE_DIM))
        augmented_covariance[:STATE_DIM, :STATE_DIM] = covariance
        augmented_covariance[STATE_DIM:, STATE_DIM:] = noise_covariance
        augmented_mean = np.zeros(2 * STATE_DIM)
        return self._sigma_points(augmented_mean, augmented_covariance, mean)

    def _sigma_points(
        self, mean_error: np.ndarray, covariance: np.ndarray, reference_pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dimension = mean_error.size
        scale = self.config.alpha**2 * (dimension + self.config.kappa)
        lambda_ = scale - dimension
        try:
            root = np.linalg.cholesky(self._stabilize((dimension + lambda_) * covariance))
        except np.linalg.LinAlgError as error:
            raise ValueError("Covariance must be positive definite.") from error

        errors = np.vstack((mean_error, mean_error + root.T, mean_error - root.T))
        poses = np.array([self._apply_error(reference_pose, item[:STATE_DIM]) for item in errors])
        points = np.hstack((poses, errors[:, STATE_DIM:]))
        wm = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + lambda_)))
        wc = wm.copy()
        wm[0] = lambda_ / (dimension + lambda_)
        wc[0] = wm[0] + (1.0 - self.config.alpha**2 + self.config.beta)
        return points, wm, wc

    def _motion_model(self, pose: np.ndarray, velocity: np.ndarray, noise: np.ndarray) -> np.ndarray:
        linear, angular = velocity[:3], velocity[3:]
        position = pose[:3] + (linear + np.cross(angular, pose[:3])) * self.config.dt + noise[:3]
        orientation = self._quaternion_multiply(self._rotation_vector_to_quaternion(noise[3:]), pose[3:])
        orientation = self._quaternion_multiply(self._rotation_vector_to_quaternion(angular * self.config.dt), orientation)
        return np.concatenate((position, self._normalize_quaternion(orientation)))

    def _measurement_model(self, pose: np.ndarray, noise: np.ndarray) -> np.ndarray:
        return self._apply_error(pose, noise)

    @staticmethod
    def _pose_mean(poses: np.ndarray, weights: np.ndarray) -> np.ndarray:
        position = weights @ poses[:, :3]
        quaternion_matrix = sum(weight * np.outer(pose[3:], pose[3:]) for pose, weight in zip(poses, weights))
        _, eigenvectors = np.linalg.eigh(quaternion_matrix)
        quaternion = eigenvectors[:, -1]
        if quaternion[0] < 0:
            quaternion *= -1
        return np.concatenate((position, quaternion))

    @classmethod
    def _pose_covariance(cls, poses: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> np.ndarray:
        covariance = sum(weight * np.outer(cls._pose_error(pose, mean), cls._pose_error(pose, mean)) for pose, weight in zip(poses, weights))
        return cls._stabilize(covariance)

    @classmethod
    def _apply_error(cls, pose: np.ndarray, error: np.ndarray) -> np.ndarray:
        return np.concatenate((pose[:3] + error[:3], cls._quaternion_multiply(cls._rotation_vector_to_quaternion(error[3:]), pose[3:])))

    @classmethod
    def _pose_error(cls, pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
        delta = cls._quaternion_multiply(pose[3:], cls._quaternion_conjugate(reference[3:]))
        return np.concatenate((pose[:3] - reference[:3], cls._quaternion_to_rotation_vector(delta)))

    @staticmethod
    def _as_pose(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=float).reshape(POSE_DIM)
        pose[3:] = PoseUKF._normalize_quaternion(pose[3:])
        return pose

    @staticmethod
    def _as_covariance(covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (STATE_DIM, STATE_DIM):
            raise ValueError(f"initial_covariance must have shape ({STATE_DIM}, {STATE_DIM}).")
        return PoseUKF._stabilize(covariance)

    @staticmethod
    def _stabilize(matrix: np.ndarray) -> np.ndarray:
        symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
        return symmetric + np.eye(symmetric.shape[0]) * 1e-12

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Quaternion must not be zero.")
        return np.asarray(quaternion, dtype=float) / norm

    @classmethod
    def _quaternion_multiply(cls, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        return cls._normalize_quaternion(np.array((
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )))

    @staticmethod
    def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
        return np.array((quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]))

    @staticmethod
    def _rotation_vector_to_quaternion(vector: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(vector)
        if angle < 1e-12:
            return np.array((1.0, 0.5 * vector[0], 0.5 * vector[1], 0.5 * vector[2]))
        axis = vector / angle
        return np.concatenate(([np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)))

    @classmethod
    def _quaternion_to_rotation_vector(cls, quaternion: np.ndarray) -> np.ndarray:
        quaternion = cls._normalize_quaternion(quaternion)
        if quaternion[0] < 0:
            quaternion *= -1
        vector_norm = np.linalg.norm(quaternion[1:])
        if vector_norm < 1e-12:
            return 2.0 * quaternion[1:]
        angle = 2.0 * np.arctan2(vector_norm, quaternion[0])
        return quaternion[1:] * angle / vector_norm
