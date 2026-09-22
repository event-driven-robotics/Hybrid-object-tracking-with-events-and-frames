"""Experiment orchestration for velocity-only and pose-corrected UKF runs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from src.ukf.dataset import LegacyDataset
from .geometry import pose_error
from src.ukf.models import ExperimentConfig, ExperimentMode, ExperimentResult
from .pose_ukf import PoseUKF, PoseUKFConfig


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run one reproducible filter experiment and write its numeric results.

    Velocity-only mode never corrects from poses.  Pose-only mode predicts with
    zero velocity and corrects periodically.  Velocity-pose uses both.
    """
    dataset = LegacyDataset(config.dataset, config.sampling_fps, config.pose_fps)
    velocities = dataset.velocities(config.use_predicted_velocity)
    measurements = dataset.measurements(config.measurement_source)
    initial_pose = dataset.initial_ground_truth_pose()
    filter_ = PoseUKF(
        initial_pose=initial_pose,
        initial_covariance=np.eye(6) * 1e-4,
        config=PoseUKFConfig(
            process_noise=np.eye(6) * config.process_noise_cov,
            measurement_noise=np.eye(6) * config.measurement_noise_cov,
            dt=1.0 / config.velocity_fps,
        ),
    )

    estimates: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    applied_updates: list[bool] = []
    zero_velocity = np.zeros(6)

    for index, velocity in enumerate(velocities):
        velocity = np.asarray(velocity, dtype=float).reshape(6)
        if config.mode is ExperimentMode.POSE:
            velocity = zero_velocity
        filter_.predict(velocity)

        should_update = (
            config.mode is not ExperimentMode.VELOCITY
            and index > 0
            and (index + 1) % config.update_interval == 0
        )
        if should_update:
            # DOPE is stored at the dataset sampling rate; the SE(3) file is
            # already aligned with the velocity stream.
            measurement_index = (
                round((index + 1) * config.sampling_fps / config.velocity_fps)
                if config.measurement_source == "dope"
                else index
            )
            measurement_index = min(measurement_index, len(measurements) - 1)
            filter_.update(measurements[measurement_index])

        estimates.append(filter_.mean.copy())
        ground_truth.append(dataset.ground_truth_pose(index, config.velocity_fps))
        applied_updates.append(should_update)

    result = ExperimentResult(
        estimated_poses=np.asarray(estimates),
        ground_truth_poses=np.asarray(ground_truth),
        velocities=np.asarray(velocities, dtype=float),
        updates_applied=np.asarray(applied_updates, dtype=bool),
    )
    save_result(result, config.output_dir, config.mode)
    return result


def save_result(result: ExperimentResult, output_dir: Path, mode: ExperimentMode) -> Path:
    """Save estimated poses, ground truth, and six-column pose errors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = np.asarray([pose_error(estimate, truth) for estimate, truth in zip(
        result.estimated_poses, result.ground_truth_poses
    )])
    output_path = output_dir / f"{mode.value}_errors.csv"
    np.savetxt(
        output_path,
        errors,
        delimiter=",",
        header="x_error,y_error,z_error,rx_error,ry_error,rz_error",
        comments="",
    )
    np.savetxt(
        output_dir / f"{mode.value}_estimated_poses.csv",
        result.estimated_poses,
        delimiter=",",
        header="x,y,z,qw,qx,qy,qz",
        comments="",
    )
    np.savetxt(
        output_dir / f"{mode.value}_ground_truth_poses.csv",
        result.ground_truth_poses,
        delimiter=",",
        header="x,y,z,qw,qx,qy,qz",
        comments="",
    )
    return output_path
