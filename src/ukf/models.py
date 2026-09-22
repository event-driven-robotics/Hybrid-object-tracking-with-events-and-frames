"""Data models shared by loading, experiments, and visualization."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np


class ExperimentMode(str, Enum):
    VELOCITY = "velocity"
    POSE = "pose"
    VELOCITY_POSE = "velocity-pose"


@dataclass(frozen=True)
class DatasetPaths:
    sequence_dir: Path
    velocity_csv: Path
    predicted_velocity_csv: Path | None = None
    dope_pose_file: Path | None = None
    se3_pose_csv: Path | None = None

    @classmethod
    def from_sequence_dir(cls, sequence_dir: Path) -> "DatasetPaths":
        sequence_dir = sequence_dir.expanduser().resolve()
        root = sequence_dir.parent
        return cls(
            sequence_dir=sequence_dir,
            velocity_csv=root / "velocity.csv",
            predicted_velocity_csv=root / "result.csv",
            dope_pose_file=root / "dope_poses_ycb.txt",
            se3_pose_csv=root / "mustard_translation_y_gt_velocity_1m_s_se3_pose.csv",
        )


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetPaths
    mode: ExperimentMode
    sampling_fps: int = 500
    pose_fps: int = 500
    velocity_fps: int = 60
    pose_update_fps: int = 10
    process_noise_cov: float = 0.001
    measurement_noise_cov: float = 0.001
    measurement_source: str = "dope"
    use_predicted_velocity: bool = True
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.velocity_fps <= 0 or self.pose_update_fps <= 0 or self.pose_fps <= 0:
            raise ValueError("All frequencies must be positive.")
        ratio = self.velocity_fps / self.pose_update_fps
        if not ratio.is_integer():
            raise ValueError("velocity_fps must be divisible by pose_update_fps.")
        if self.measurement_source not in {"dope", "se3"}:
            raise ValueError("measurement_source must be 'dope' or 'se3'.")

    @property
    def update_interval(self) -> int:
        return int(self.velocity_fps / self.pose_update_fps)


@dataclass
class ExperimentResult:
    estimated_poses: np.ndarray
    ground_truth_poses: np.ndarray
    velocities: np.ndarray
    updates_applied: np.ndarray
