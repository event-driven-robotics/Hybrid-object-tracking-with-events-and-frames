"""Adapter for the project's existing data-loading utilities.

Keeping legacy imports here prevents the UKF core from depending on a specific
dataset layout or on ``lib_func.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import rotation_matrix_to_quaternion
from .models import DatasetPaths


class LegacyDataset:
    """Read the original UE dataset through ``Data_loading`` and ``lib_func``.

    The two legacy modules are imported only when a dataset is actually read,
    so importing the package and running unit tests does not require rendering
    or dataset dependencies.
    """

    def __init__(self, paths: DatasetPaths, sampling_fps: int, pose_fps: int):
        self.paths = paths
        self.sampling_fps = sampling_fps
        self.pose_fps = pose_fps
        self._data_load, self._lib_func = self._load_legacy_modules()
        self.loader = self._data_load.DataLoad(str(paths.sequence_dir), sampling_fps, pose_fps)

    @staticmethod
    def _load_legacy_modules() -> tuple[Any, Any]:
        try:
            from . import Data_loader
            from src.kf import lib_func
        except ImportError as error:
            raise ImportError(
                "Dataset execution requires the original Data_loading.py and lib_func.py "
                "to be available on PYTHONPATH."
            ) from error
        return Data_loader, lib_func

    def velocities(self, use_predicted: bool) -> np.ndarray:
        path = self.paths.predicted_velocity_csv if use_predicted else self.paths.velocity_csv
        if path is None:
            raise ValueError("No predicted-velocity CSV was configured.")
        return self._numeric_rows(self._lib_func.read_velocity(str(path)))

    def ground_truth_pose(self, velocity_index: int, velocity_fps: int) -> np.ndarray:
        pose_index = round((velocity_index + 1) * self.sampling_fps / velocity_fps)
        pose_index = min(pose_index, len(self.loader.pose_list) - 1)
        transform = self.loader.load_gt_pose(self.loader.pose_list[pose_index])
        return np.r_[transform[:3, 3], rotation_matrix_to_quaternion(transform[:3, :3])]

    def initial_ground_truth_pose(self) -> np.ndarray:
        transform = self.loader.load_gt_pose(self.loader.pose_list[0])
        return np.r_[transform[:3, 3], rotation_matrix_to_quaternion(transform[:3, :3])]

    def measurements(self, source: str) -> np.ndarray:
        if source == "dope":
            if self.paths.dope_pose_file is None:
                raise ValueError("No DOPE pose file was configured.")
            raw_poses = self._lib_func.read_dope_poses(str(self.paths.dope_pose_file))
            return np.asarray([self._dope_to_pose(item) for item in raw_poses])
        if source == "se3":
            if self.paths.se3_pose_csv is None:
                raise ValueError("No SE(3) pose CSV was configured.")
            return self._numeric_rows(self._lib_func.read_velocity(str(self.paths.se3_pose_csv)))
        raise ValueError(f"Unsupported measurement source: {source}")

    def _dope_to_pose(self, dope_pose: np.ndarray) -> np.ndarray:
        dope_pose = np.asarray(dope_pose, dtype=float).reshape(-1)
        quaternion = self._lib_func.quaternion_from_rodrigues(dope_pose[3:].reshape(1, -1)).elements
        return np.r_[dope_pose[:3], quaternion]

    @staticmethod
    def _numeric_rows(values: Any) -> np.ndarray:
        """Convert legacy CSV output and remove a non-numeric header when present."""
        array = np.asarray(values)
        try:
            numeric = array.astype(float)
        except (TypeError, ValueError):
            numeric = array[1:].astype(float)
        if numeric.ndim != 2:
            raise ValueError("Expected a two-dimensional numeric data table.")
        return numeric
