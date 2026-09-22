"""Plots and metrics, deliberately separate from filtering and data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import pose_error
# from .models import ExperimentResult
from src.ukf.models import ExperimentMode, ExperimentResult


def error_series(result: ExperimentResult) -> np.ndarray:
    return np.asarray([pose_error(estimate, truth) for estimate, truth in zip(
        result.estimated_poses, result.ground_truth_poses
    )])


def rmse(result: ExperimentResult) -> dict[str, float]:
    errors = error_series(result)
    return {
        "translation_m": float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        "rotation_rad": float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
    }


def save_summary_plot(result, output_path, velocity_fps, show=False):
    """Save a compact pose/error plot and optionally display it."""
    import matplotlib.pyplot as plt

    time = np.arange(len(result.estimated_poses)) / velocity_fps
    errors = error_series(result)

    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    labels = ("x", "y", "z")

    for axis, component, label in zip(axes[0], range(3), labels):
        axis.plot(
            time,
            result.ground_truth_poses[:, component],
            label="ground truth",
        )
        axis.plot(
            time,
            result.estimated_poses[:, component],
            label="UKF",
        )
        axis.set_title(f"position {label}")
        axis.set_ylabel("m")
        axis.grid()

    for axis, component, label in zip(axes[1], range(3), labels):
        axis.plot(time, np.degrees(errors[:, component + 3]))
        axis.set_title(f"rotation error {label}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("deg")
        axis.grid()

    axes[0, 0].legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)

    if show:
        plt.show()

    plt.close(figure)


def save_mode_comparison(
    results: dict[ExperimentMode, ExperimentResult],
    output_path: Path,
    velocity_fps: int,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        ExperimentMode.VELOCITY: "tab:green",
        ExperimentMode.POSE: "tab:orange",
        ExperimentMode.VELOCITY_POSE: "tab:blue",
    }

    reference = next(iter(results.values()))
    time = np.arange(len(reference.ground_truth_poses)) / velocity_fps

    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axis_labels = ("x", "y", "z")

    for axis, component, label in zip(axes[0], range(3), axis_labels):
        axis.plot(
            time,
            reference.ground_truth_poses[:, component],
            color="black",
            label="ground truth",
        )

        for mode, result in results.items():
            axis.plot(
                time,
                result.estimated_poses[:, component],
                color=colors[mode],
                label=mode.value,
            )

        axis.set_title(f"Position {label}")
        axis.set_ylabel("m")
        axis.grid()

    for axis, component, label in zip(axes[1], range(3), axis_labels):
        for mode, result in results.items():
            errors = error_series(result)
            axis.plot(
                time,
                np.degrees(errors[:, component + 3]),
                color=colors[mode],
                label=mode.value,
            )

        axis.set_title(f"Rotation error {label}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("deg")
        axis.grid()

    axes[0, 0].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)

    if show:
        plt.show()

    plt.close(figure)
