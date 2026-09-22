""" Runner for UKF experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.ukf.experiment import run_experiment
from src.ukf.models import DatasetPaths, ExperimentConfig, ExperimentMode
from src.ukf.visualization import rmse, save_mode_comparison, save_summary_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quaternion-pose UKF experiments."
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Example: /data/sequence/photorealistic1",
    )
    parser.add_argument(
        "--mode",
        choices=[*(item.value for item in ExperimentMode), "all"],
        default="all",
        help="Run one mode, or all three modes (default: all).",
    )
    parser.add_argument("--measurement-source", choices=("dope", "se3"), default="dope")
    parser.add_argument("--use-ground-truth-velocity", action="store_true")
    parser.add_argument("--sampling-fps", type=int, default=500)
    parser.add_argument("--pose-fps", type=int, default=500)
    parser.add_argument("--velocity-fps", type=int, default=60)
    parser.add_argument("--pose-update-fps", type=int, default=10)
    parser.add_argument("--process-noise", type=float, default=0.001)
    parser.add_argument("--measurement-noise", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots as well as saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_config = ExperimentConfig(
        dataset=DatasetPaths.from_sequence_dir(args.sequence_dir),
        mode=ExperimentMode.VELOCITY,
        sampling_fps=args.sampling_fps,
        pose_fps=args.pose_fps,
        velocity_fps=args.velocity_fps,
        pose_update_fps=args.pose_update_fps,
        process_noise_cov=args.process_noise,
        measurement_noise_cov=args.measurement_noise,
        measurement_source=args.measurement_source,
        use_predicted_velocity=not args.use_ground_truth_velocity,
        output_dir=args.output_dir,
    )

    modes = (
        tuple(ExperimentMode)
        if args.mode == "all"
        else (ExperimentMode(args.mode),)
    )

    results = {}

    for mode in modes:
        config = replace(base_config, mode=mode)

        print(f"\nRunning UKF experiment: {mode.value}")
        result = run_experiment(config)
        results[mode] = result

        metrics = rmse(result)
        save_summary_plot(
            result,
            args.output_dir / f"{mode.value}_summary.png",
            config.velocity_fps,
            show=args.show,
        )

        print(f"  Translation RMSE: {metrics['translation_m']:.6f} m")
        print(f"  Rotation RMSE:    {metrics['rotation_rad']:.6f} rad")

    if len(results) > 1:
        save_mode_comparison(
            results,
            args.output_dir / "mode_comparison.png",
            base_config.velocity_fps,
            show=args.show,
        )


if __name__ == "__main__":
    main()