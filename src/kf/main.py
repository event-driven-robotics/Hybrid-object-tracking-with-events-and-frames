import argparse
import numpy as np
from tqdm import tqdm
# from .config import TrackingConfig
from dataset import DataLoader
from config import DatasetConfig

# from .evaluation import TrackingEvaluator
# from .measurement import EventMeasurementModel
from src.kf.velocity_tracker import EventVelocityTracker
from ekf import create_ekf
from src.kf.measurement import EventMeasurementModel
from src.kf.visualization import Visualizer
from typing import Optional
from evaluation import TrackingEvaluator
from src.kf.utils import save_velocity


# from .visualization import (
#     TrackingVisualizer,
#     plot_velocity_result,
# )

def run_tracker(
        dataset: DataLoader,  # data
        tracker: EventVelocityTracker,
        evaluator: TrackingEvaluator,
        visualizer: Optional[Visualizer] = None):

    previous_rgb = None
    velocity_results = []

    for frame in tqdm(dataset, desc="Processing frames"):  # iterate frames
        pre_rgb = frame.rgb_pre
        cur_rgb = frame.rgb_pre

        vis_frame, estimate = tracker.process_frame(
            frame=frame,
            previous_rgb=pre_rgb,
            current_rgb=cur_rgb,
            # diagnostics=evaluator.diagnostics,
        )

        evaluator.add(
            estimate=estimate,
            ground_truth=frame.ground_truth_velocity,
        )

        velocity_results.append(estimate.velocity)

        if visualizer is not None:
            visualizer.show_frame(vis_frame, vis_frame)

        previous_rgb = frame.rgb

    return np.asarray(
            velocity_results,
            dtype=np.float64,
        )


def main():
    parser = argparse.ArgumentParser(description="Run event velocity tracker")
    parser.add_argument(
        "--data-sequence-path",
        required=True,
        help="data_path/photorealistic1",
    )
    args = parser.parse_args()

    config = DatasetConfig(
        data_sequence_path=args.data_sequence_path
    )

    dataset = DataLoader(path_dir=config.data_sequence_path, sampling_fps=config.sampling_fps, output_fps=config.output_fps)
    ev_data = dataset.events_loading(events_path=config.data_sequence_path)

    # create kalman filter tracker
    ekf = create_ekf(config)

    # initialize measurement model
    measurement_model = EventMeasurementModel(config)

    tracker = EventVelocityTracker(
        ekf=ekf,
        image_height=config.image_height,
        image_width=config.image_width,
        measurement_model=measurement_model,
    )

    visualizer = Visualizer()
    evaluator = TrackingEvaluator()

    result = run_tracker(
        dataset=dataset,
        tracker=tracker,
        evaluator=evaluator,
        visualizer=visualizer)

    evaluator.plot(
        frame_period=1.0 / config.output_fps
    )
    save_velocity(result, config.output_directory)


if __name__ == "__main__":
    main()
