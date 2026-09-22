from dataclasses import dataclass
import numpy as np
from dataset import FrameData
import cv2
from typing import Optional
import copy
from models import TrackingEstimate


@dataclass(frozen=True)
class TrackerConfig:
    """
    Tracker params。
    """

    measurement_noise: float = 0.1  # TODO read from file
    gradient_kernel_size: int = 3
    use_segmentation: bool = True


class EventVelocityTracker:
    """
    estimate object 6d velocity using event measurements
    state: (vx, vy, vz, wx, wy, wz)
    """

    def __init__(self,
                 ekf,
                 measurement_model,
                 image_height: int,
                 image_width: int,
                 config: Optional[TrackerConfig] = None) -> None:
        self._ekf = ekf
        self._measurement_model = measurement_model
        self._image_height = image_height
        self._image_width = image_width
        self._config = config or TrackerConfig()
        self.event_timestamp = np.zeros((image_height, image_width))

    def process_frame(self,
                      frame: FrameData,
                      previous_rgb: np.ndarray,
                      current_rgb: np.ndarray,
                      ):

        vis_frame = copy.deepcopy(current_rgb)
        gradient_x, gradient_y = self._compute_gradient(current_rgb, previous_rgb)

        # try gaussian smoother
        cv2.GaussianBlur(gradient_x, (3, 3), cv2.BORDER_DEFAULT)
        cv2.GaussianBlur(gradient_y, (3, 3), cv2.BORDER_DEFAULT)

        predicted_mean, predicted_covariance = self._ekf.forward_kinematic()

        current_state = predicted_mean
        current_covariance = predicted_covariance

        model_state = [predicted_mean, predicted_covariance]

        self.event_timestamp = frame.event_timestamp

        for event in frame.events:
            x = event.x
            y = event.y
            t = event.timestamp

            depth = frame.depth[x, y]
            if frame.segmentation[x, y] == 1:
                if event.polarity == True:
                    vis_frame[x, y] = (179, 100, 27)
                else:
                    vis_frame[x, y] = (13, 150, 19)

                delta_t = t - self.event_timestamp[x, y]

                measurement = self._measurement_model.compute_residual(
                    event=event,
                    delta_t=delta_t,
                    gradient_x=gradient_x,
                    gradient_y=gradient_y,
                    depth_image=frame.depth,
                    reference_frame_timestamp=frame.reference_frame_timestamp,
                    motion_state=current_state,
                )

                if measurement is None:
                    continue

                implicit_measurement = (
                    self._ekf.implicit_measurement(
                        -measurement.residual,
                        measurement.jacobian,
                    )
                )

                estimated_state = self._ekf.update_model_state(
                    model_state,
                    implicit_measurement,
                    self._config.measurement_noise,
                )

                current_state = estimated_state.read_mean()
                current_covariance = estimated_state.read_cov()

                model_state = [
                    current_state,
                    current_covariance,
                ]

                self._save_event_time(event)

        estimate = TrackingEstimate(
            frame_index=frame.index,
            velocity=np.asarray(current_state).reshape(-1).copy(),
            covariance=np.asarray(current_covariance).copy(),
        )

        return vis_frame, estimate

    def _compute_gradient(self, intensity_change, rgb_img):
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        less_than_one = rgb_gray < 1
        rgb_gray[less_than_one] = 1
        rgb_gray = np.log(rgb_gray)

        rgb_img_gradient_x = cv2.Sobel(np.float32(rgb_gray), cv2.CV_64F, 1, 0, ksize=3)
        rgb_img_gradient_y = cv2.Sobel(np.float32(rgb_gray), cv2.CV_64F, 0, 1, ksize=3)

        return rgb_img_gradient_x, rgb_img_gradient_y

    def _save_event_time(self, event):
        self.event_timestamp[event.x, event.y] = event.timestamp
