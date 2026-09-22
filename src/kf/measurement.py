import numpy as np
import lib_func
from dataset import Event, EventMeasurement


class EventMeasurementModel:
    """
    compute measurement residual for KF
    """
    # TODO: measurement bias should read from config file
    def __init__(self, config) -> None:
        self._camera_focal_x = config.camera_focal_x
        self._camera_focal_y = config.camera_focal_y
        self._camera_cx = config.camera_cx
        self._camera_cy = config.camera_cy
        self._measurement_bias = config.event_trigger_parameter
        self._img_width = config.image_width
        self._img_height = config.image_height

    def compute_residual(self,
                         event: Event,
                         delta_t: float,
                         gradient_x: np.ndarray,
                         gradient_y: np.ndarray,
                         depth_image: np.ndarray,
                         reference_frame_timestamp: np.float32,
                         motion_state: np.ndarray) -> EventMeasurement:

        depth = depth_image[event.x, event.y]
        motion_vector = np.asarray(motion_state[:6], dtype=np.float64).reshape(-1, 1)
        pixel_velocity, interaction_matrix, _ = (
            lib_func.from_3d_to_2d_motion(
                depth,
                event.x,
                event.y,
                self._camera_focal_x,
                self._camera_focal_y,
                motion_vector,
                self._camera_cx,
                self._camera_cy
            )
        )
        compensated_x = int(event.x - 1 * (event.timestamp - reference_frame_timestamp) * pixel_velocity[1].item())
        compensated_y = int(event.y - 1 * (event.timestamp - reference_frame_timestamp) * pixel_velocity[0].item())

        if not (0 <= compensated_x < self._img_height and 0 <= compensated_y < self._img_width):
            return None
        else:
            image_gradient = np.array(
                [
                    gradient_x[compensated_x, compensated_y],
                    gradient_y[compensated_x, compensated_y],
                ],
                dtype=np.float64,
            )

            polarity_sign = (
                1.0 if event.polarity else -1.0
            )

            predicted_change = (-polarity_sign * np.dot(image_gradient, pixel_velocity) * delta_t)
            residual = (float(predicted_change) - self._measurement_bias)
            jacobian = (-polarity_sign * np.dot(image_gradient, interaction_matrix) * delta_t)

        return EventMeasurement(residual=residual, jacobian=np.asarray(jacobian, dtype=np.float64).reshape(-1))
