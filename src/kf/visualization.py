import cv2
import numpy as np
from dataset import FrameData
import matplotlib.pyplot as plt
from models import TrackingEstimate


class Visualizer:
    def show_frame(self, frame: np.array, estimate: TrackingEstimate) -> None:
        vis_frame = frame.copy()
        cv2.imshow("test_rgb", vis_frame)

        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyAllWindows()


