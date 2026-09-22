import numpy as np
import matplotlib.pyplot as plt


class TrackingEvaluator:

    def __init__(self):
        self.estimated_velocity = []
        self.ground_truth_velocity = []

    def add(self, estimate, ground_truth):
        self.estimated_velocity.append(estimate.velocity.copy())

        self.ground_truth_velocity.append(
            ground_truth.copy()
        )

    def plot(self, frame_period: float):
        """
        draw curve of the estimated velocity and ground truth
        """
        estimated = np.asarray(self.estimated_velocity[1:], dtype=np.float64)

        ground_truth = np.asarray(self.ground_truth_velocity[1:], dtype=np.float64)

        if len(estimated) == 0:
            print("No tracking results")
            return

        time = (np.arange(len(estimated)) * frame_period)
        labels = ["vx", "vy", "vx", "wx", "wy", "wz"]

        units = ["m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        axes = axes.flatten()

        for i in range(6):

            axes[i].plot(
                time,
                estimated[:, i],
                label="Estimated",
            )

            axes[i].plot(
                time,
                ground_truth[:, i],
                label="Ground Truth",
            )

            axes[i].set_title(labels[i])
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel(units[i])

            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()