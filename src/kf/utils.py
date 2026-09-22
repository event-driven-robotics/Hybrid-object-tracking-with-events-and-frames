import csv
import numpy as np


def save_velocity(
    velocity: np.ndarray,
    save_path: str,
) -> None:
    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    if velocity.ndim != 2 or velocity.shape[1] != 6:
        raise ValueError(
            f"velocity must have shape (N, 6), "
            f"got {velocity.shape}"
        )

    header = [
        "l_v_x",
        "l_v_y",
        "l_v_z",
        "a_v_x",
        "a_v_y",
        "a_v_z",
    ]

    with open(
        save_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(header)
        writer.writerows(velocity)