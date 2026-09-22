"""Small, dependency-free helpers for 6-DoF pose representations."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Quaternion must not be zero.")
    return quaternion / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quaternion(left)
    w2, x2, y2, z2 = normalize_quaternion(right)
    return normalize_quaternion(np.array((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )))


def rotation_vector_to_quaternion(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return normalize_quaternion(np.r_[1.0, vector / 2.0])
    return np.r_[np.cos(angle / 2.0), vector / angle * np.sin(angle / 2.0)]


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    vector_norm = np.linalg.norm(quaternion[1:])
    if vector_norm < 1e-12:
        return 2 * quaternion[1:]
    return quaternion[1:] * (2 * np.arctan2(vector_norm, quaternion[0]) / vector_norm)


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a scalar-first quaternion."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        quaternion = np.array((0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                               (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index, final_index = (index + 1) % 3, (index + 2) % 3
        scale = 2 * np.sqrt(1 + matrix[index, index] - matrix[next_index, next_index] - matrix[final_index, final_index])
        quaternion = np.empty(4)
        quaternion[0] = (matrix[final_index, next_index] - matrix[next_index, final_index]) / scale
        quaternion[index + 1] = 0.25 * scale
        quaternion[next_index + 1] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        quaternion[final_index + 1] = (matrix[final_index, index] + matrix[index, final_index]) / scale
    return normalize_quaternion(quaternion)


def pose_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return translation and rotation-vector error for two 7-D poses."""
    estimate, reference = np.asarray(estimate, dtype=float).reshape(7), np.asarray(reference, dtype=float).reshape(7)
    inverse_reference = reference[3:] * np.array((1, -1, -1, -1))
    return np.r_[estimate[:3] - reference[:3], quaternion_to_rotation_vector(quaternion_multiply(estimate[3:], inverse_reference))]
