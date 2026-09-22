from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class TrackingEstimate:
    frame_index: int
    velocity: FloatArray
    covariance: FloatArray


@dataclass
class TrackingDiagnostics:
    innovations: List[float] = field(default_factory=list)
    event_delta_times: List[float] = field(default_factory=list)


@dataclass
class TrackingEstimate:
    frame_index: int
    velocity: np.ndarray
    covariance: np.ndarray

@dataclass(frozen=True)
class TrackingResult:
    estimated_velocity: FloatArray
    ground_truth_velocity: FloatArray
    diagnostics: TrackingDiagnostics