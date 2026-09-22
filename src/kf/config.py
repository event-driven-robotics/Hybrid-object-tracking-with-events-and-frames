from dataclasses import dataclass, field
from pathlib import Path
import os
import json


@dataclass(frozen=True)
class DatasetConfig:
    data_sequence_path: str
    image_width: int = 640
    image_height: int = 480
    sampling_fps: int = 500
    output_fps: int = 60

    # Initialized from _camera_settings.json
    camera_cx: float = field(init=False)
    camera_cy: float = field(init=False)
    camera_focal_x: float = field(init=False)
    camera_focal_y: float = field(init=False)

    def __post_init__(self):
        camera_path = os.path.join(
            self.data_sequence_path,
            "_camera_settings.json"
        )
        tracker_param_path = os.path.join(Path(self.data_sequence_path).parent, "filter_param.json")

        with open(camera_path, "r", encoding="utf-8") as f:
            camera_json = json.load(f)

        intrinsic = (
            camera_json["camera_settings"][0]["intrinsic_settings"]
        )

        # frozen=True prevents normal assignment, so use object.__setattr__
        object.__setattr__(self, "camera_focal_x", intrinsic["fx"])
        object.__setattr__(self, "camera_focal_y", intrinsic["fy"])
        object.__setattr__(self, "camera_cx", intrinsic["cx"])
        object.__setattr__(self, "camera_cy", intrinsic["cy"])

        with open(tracker_param_path, "r", encoding="utf-8") as f:
            tracker_param = json.load(f)

        object.__setattr__(self, "initial_state", tracker_param["initial_state"])
        object.__setattr__(self, "noise", tracker_param["noise"])
        object.__setattr__(self, "sigma_mu", tracker_param["sigma_mu"])
        object.__setattr__(self, "event_trigger_parameter", tracker_param["event_trigger_parameter"])

        object.__setattr__(self, "image_width", 640)
        object.__setattr__(self, "image_height", 480)
        object.__setattr__(self, "output_directory", self.data_sequence_path.split("photorealistic1")[0] + "result.csv")

