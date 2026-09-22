

# Hybrid Object Tracking with Events and Frames


Dataset: https://zenodo.org/records/22897787


```
@INPROCEEDINGS{10342300,
  author={Li, Zhichao and Piga, Nicola A. and Di Pietro, Franco and Iacono, Massimiliano and Glover, Arren and Natale, Lorenzo and Bartolozzi, Chiara},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Hybrid Object Tracking with Events and Frames}, 
  year={2023},
  volume={},
  number={},
  pages={9057-9064},
  keywords={Optical filters;Training;Codes;Heuristic algorithms;Pose estimation;Vision sensors;6-DOF},
  doi={10.1109/IROS55552.2023.10342300}}
```

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run Velocity Tracker

Run the following command from the project root:

```bash
PYTHONPATH=. python3 src/kf/main.py \
  --data-sequence-path /path/to/your/dataset
```

Example:

```bash
PYTHONPATH=. python3 src/kf/main.py \
  --data-sequence-path /home/user/UE_dataset/translation_y_gt_velocity_1m_s/photorealistic1
```

### Run Pose Tracker

```bash
PYTHONPATH=. python3 scripts/run_pose_tracker.py \
  /path/to/your/dataset \
  --mode all \
  --show
```

Frame-based pose estimates are obtained using [DOPE](https://github.com/NVlabs/Deep_Object_Pose)

### Output

The estimated velocity is saved in the folder of the data sequence
and evaluation results are saved to the output directory



### Acknowledgements

This project uses [DOPE](https://github.com/NVlabs/Deep_Object_Pose) for frame-based pose estimation.