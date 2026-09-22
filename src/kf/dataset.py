import os
import cv2
import math
import json
import bimvee
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from bimvee.importIitYarp import importIitYarpBinaryDataLog
from pyquaternion import Quaternion
from dataclasses import dataclass
from src.kf import lib_func
from typing import List


@dataclass
class Event:
    timestamp: float
    x: int
    y: int
    polarity: bool


# TODO prepare event window and the referred frame at the same time
@dataclass(frozen=True)
class FrameData:
    index: int
    start_time: float
    end_time: float

    rgb: np.ndarray
    rgb_pre: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    reference_frame_timestamp: np.float32

    events: List[Event]
    event_timestamp: np.ndarray
    ground_truth_velocity: np.ndarray


@dataclass(frozen=True)
class EventMeasurement:
    """
    one event could derive one EKF measurement
    """
    residual: float       # q_xn
    jacobian: np.ndarray  # h_n


class DataLoader:
    def __init__(self, path_dir, sampling_fps, output_fps):
        self.path_dir = path_dir
        self.fps = sampling_fps
        self.output_fps = output_fps
        self.ev_data = {}
        self.event = []
        self.timestamp = []
        self.pixel_x = []
        self.pixel_y = []
        self.polar = []
        self.time_windows = []
        self.width = 640
        self.height = 480
        self.ms_to_idx = {}
        self.frame_timestamp_initial = []
        self.ev_img_list = sorted(
            [x for x in os.listdir(path_dir) if x.__contains__('ec') and os.path.splitext(x)[-1] == '.png'])
        self.rgb_img_list = sorted(
            [x for x in os.listdir(path_dir) if x.__contains__('.png') and len(x.split('.')) == 2])
        self.depth_list = sorted(
            [x for x in os.listdir(path_dir) if x.__contains__('depth') and os.path.splitext(x)[-1] == '.png'])
        self.segmentation_list = sorted(
            [x for x in os.listdir(path_dir) if x.__contains__('cs') and os.path.splitext(x)[-1] == '.png'])
        self.pose_list = self.read_gt_pose_file()
        # self.linear_velocity, self.angular_velocity = self.get_gt_velocity(path_dir, raw_fps, output_fps, self.pose_list)
        self.json_file_list = sorted([x for x in os.listdir(path_dir) if
                                      x.__contains__('.json') and x.__contains__('camera') == False and x.__contains__(
                                          'object') == False])
        self.jacobian = self.initial_jacobian()
        self.gt_velocities = lib_func.read_velocity(os.path.join(path_dir.split('photorealistic1')[0], 'velocity.csv'))
        self.load_frame_timestamp_initial()

    def events_loading(self, events_path):
        # events = importIitYarpBinaryDataLog(filePathOrName=events_path)
        events = bimvee.importIitYarp.importIitYarp(filePathOrName=events_path.split('photorealistic1')[0])
        '''
        {'info': {'filePathOrName': '/home/lzc/YCB_dataset_sample/YCB/7/photorealistic1/binaryevents.log',
          'importedToByte': 79998,
          'zeroTimestamps': True,
          'fileFormat': 'iityarp'},
         'data': {'left': {'dvs': {'ts': array([0.00000e+00, 0.00000e+00, 1.60000e-07, ..., 1.70232e-03,
                   1.70232e-03, 1.70248e-03]),
            'x': array([383, 294, 558, ..., 478, 347,  13], dtype=uint16),
            'y': array([ 34, 153,  85, ..., 271, 443, 259], dtype=uint16),
            'pol': array([ True,  True,  True, ..., False, False, False]),
            'tsOffset': -0.0}}}}
        '''
        if isinstance(events, list):
            events = events[0]
        self.timestamp = events['data']['left']['dvs']['ts'].reshape((-1, 1))
        self.pixel_x = events['data']['left']['dvs']['x'].reshape((-1, 1))
        self.pixel_y = events['data']['left']['dvs']['y'].reshape((-1, 1))
        self.polar = events['data']['left']['dvs']['pol'].reshape((-1, 1))
        self.initial_ev_timestamp = -events['data']['left']['dvs']['tsOffset']
        self.event = np.concatenate((self.timestamp, self.pixel_x, self.pixel_y, self.polar), axis=1)

        self.ev_data['x'] = self.pixel_x
        self.ev_data['y'] = self.pixel_y
        self.ev_data['p'] = self.polar
        self.ev_data['t'] = self.timestamp
        self.ev_data['initial_ev_timestamp'] = self.initial_ev_timestamp
        return self.ev_data
        # return self.timestamp, self.pixel_x, self.pixel_y, self.polar, self.initial_ev_timestamp

    def build_ms_to_idx(self):
        t_us = (self.timestamp * 10 ** 6).astype(np.int64)
        max_ms = math.ceil(self.events[-1, 0] * 1000)
        max_ms = np.arange(max_ms + 1, dtype=np.int64) * 1000  # second to ms
        self.ms_to_idx = np.searchsorted(t_us, max_ms, side='left').astype(np.int64)

    def get_timestamp_ms(self):
        self.time_windows = np.arange(self.timestamp[0], self.timestamp[-1], 1 / self.output_fps)
        return self.time_windows * 1000  # convert second to ms

    def events_display(self, img_width, img_height):
        timestamp = self.timestamp
        pixel_x = self.pixel_x
        pixel_y = self.pixel_y
        polar = self.polar
        time_gap_ec_png = 1 / self.output_fps
        # time_gap_ec_png = 1 / self.fps
        num_time_gap = (timestamp.max() - timestamp.min()) / time_gap_ec_png
        for i in tqdm(range(int(num_time_gap))):
            event_img = np.ones((img_height, img_width, 3), np.uint8) * 255
            positive_event_flag = np.zeros((img_height, img_width), dtype=bool)
            negative_event_flag = np.zeros((img_height, img_width), dtype=bool)
            event_index = (timestamp > i * time_gap_ec_png) & (timestamp < (i + 1) * time_gap_ec_png)
            for x, y, p in zip(pixel_x[event_index], pixel_y[event_index], polar[event_index]):
                if p == True:
                    positive_event_flag[y, x] = True
                else:
                    negative_event_flag[y, x] = True
            mix_event_flag = positive_event_flag & negative_event_flag
            only_positive_flag = positive_event_flag ^ mix_event_flag
            only_negative_flag = negative_event_flag ^ mix_event_flag

            event_img[mix_event_flag] = (27, 174, 179)  # yellow positive&negative
            event_img[only_positive_flag] = (179, 100, 27)  # blue, positive, motion background edge
            event_img[only_negative_flag] = (13, 150, 19)  # green, negative, motion forward edge
            cv2.imshow('events', event_img)
            cv2.waitKey(0)

    def load_rgbd(self, rgb_frame_file_name, depth_frame_file_name):
        return cv2.imread(os.path.join(self.path_dir, rgb_frame_file_name)), Image.open(
            os.path.join(self.path_dir, depth_frame_file_name))

    def load_event_img(self, event_frame_file_name):
        return plt.imread(os.path.join(self.path_dir, event_frame_file_name))
        # return np.load(os.path.join(self.path_dir, event_frame_file_name))

    def load_segmentation(self, segmentation_file_name):
        return cv2.imread(os.path.join(self.path_dir, segmentation_file_name))[:, :, 0].astype(bool)

    def read_gt_pose_file(self):
        file_list = os.listdir(self.path_dir)
        file_list_len = len(file_list)
        info_list = []
        # Pick up all files
        for i in range(file_list_len):
            if (len(file_list[i].split('.')) == 2) and (file_list[i].split('.')[1] == 'json') and (
                    file_list[i].split('.')[0][0] != '_'):
                info_list.append(file_list[i])
        # sort for info_list
        info_list.sort()
        return info_list

    def load_gt_pose(self, pose_file_name):
        with open(os.path.join(self.path_dir, pose_file_name), 'r', encoding='utf-8') as f:
            info_json = json.load(f)
            # from pose_transform
            # raw_pose_transform = np.array(info_json['objects'][0]['pose_transform'])

            # from quaternion
            raw_quanternion_xyzw = np.array(info_json['objects'][0]['quaternion_xyzw'])[[3, 0, 1, 2]]

            object_pose_cur = Quaternion(raw_quanternion_xyzw).transformation_matrix
            raw_position = np.array(info_json['objects'][0]['location'])
            object_pose_cur[0:3, 3] = raw_position

            # object_pose = raw_pose_transform
            object_pose_cur[0:3, 3] /= 100  # cm to m
        return object_pose_cur

    def load_frame_timestamp_initial(self):
        self.frame_timestamp_initial = lib_func.read_timestamp(self.json_file_list[0], self.path_dir)

    def load_frame(self, frame_index: int) -> FrameData:
        # load rgb_d frames
        rgb_img_cur, depth_img_cur = self.load_rgbd(
            self.rgb_img_list[int((frame_index + 1) * self.fps / self.output_fps)],
            self.depth_list[int((frame_index + 1) * self.fps / self.output_fps)])
        depth_img_cur = np.array(depth_img_cur.getdata()).reshape(self.height, self.width)

        rgb_img_pre, depth_img_pre = self.load_rgbd(
            self.rgb_img_list[int(frame_index * self.fps / self.output_fps)],
            self.depth_list[int(frame_index * self.fps / self.output_fps)])
        depth_img_pre = np.array(depth_img_pre.getdata()).reshape(self.height, self.width)

        seg_img_pre = self.load_segmentation(
            self.segmentation_list[int((frame_index + 0) * self.fps / self.output_fps)])
        seg_img_pre = seg_img_pre.astype(np.uint8)
        # seg_img_cur = lib_func.dilation(seg_img_cur)

        #
        bbox_x_cur, bbox_y_cur, _, _, frame_timestamp_cur = lib_func.read_object_bbox_fc(
            self.json_file_list[int((frame_index + 1) * self.fps / self.output_fps)], self.path_dir)

        bbox_x_pre, bbox_y_pre, _, _, frame_timestamp_pre = lib_func.read_object_bbox_fc(
            self.json_file_list[int(frame_index * self.fps / self.output_fps)], self.path_dir)

        frame_timestamp_pre -= self.frame_timestamp_initial
        frame_timestamp_cur -= self.frame_timestamp_initial

        # load events
        frame_start_time = frame_index * (1 / self.output_fps)
        frame_end_time = (frame_index + 1) * (1 / self.output_fps)
        event_index = (self.timestamp >= frame_start_time) & (self.timestamp < frame_end_time)
        events = []
        for y, x, p, t in zip(
                self.pixel_x[event_index],
                self.pixel_y[event_index],
                self.polar[event_index],
                self.timestamp[event_index],
        ):
            event = Event(
                timestamp=float(t),
                x=int(x),
                y=int(y),
                polarity=bool(p),
            )
            events.append(event)

        # # create event timestamp each frame
        event_timestamp = np.ones((self.height, self.width)) * frame_index / self.output_fps

        return FrameData(
            index=frame_index,
            start_time=frame_start_time,
            end_time=frame_end_time,
            rgb=rgb_img_cur,
            rgb_pre=rgb_img_pre,
            depth=depth_img_pre,
            segmentation=seg_img_pre,
            reference_frame_timestamp=frame_timestamp_pre,
            events=events,
            event_timestamp=event_timestamp,
            ground_truth_velocity=self.gt_velocities[int(frame_index * self.fps / self.output_fps)],
        )

    def initial_jacobian(self):
        jacobian = {}
        for u in range(self.width):
            jacobian[u] = {}
        return jacobian

    def eval_jacobian(self, d, u, v, lambda_, camera_cx, camera_cy):
        return np.array(
            ([lambda_ / d, 0, -(u - camera_cx) / d, -(u - camera_cx) * (v - camera_cy) / lambda_,
              (lambda_ ** 2 + (u - camera_cx) ** 2) / lambda_, -(v - camera_cy) * lambda_ / lambda_],
             [0, lambda_ / d, -(v - camera_cy) / d, -(lambda_ ** 2 + (v - camera_cy) ** 2) / lambda_,
              (u - camera_cx) * (
                      v - camera_cy) / lambda_, (u - camera_cx) * lambda_ / lambda_]))

    def jacobi_matrix(self, path_dir):
        with open(os.path.join(path_dir, '_camera_settings.json'), 'r', encoding='utf-8') as f:
            camera_json = json.load(f)
            camera_focal_x = camera_json['camera_settings'][0]['intrinsic_settings']['fx']
            camera_cx = camera_json['camera_settings'][0]['intrinsic_settings']['cx']
            camera_cy = camera_json['camera_settings'][0]['intrinsic_settings']['cy']

        for u in range(self.width):
            for v in range(self.height):
                self.jacobian[u][v] = self.eval_jacobian(1, u, v, camera_focal_x, camera_cx, camera_cy)

    def __iter__(self):
        # frame_num = len(self.rgb_img_list)
        frame_num = int(self.output_fps / self.fps * len(self.rgb_img_list))
        for frame_index in range(frame_num-1):
            yield self.load_frame(frame_index)

