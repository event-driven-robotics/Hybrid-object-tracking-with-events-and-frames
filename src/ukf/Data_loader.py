import time
import matplotlib.pyplot as plt
import numpy as np
from bimvee.importIitYarp import importIitYarpBinaryDataLog
import bimvee
import cv2
import os
from tqdm import tqdm
from PIL import Image
import json
from pyquaternion import Quaternion
from typing import Dict


class DataLoad:
    def __init__(self, path_dir, raw_fps, output_fps):
        self.path_dir = path_dir
        self.fps = raw_fps
        self.output_fps = output_fps
        self.timestamp = []
        self.pixel_x = []
        self.pixel_y = []
        self.polar = []
        self.width = 640
        self.height = 480
        # self.ev_img_list = sorted([x for x in os.listdir(path_dir) if x.__contains__('ec') and os.path.splitext(x)[-1] == '.png'])
        self.rgb_img_list = sorted([x for x in os.listdir(path_dir) if x.__contains__('.png') and len(x.split('.')) == 2])
        self.depth_list = sorted([x for x in os.listdir(path_dir) if x.__contains__('depth') and os.path.splitext(x)[-1] == '.png'])
        self.segmentation_list = sorted([x for x in os.listdir(path_dir) if x.__contains__('cs') and os.path.splitext(x)[-1] == '.png'])
        self.pose_list = self.read_gt_pose_file()
        # self.linear_velocity, self.angular_velocity = self.get_gt_velocity(path_dir, raw_fps, output_fps, self.pose_list)
        self.json_file_list = sorted([x for x in os.listdir(path_dir) if x.__contains__('.json') and x.__contains__('camera')==False and x.__contains__('object')==False])
        self.jacobian = self.inital_jacobian()
        self.filter_param = path_dir
        # prepare for event windows iterator
        self.events = []
        self.frame_timestamps = dict()
        self.frame_timestamps['ts'] = []
        self.event_batch_timestamp = dict()
        self.event_batch_timestamp['ts'] = []

        object_info_file = open(os.path.join(path_dir, self.json_file_list[0]))
        object_info = json.load(object_info_file)
        self.initial_timestamp = object_info['timestamp']
        previous_frame_timestamp = 0
        for f in range(int(len(self.rgb_img_list) * (output_fps / raw_fps))):
            object_info_file = open(os.path.join(path_dir, self.json_file_list[int(f * raw_fps / output_fps)]))
            object_info = json.load(object_info_file)
            next_timestamp = object_info['timestamp'] - self.initial_timestamp
            for stamp in range(10):
                self.event_batch_timestamp['ts'].append(previous_frame_timestamp+(next_timestamp-previous_frame_timestamp)/10*(stamp+1))
            self.frame_timestamps['ts'].append(next_timestamp)
            previous_frame_timestamp = next_timestamp

    def events_loading(self, events_path):
        # events = importIitYarpBinaryDataLog(filePathOrName=events_path)  # old version of bimvee e.g. version: 1.0.9
        self.events = bimvee.importIitYarp.importIitYarp(filePathOrName=events_path)
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
        self.timestamp = self.events['data']['left']['dvs']['ts'].reshape((-1, 1))
        self.pixel_x = self.events['data']['left']['dvs']['x'].reshape((-1, 1))
        self.pixel_y = self.events['data']['left']['dvs']['y'].reshape((-1, 1))
        self.polar = self.events['data']['left']['dvs']['pol'].reshape((-1, 1))
        self.initial_ev_timestamp = -self.events['data']['left']['dvs']['tsOffset']
        return self.timestamp, self.pixel_x, self.pixel_y, self.polar, self.initial_ev_timestamp

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

            only_positive_flag = positive_event_flag
            only_negative_flag = negative_event_flag

            event_img[only_positive_flag] = (2, 2, 228)  # blue, positive, motion background edge
            event_img[only_negative_flag] = (229, 8, 8)  # green, negative, motion forward edge
            cv2.imshow('events', event_img)
            cv2.imwrite('/home/lzc/rgb_events_2/' + str(i) + '.png', event_img)
            cv2.waitKey(1)

    def load_rgbd(self, rgb_frame_file_name, depth_frame_file_name):
        return cv2.imread(os.path.join(self.path_dir, rgb_frame_file_name)), np.asarray(Image.open(os.path.join(self.path_dir, depth_frame_file_name)))

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

    def load_filter_param(self, param_filter_param):
        with open(os.path.join(self.path_dir, param_filter_param), 'r', encoding='utf-8') as f:
            param_json = json.load(f)
            state_mean = param_json['initial_state']['mean']
            state_cov = param_json['initial_state']['cov']

            noise_mean = param_json['noise']['mean']
            noise_cov = param_json['noise']['cov']
            sigma = param_json['sigma_mu']
            event_trigger_parameter = param_json['event_trigger_parameter']
        return [state_mean, state_cov, noise_mean, noise_cov, sigma, event_trigger_parameter]


    def inital_jacobian(self):
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
        return self.jacobian


class EventIterator:
    def __init__(self, events: Dict, frame_timestamps: Dict):

        self._index = 0
        self._events = events
        self._frame_timestamps = frame_timestamps
        self._samples_count = len(frame_timestamps['ts'])  # time period
        self._events_count = len(events['ts'])

        self._batch_indices = [0] * (self._samples_count+1)

        frame_i = 0
        for ev_i in range(self._events_count):
            if self._events['ts'][ev_i] > frame_timestamps['ts'][frame_i]:
                self._batch_indices[frame_i] = ev_i
                frame_i += 1
                if(frame_i >= self._samples_count - 9):
                    break

    def __iter__(self):
        return self

    def __len__(self):
        return self._samples_count

    def __next__(self):

        if self._index >= self._samples_count-10:
            raise StopIteration

        i1 = self._batch_indices[self._index]
        i2 = self._batch_indices[self._index+1]

        retv = dict()
        retv['ts'] = self._events['ts'][i1:i2]
        retv['x'] = self._events['x'][i1:i2]
        retv['y'] = self._events['y'][i1:i2]
        retv['pol'] = self._events['pol'][i1:i2]

        self._index += 1
        return retv, len(retv['ts'])

