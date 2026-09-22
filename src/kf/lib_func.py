import os
import csv
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import copy
import time
import pyquaternion as Quaternion


def from_3d_to_2d_motion(depth, v, u, lambda_x, lambda_y, motion_vector_3d, camera_cx,
                         camera_cy):
    # define interaction matrix for each pixel
    z = depth / 1000
    B = np.array(
        ([lambda_x / z, 0, -(u - camera_cx) / z, -(u - camera_cx) * (v - camera_cy) / lambda_y,
          (lambda_x ** 2 + (u - camera_cx) ** 2) / lambda_x, -(v - camera_cy) * lambda_x / lambda_y],
         [0, lambda_y / z, -(v - camera_cy) / z, -(lambda_y ** 2 + (v - camera_cy) ** 2) / lambda_y, (u - camera_cx) * (
                 v - camera_cy) / lambda_x, (u - camera_cx) * lambda_y / lambda_x]))

    motion_vector_3d = motion_vector_3d.reshape(6, 1)
    return np.dot(B, motion_vector_3d), B, np.dot(B[:, 3:6], motion_vector_3d[3:6, 0].reshape(3, 1))


def epe(event_flow, dense_flow_vector, mask):
    epe_value = 0
    vIndeices = mask.nonzero()
    for x, y in zip(vIndeices[0], vIndeices[1]):
        epe_value += np.linalg.norm(event_flow[x, y] - dense_flow_vector[x, y])
    avg_epe_value = epe_value / mask.sum()
    if np.isnan(avg_epe_value):
        return 0
    else:
        return avg_epe_value


def draw_epe(epe):
    x = np.arange(0, len(epe), 1)
    plt.plot(x, epe, linewidth=2)
    plt.title("EPE of optical flow", fontsize=14)
    plt.xlabel("'frame'", fontsize=12)
    plt.ylabel("error", fontsize=12)
    plt.show()


def draw_vector(rgb, mask):
    for i in range(rgb.shape[0] - 10):
        for j in range(rgb.shape[1] - 10):
            if i % 8 == 0 and j % 8 == 0 and mask[i, j]:
                # print(i, j)
                rgb = cv2.arrowedLine(rgb, (j, i), (j + 4, i), (1, 3, 175), 1, 8, 0, 0.2)
                cv2.imshow("vectors", rgb)


def read_velocity(file_path):
    gt_velocities = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            gt_velocities.append(line)
    return gt_velocities


def dilation(img):
    kernel = np.ones((40, 40), np.uint8)
    dict = cv2.dilate(img, kernel, iterations=1)
    return dict


def save_rgbd_sample(save_path, rgb, depth, mask, cx, cy, fx):
    fy = fx
    point_num = 480 * 640  # mask.sum()
    type = np.array(['NCOFF'])
    with open(save_path, 'ab') as f:
        np.savetxt(f, type, delimiter=" ", fmt='%s')
        np.savetxt(f, np.array([(point_num, 0, 0)]), delimiter=" ", fmt="%d")
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                # if mask[i, j] == True:
                x = (j - cx) * depth[i, j] / 1000 / fx
                y = (i - cy) * depth[i, j] / 1000 / fy
                data = np.array([(x, y, depth[i, j] / 1000, 0, 0, -1, rgb[i, j][2], rgb[i, j][1], rgb[i, j][0], 255)])
                fmt = '%0.3f', '%0.3f', '%0.3f', '%d', '%d', '%d', '%d', '%d', '%d', '%d'
                np.savetxt(f, data, fmt=fmt)


def pixel_velocity(mask, depth_img, motion_3d_gt, camera_focal, camera_cx, camera_cy, delta_t, load_data):
    u_dot_matrix = np.zeros((480, 640, 2))
    # print(motion_3d_gt)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]==True:
                u_dot, interaction_matrix_b, rot = from_3d_to_2d_motion(depth_img[i, j],
                                                                        i,
                                                                        j,
                                                                        camera_focal,
                                                                        motion_3d_gt,
                                                                        camera_cx,
                                                                        camera_cy,
                                                                        load_data)
                u_dot_matrix[i, j, :] += u_dot.squeeze() * delta_t
    return u_dot_matrix


def timestamp_img(event_timestamp_):
    event_timestamp_ = event_timestamp_ / event_timestamp_.max() * 255
    event_timestamp_ = cv2.merge([event_timestamp_, event_timestamp_, event_timestamp_])
    return event_timestamp_


def draw_event_num_distribution(event_num_p, event_num_n, q_m_each_frame, eg_delta_t, measurement_hot_map):
    values = []
    measurement_values = []
    event_num_p_ = event_num_p.reshape((1, 480 * 640))
    event_p_list = event_num_p_.tolist()

    event_num_n_ = event_num_n.reshape((1, 480 * 640))
    event_n_list = event_num_n_.tolist()

    event_list = event_p_list[0] + event_n_list[0]
    event_list_ = set(event_list)
    for item in sorted(event_list_):
        if item != 0.0:
            # print("the %d has found %d" % (item, event_list.count(item)))
            values.append([item, event_list.count(item)])

    for k, g in groupby(sorted(q_m_each_frame), key=lambda x: x // 1):
        # print('{}-{}: {}'.format(k * 10, (k + 1) * 10 + 1, len(list(g))))
        measurement_values.append([k, len(list(g))])
    values = np.array(values)
    measurement_values = np.array(measurement_values)

    event_axis = np.arange(0, len(eg_delta_t), 1)
    eg_delta_t = np.array(eg_delta_t)

    plt.clf()
    # plt.figure()
    try:
        plt.subplot(1, 3, 1)
        plt.plot(values[:, 0], values[:, 1], label='event_distribtion')
        plt.xlabel('(+-)event_num/pixel', fontsize=12)
        plt.ylabel('pixel_num', fontsize=12)
        plt.subplot(1, 3, 2)
        plt.plot(measurement_values[:, 0], measurement_values[:, 1], label='event_measurement_distribtion')

        plt.subplot(1, 3, 3)
        plt.plot(event_axis, eg_delta_t, label='event_delta_t')
        plt.xlabel('event_id', fontsize=12)
        plt.ylabel('Δt', fontsize=12)
        plt.title('a time window for events')

        plt.show(block=False)
        plt.pause(0.05)
        plt.ioff
    except IndexError:
        print('skip this frame')
    else:
        pass
    cv2.imshow("measurement_heat_map", measurement_hot_map)
    cv2.waitKey(1)


def visualize_gradient(y, x, rgb_img_gradient_x, rgb_img_gradient_y, rgb_img_pre):
    rgb_img_pre = cv2.arrowedLine(rgb_img_pre, (x, y), (int(x + rgb_img_gradient_x), int(y + rgb_img_gradient_y)),
                                  (1, 3, 175), 1)
    return rgb_img_pre


def gradient(intensity_change, rgb_img):
    rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    less_than_one = rgb_gray < 1
    rgb_gray[less_than_one] = 1
    rgb_gray = np.log(rgb_gray)

    rgb_img_gradient_x = cv2.Sobel(np.float32(rgb_gray), cv2.CV_64F, 1, 0, ksize=3)
    rgb_img_gradient_y = cv2.Sobel(np.float32(rgb_gray), cv2.CV_64F, 0, 1, ksize=3)

    return rgb_img_gradient_x, rgb_img_gradient_y


# visualize gradient direction
def visual_gradient_direction(rgb_img_pre, json_file_name, path_dir, event_map, seg_img):
    # import pandas as pd
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    # obtain img intensity
    rgb_gray = cv2.cvtColor(rgb_img_pre, cv2.COLOR_BGR2GRAY)
    less_than_one = rgb_gray < 1
    rgb_gray[less_than_one] = 1
    rgb_gray = np.log(rgb_gray)

    norm_rgb_gray = (rgb_gray - rgb_gray.min()) / (rgb_gray.max() - rgb_gray.min())

    # obtain bbox
    x_position_start, y_position_start, width, height, _ = read_object_bbox(json_file_name, path_dir)
    x_position_start += 10
    y_position_start += 10
    bbox_size = min(width, height)

    # obtain img gradient
    rgb_img_gradient_x, rgb_img_gradient_y = gradient(rgb_img_pre, rgb_img_pre)

    # normalize gradient
    # gradient_max = max(rgb_img_gradient_x.max(), rgb_img_gradient_y.max())
    # gradient_min = min(rgb_img_gradient_x.min(), rgb_img_gradient_y.max())
    # rgb_img_gradient_x = rgb_img_gradient_x / (gradient_max - gradient_min)
    # rgb_img_gradient_y = rgb_img_gradient_y / (gradient_max - gradient_min)

    plt.figure()
    ax = plt.subplot(111)

    for high in range(1, bbox_size):
        for i in range(bbox_size - 30):
            x1 = np.linspace(i, i + 1, bbox_size)
            ax.fill_between(x1, bbox_size - (high - 1), bbox_size - high, cmap='cool', facecolor=(
            0 / 256, 0 / 256, 0 / 256,
            1 - norm_rgb_gray[high + x_position_start][i + y_position_start]))  # base[i][high]))

            if event_map[high + x_position_start][i + y_position_start] == 1 and seg_img[high + x_position_start][i + y_position_start] == 1:
                ax.arrow(i, bbox_size - high, rgb_img_gradient_x[high + x_position_start][i + y_position_start],
                         -rgb_img_gradient_y[high + x_position_start][i + y_position_start], head_width=0.5,
                         head_length=0.5, fc='green', ec='green', length_includes_head=True)
            else:
                pass
    plt.xlim(0, bbox_size - 31)
    plt.ylim(0, bbox_size - 1)

    plt.show()
    norm_rgb_gray *= norm_rgb_gray * 255
    return norm_rgb_gray.astype(np.uint8)


def read_object_bbox(json_file_name, path_dir):
    import os
    import json
    object_info_file = open(os.path.join(path_dir, json_file_name))
    object_info = json.load(object_info_file)
    bbox_top_left = object_info['objects'][0]['bounding_box']['top_left']
    bbox_bottom_right = object_info['objects'][0]['bounding_box']['bottom_right']
    timestamp = object_info['timestamp']
    x_position = bbox_top_left[0]  # 0:480
    y_position = bbox_top_left[1]  # 0:640
    width = bbox_bottom_right[1] - y_position
    height = bbox_bottom_right[0] - x_position
    return int(x_position - 15), int(y_position - 15), int(width + 45), int(height + 45), timestamp


# using plotly lib, visualize 3D image for gradients
def visual_img_plotly(gradient_x, gradient_y, event_map, img_height, img_width, json_file_name, path_dir):
    import plotly.graph_objects as go
    x_position_start, y_position_start, width, height = read_object_bbox(json_file_name, path_dir)
    event_map_x = np.zeros((img_height, img_width))
    event_map_y = np.zeros((img_height, img_width))
    event_position = event_map > 0
    event_map_x[event_position] = gradient_x[event_position]
    event_map_y[event_position] = gradient_y[event_position]
    position = np.where(event_position == True)
    x_position = position[1] - y_position_start  # [0:640]
    y_position = position[0] - x_position_start  # [0:320]
    z_value_x = []
    z_value_y = []
    for i in range(x_position.shape[0]):
        z_value_x.append(event_map_x[y_position[i] + x_position_start, x_position[i] + y_position_start])
        z_value_y.append(event_map_y[y_position[i] + x_position_start, x_position[i] + y_position_start])

    trace_x = go.Scatter3d(
        x=x_position,  # [0:640]
        y=y_position,  # [0:320]
    )
    trace_y = go.Scatter3d(
        x=x_position,  # [0:640]
        y=y_position,  # [0:320]
        z=np.array(z_value_y),
        mode='markers',
        marker=dict(
            size=3,
            color='rgb(0,139,139)',  # color reference：https://tool.oschina.net/commons?type=3
        )
    )

    fig1 = go.Figure(data=[
        go.Surface(z=gradient_x[x_position_start:x_position_start + height, y_position_start:y_position_start + width]),
        trace_x])
    fig1.show()
    fig2 = go.Figure(data=[
        go.Surface(z=gradient_y[x_position_start:x_position_start + height, y_position_start:y_position_start + width]),
        trace_y])
    fig2.show()
    print('test')


def read_object_bbox_fc(json_file_name, path_dir):
    # this function is for testing franco's method
    object_info_file = open(os.path.join(path_dir, json_file_name))
    object_info = json.load(object_info_file)
    bbox_top_left = object_info['objects'][0]['bounding_box']['top_left']
    bbox_bottom_right = object_info['objects'][0]['bounding_box']['bottom_right']
    timestamp = object_info['timestamp']
    x_position = (bbox_top_left[0] + bbox_bottom_right[0])/2  # 0:480
    y_position = (bbox_top_left[1] + bbox_bottom_right[1])/2  # 0:640
    width = bbox_bottom_right[1] - y_position
    height = bbox_bottom_right[0] - x_position
    return int(x_position), int(y_position), int(width + 45), int(height + 45), timestamp


def read_timestamp(json_file_name, path_dir):
    object_info_file = open(os.path.join(path_dir, json_file_name))
    object_info = json.load(object_info_file)
    timestamp = object_info['timestamp']
    return timestamp


def read_dope_poses(file_path):
    import pandas as pd
    import os
    csv_path = file_path.split(".txt")[0] + ".csv"
    with open(file_path, encoding='utf-8') as f:
        contents = []
        readlines = f.readlines()  # readlines是一个列表
        for i in readlines:
            line = i.strip().split(" ")  # 按逗号分割开
            contents.append(line)  # contents二维列表
    df = pd.DataFrame(contents)
    df.to_csv(csv_path, header=False, index=False)

    dope_poses = []
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            dope_poses.append(line)
    return dope_poses


def quaternion_from_rodrigues(rotation_vector):  # [x,y,z,w]
    return Quaternion.Quaternion(axis=rotation_vector[0, 0:3], angle=rotation_vector[0, 3])  # Using radians and a Numpy 3 - array

