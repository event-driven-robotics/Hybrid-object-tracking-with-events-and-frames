# This file is for gt velocity validation
import lib_func
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from scipy.linalg import expm


# load gt velocity
def load_cvs_file(path_dir):
    csv_path = path_dir.split('photorealistic1')[0]
    gt_velocities = lib_func.read_velocity(os.path.join(csv_path, 'velocity.csv'))
    return gt_velocities


# calculate pose iteratively
def velocity_to_pose(args, info_list, ratio, delta_t, gt_velocities):
    gt_location = []
    gt_quaternion_xyzw = []
    for i in tqdm(range(int(len(info_list) / ratio))):
        i = int(i * ratio)

        with open(os.path.join(args.frame_json_file_path, info_list[i]), 'r', encoding='utf-8') as f:
            info_json = json.load(f)
            gt_location.append(np.array(info_json['objects'][0]['location']))
            gt_quaternion_xyzw.append(np.array(info_json['objects'][0]['quaternion_xyzw'])[[3, 0, 1, 2]])
    # location
    calculated_location_saved = []
    calculated_location_saved.append(gt_location[0])
    calculated_location = gt_location[0]

    # rotation
    calculated_rotation_saved = []
    calculated_rotation_saved.append(gt_quaternion_xyzw[0])
    calculated_rotation = gt_quaternion_xyzw[0]
    initial_rotation = Quaternion(calculated_rotation).rotation_matrix
    rotation = []
    rotation.append(initial_rotation)
    for i in range(len(info_list) - 1):

        # angular velocity theta*n
        w = np.array(gt_velocities[i + 1]).reshape((1, -1)).astype('float64')[0][3:6]

        # -w*p, m/s
        velocity_compensation = np.cross(w, -1 * calculated_location) / 100  # cm to meter

        # Vo m/s
        Vo = np.array(gt_velocities[i + 1]).reshape((1, -1)).astype('float64')[0][0:3]

        # Vp m/s
        Vp = Vo - velocity_compensation
        
        # location = location + v * delta_t
        calculated_location = calculated_location + Vp * ratio * delta_t * 100  # - velocity_compensation
        calculated_location_saved.append(calculated_location)

        # rotation
        # R1*delta_R = R2
        n_theta = np.array(gt_velocities[i + 1]).reshape((1, -1)).astype('float64')[0][3:6] * ratio * delta_t
        R = np.zeros([3, 3])
        R[2, 1] = n_theta[0]
        R[1, 2] = -n_theta[0]
        R[2, 0] = -n_theta[1]
        R[0, 2] = n_theta[1]
        R[1, 0] = n_theta[2]
        R[0, 1] = -n_theta[2]

        delta_R = expm(R)
        initial_rotation = np.dot(delta_R, initial_rotation)
        rotation.append(initial_rotation)
        quaternion_temp = Quaternion(matrix=initial_rotation).elements
        if quaternion_temp[0] < 0:
            quaternion_temp *= -1
        calculated_rotation_saved.append(quaternion_temp)

    print(calculated_location_saved)

    return gt_location, gt_quaternion_xyzw, calculated_location_saved, calculated_rotation_saved


def draw(gt_location, gt_quaternion_xyzw, calculated_pose_saved, calculated_quaternion_saved, ratio, delta_t):
    x_axis_linear_v = np.arange(0, len(gt_location), 1) * ratio * delta_t
    gt_location = np.array(gt_location)
    calculated_pose_saved = np.array(calculated_pose_saved)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x_axis_linear_v, gt_location[:, 0], label='gt_location_x')
    plt.plot(x_axis_linear_v, gt_location[:, 1], label='gt_location_y')
    plt.plot(x_axis_linear_v, gt_location[:, 2], label='gt_location_z')
    plt.legend()
    plt.title('gt_pose_xyz', fontsize=14)
    plt.xlabel('time(second)', fontsize=12)
    plt.ylabel('cm', fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(x_axis_linear_v, calculated_pose_saved[:, 0], label='calculated_location_x')
    plt.plot(x_axis_linear_v, calculated_pose_saved[:, 1], label='calculated_location_y')
    plt.plot(x_axis_linear_v, calculated_pose_saved[:, 2], label='calculated_location_z')
    plt.legend()
    plt.title('calculated_location_xyz', fontsize=14)
    plt.xlabel('time(second)', fontsize=12)
    plt.ylabel('cm', fontsize=12)

    gt_quaternion_xyzw = np.array(gt_quaternion_xyzw)
    calculated_quaternion_saved = np.array(calculated_quaternion_saved)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x_axis_linear_v, gt_quaternion_xyzw[:, 0], label='gt_quaternion_x')
    plt.plot(x_axis_linear_v, gt_quaternion_xyzw[:, 1], label='gt_quaternion_y')
    plt.plot(x_axis_linear_v, gt_quaternion_xyzw[:, 2], label='gt_quaternion_z')
    plt.plot(x_axis_linear_v, gt_quaternion_xyzw[:, 3], label='gt_quaternion_w')
    plt.legend()
    plt.title('gt_quaternion_xyzw', fontsize=14)
    plt.xlabel('time(second)', fontsize=12)
    plt.ylabel('1', fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(x_axis_linear_v, calculated_quaternion_saved[:, 0], label='calculated_quaternion_saved_x')
    plt.plot(x_axis_linear_v, calculated_quaternion_saved[:, 1], label='calculated_quaternion_saved_y')
    plt.plot(x_axis_linear_v, calculated_quaternion_saved[:, 2], label='calculated_quaternion_saved_z')
    plt.plot(x_axis_linear_v, calculated_quaternion_saved[:, 3], label='calculated_quaternion_saved_w')
    plt.legend()
    plt.title('calculated_quaternion_xyzw', fontsize=14)
    plt.xlabel('time(second)', fontsize=12)
    plt.ylabel('1', fontsize=12)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_json_file_path", type=str,
                        default="/home/lzc/UE_dataset/pitch_502Hz_0_1/photorealistic1/")

    parser.add_argument("--velocity_file_path", type=str,
                        default="/home/lzc/UE_dataset/pitch_502Hz_0_1/velocity.csv")

    args = parser.parse_args()
    print(args.frame_json_file_path)

    raw_data_frequency = 502
    output_frequency = 502

    info_file_path = args.frame_json_file_path
    # velocity_file_path = args.velocity_file_path

    file_list = os.listdir(info_file_path)
    file_list_len = len(file_list)
    info_list = []
    np.set_printoptions(suppress=True)

    # Pick up all RGB images
    for i in range(file_list_len):
        if (len(file_list[i].split('.')) == 2) and (file_list[i].split('.')[1] == 'json') and (
                file_list[i].split('.')[0][0] != '_'):
            info_list.append(file_list[i])

    # sort for info_list
    info_list.sort()

    gt_velocities = load_cvs_file(info_file_path)

    delta_t = 1 / raw_data_frequency
    ratio = raw_data_frequency / output_frequency
    gt_location, gt_quaternion_xyzw, calculated_location_saved, calculated_quaternion_saved = velocity_to_pose(args,
                                                                                                               info_list,
                                                                                                               ratio,
                                                                                                               delta_t,
                                                                                                               gt_velocities)
    draw(gt_location, gt_quaternion_xyzw, calculated_location_saved, calculated_quaternion_saved, ratio, delta_t)


if __name__ == '__main__':
    main()
