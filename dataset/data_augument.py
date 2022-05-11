import os
import math
import time
import cv2
import random
import numpy as np
import numba
import tensorflow as tf

from utils.pointcloud_utils import extract_pc_in_box2d, random_sample_numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息,去掉tf一些烦人的输出


def random_flip(pcds, gts, maps, student_data=None):
    # pcds: [B,N,4]
    # maps: [B,100,100,3]
    # gts: [B,41]
    B = pcds.shape[0]
    if student_data is None:
        for b in range(B):
            p = random.random()
            if p < 0.5:
                continue
            pcds[b, :, 1] = -pcds[b, :, 1]
            gts[b, 1:] = -gts[b, 1:]
            maps[b] = maps[b, :, ::-1, :].copy()
        return pcds, maps, gts
    else:
        for b in range(B):
            p = random.random()
            if p < 0.5:
                continue
            pcds[b, :, 1] = -pcds[b, :, 1]
            student_data[b, :, 1] = -student_data[b, :, 1]
            gts[b, 1:] = -gts[b, 1:]
            maps[b] = maps[b, :, ::-1, :].copy()
        return pcds, student_data, maps, gts


def random_shift(pcds, gts):
    B = pcds.shape[0]
    for b in range(B):
        y_shift = random.uniform(-5, 5)
        pcds[b, :, 1] += y_shift
        gts[b, 1:] += y_shift
    return pcds, gts


@numba.jit(nopython=True)
def shuffle_points(pcds):
    shuffle_idx = np.random.permutation(pcds.shape[0])
    points = pcds[shuffle_idx]
    return points


@numba.jit(nopython=True)
def _rotate_points(this_points, rotate_mat, box_2d, num_points):
    this_points = shuffle_points(this_points)
    r_points = np.transpose(this_points[:, :3])
    r_points = rotate_mat @ r_points
    r_points = np.transpose(r_points)
    this_points[:, :3] = r_points

    this_points = extract_pc_in_box2d(this_points, box_2d)
    this_points = random_sample_numpy(this_points, num_points)
    return this_points


def get_rotate_pcds(in_points, num_points, box_2d, yaw):
    # in_points: (N,4)
    rotate_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                           [math.sin(yaw), math.cos(yaw), 0],
                           [0, 0, 1]], dtype=np.float32)

    out_points = _rotate_points(in_points, rotate_mat, box_2d, num_points)
    out_points = out_points.astype(np.float32)
    return out_points


@numba.jit(nopython=True)
def get_index(x_points, x_s):
    # get the index of the first x bigger than x_s
    index = -1
    for (x_idx, x_point) in enumerate(x_points):
        if x_point > x_s:
            index = x_idx
            break
    return index


@numba.jit(nopython=True)
def _get_rotate_waypoints(x_points, y_points, horizon_res, max_s, is_negative):
    length = 0
    out_waypoints = []
    for x_s in np.arange(horizon_res, max_s, horizon_res):
        j_index = get_index(x_points, x_s)
        if j_index == -1:
            break
        length += 1
        i_index = j_index - 1
        x_i = x_points[i_index]
        y_i = y_points[i_index]

        x_j = x_points[j_index]
        y_j = y_points[j_index]

        y_s = (y_j - y_i) / (x_j - x_i) * (x_s - x_i) + y_i
        # FIXME(swc): -y_s: only for carla_first
        if is_negative:
            out_waypoints.append(-y_s)
        else:
            out_waypoints.append(y_s)

    return length, out_waypoints


def get_rotate_waypoints(in_waypoints_y,
                         yaw,
                         horizon_max,
                         horizon_res=0.5,
                         in_res=0.2,
                         is_negative=False
                         ):
    # in_waypoints_y: (151)

    rotate_mat = np.array([[math.cos(yaw), -math.sin(yaw)],
                           [math.sin(yaw), math.cos(yaw)]])

    horizon_max = int(horizon_max)
    max_s = horizon_max + horizon_res
    array_size = math.floor(horizon_max / horizon_res) + 1

    in_waypoints = []
    in_length = int(in_waypoints_y[0])
    for idx in range(in_length):
        in_x = in_res * (idx + 1)
        in_y = in_waypoints_y[idx + 1]
        in_waypoints.append([in_x, in_y])
    in_waypoints = np.asarray(in_waypoints)  # (n,2)
    in_waypoints = in_waypoints.transpose()  # (2,n)
    in_waypoints = rotate_mat @ in_waypoints
    in_waypoints = in_waypoints.transpose()

    x_points = in_waypoints[:, 0]
    y_points = in_waypoints[:, 1]

    length, out_waypoints = _get_rotate_waypoints(x_points, y_points, horizon_res, max_s, is_negative)

    out_waypoints = np.concatenate([np.array([length]), np.array(out_waypoints)])

    out_waypoints_y = np.zeros(array_size)
    out_waypoints_y[:length + 1] = out_waypoints
    out_waypoints_y = out_waypoints_y.astype(np.float32)
    return out_waypoints_y


def random_flip_conf(pcds, gts, maps, student_data=None):
    # pcds: [B,N,4]
    # maps: [B,100,100,3]
    # gts: [B,41]
    B = pcds.shape[0]
    if student_data is None:
        for b in range(B):
            p = random.random()
            if p < 0.5:
                continue
            pcds[b, :, 1] = -pcds[b, :, 1]
            gts[b, 1:] = -gts[b, 1:]
            maps[b] = maps[b, :, ::-1, :].copy()
        return pcds, maps, gts
    else:
        for b in range(B):
            p = random.random()
            if p < 0.5:
                continue
            pcds[b, :, 1] = -pcds[b, :, 1]
            student_data[b, :, 1] = -student_data[b, :, 1]
            gts[b, 1:] = -gts[b, 1:]
            maps[b] = maps[b, :, ::-1, :].copy()
        return pcds, student_data, maps, gts,


def add_noise(pcds, sigma=0.01, clip=0.05):
    n = pcds.shape[0]
    jittered_point = np.clip(sigma * np.random.randn(n, 3), -1 * clip, clip)
    pcds[:, :3] += jittered_point
    return pcds


def random_add_obstacles(pcds, gts):
    # pcds: [B,N,4]
    # gts: [B,41]
    # 在历史轨迹点中随机选择几个，在采样点周围随机生成障碍物（长方体，车辆模型点云？）
    pass


def random_scale_waypoints(in_waypoints_y, random_scale, in_res=0.2):
    """
    Args:
        in_waypoints_y: [l, y_1, ..., y_l, 0, ... ,0]
        random_scale: 0.8~1.2
        in_res: 0.2

    Returns: (n,2)
        [[l,l],
         [x,y],
         ...
    """
    in_waypoints = []
    in_length = int(in_waypoints_y[0])
    for idx in range(in_waypoints_y.shape[0] - 1):
        in_x = in_res * (idx + 1) * random_scale
        in_y = in_waypoints_y[idx + 1] * random_scale
        in_waypoints.append([in_x, in_y])
    in_waypoints = np.asarray(in_waypoints)  # (n,2)

    out_waypoints = np.concatenate([np.array([[in_length, in_length]]), in_waypoints])
    return out_waypoints


def random_scaling(in_points, in_waypoints_y):
    """
    Args:
        in_points: (B,N,4)
        in_waypoints_y: (B,M)

    Returns:
        in_points: (B,N,4)
        scaled_waypoints: (B,M,2)
    """
    B = in_points.shape[0]
    waypoint_size = in_waypoints_y[0].shape[0]
    scale_range = [0.8, 1.2]
    scaled_waypoints = np.zeros(shape=(B, waypoint_size, 2))
    for b in range(B):
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        in_points[b][:, :3] *= noise_scale
        this_waypoints_y = in_waypoints_y[b]
        scaled_waypoints_ = random_scale_waypoints(this_waypoints_y)
        scaled_waypoints[b] = scaled_waypoints_
    return in_points, scaled_waypoints


if __name__ == '__main__':
    # pcds = tf.random.normal(shape=(400, 1000, 4))
    #
    # pcds = pcds.numpy()
    #
    # s_time = time.time()
    # pcds = add_noise(pcds)
    # e_time = time.time()
    # print(f"cost time: {(e_time - s_time) * 1000.0:.1f}ms")
    #
    # s_time = time.time()
    # pcds = add_noise(pcds)
    # e_time = time.time()
    # print(f"cost time: {(e_time - s_time) * 1000.0:.1f}ms")
    #
    # s_time = time.time()
    # pcds = add_noise(pcds)
    # e_time = time.time()
    # print(f"cost time: {(e_time - s_time) * 1000.0:.1f}ms")
    a = np.loadtxt('/media/swc/swc/study/LidarData/real_world/data/00/waypoints/000000.txt')
    b = random_scale_waypoints(a, 0.5)
