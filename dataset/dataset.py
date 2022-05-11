# generate and get dataset from tfrecord files
import math
import os
import json
import random
import time

import cv2
import numpy as np
import tensorflow as tf

from utils.pointcloud_utils import read_bin, extract_pc_in_box2d, random_sample_numpy, fit_traj, load_pcd_data
from utils.map_utils import get_carla_pose, get_navg_maps

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


# all lidar global sample_poses
def get_poses(file_path):
    sample_poses = []
    f = open(file_path)
    lines = f.readlines()

    last_row = np.array([[0, 0, 0, 1]], dtype=np.float)
    T_cam2lidar = np.array([[0, 0, 1, 2],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0.08],
                            [0, 0, 0, 1]])
    T_lidar2cam = np.linalg.inv(T_cam2lidar)
    for line in lines:
        line = line.strip('\n')  # 将\n去掉
        line = np.array(line.split(" "), dtype=np.float)
        line = np.reshape(line, newshape=[3, 4])
        # 齐次
        line = np.concatenate([line, last_row], axis=0)
        T = T_cam2lidar @ line @ T_lidar2cam
        sample_poses.append(T)
    f.close()
    return sample_poses


def get_rt(x, y, yaw):
    rt = np.array([[math.cos(yaw), -math.sin(yaw), 0, x],
                   [math.sin(yaw), math.cos(yaw), 0, y],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
    return rt


class GoalDataset:
    def __init__(self, cfg):
        self.env = cfg.env
        self.train_sequences = cfg.train_sequences
        self.val_sequences = cfg.val_sequences
        self.root_dir = cfg.root_dir

        self.lidar_height = cfg.lidar_height
        self.horizon_max = cfg.horizon_max

        self.tfrecords_val_dir = self.root_dir + cfg.tfrecords_val_dir
        self.tfrecords_train_dir = self.root_dir + cfg.tfrecords_train_dir
        self.num_points = cfg.num_points
        self.box_2d = cfg.grid_range

        self.voxel_size = cfg.voxel_size
        self.max_points_voxel = cfg.max_points_voxel
        self.pc_range = cfg.pc_range
        self.max_points_voxel = cfg.max_points_voxel
        self.max_voxels = cfg.max_voxels
        self.save_every_frames = cfg.save_every_frames

        self.batch_size = cfg.batch_size
        self.shuffle_buff = cfg.shuffle_buff

    def get_train_pcd(self, file_path, num_points, box_2d, yaw):
        # NOTE(swc): z+=lidar_height
        pcd_data = np.load(file_path).reshape([-1, 4])
        # from right-forward-top to forward-left-top
        temp = pcd_data[:, 0].copy()
        pcd_data[:, 0] = pcd_data[:, 1]
        pcd_data[:, 1] = -temp
        pcd_data[:, 2] += self.lidar_height

        pcd_data = extract_pc_in_box2d(pcd_data, box_2d)
        pcd_data = random_sample_numpy(pcd_data, num_points)

        rotate_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                               [math.sin(yaw), math.cos(yaw), 0],
                               [0, 0, 1]])

        r_points = np.transpose(pcd_data[:, :3])
        r_points = rotate_mat @ r_points
        r_points = np.transpose(r_points)
        pcd_data[:, :3] = r_points
        pcd_data = pcd_data.astype(np.float32)
        return pcd_data

    def generate_d_sampling_data(self, is_training=True):
        # get points and trajectory points
        if is_training:
            sequences = self.train_sequences
        else:
            sequences = self.val_sequences
        dirs = []
        for town in sequences:
            dirs.append(os.path.join(self.root_dir, town + '_short'))

        pcds, directions, goals = [], [], []
        counts = 0
        for sub_root in dirs:
            root_files = os.listdir(sub_root)
            routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root, folder))]

            for route in routes:
                route_dir = os.path.join(sub_root, route)
                # print(route_dir)
                lidar_dir = route_dir + "/lidar/"
                measure_dir = route_dir + "/measurements/"
                lidar_files = os.listdir(lidar_dir)
                measure_files = os.listdir(measure_dir)
                lidar_files.sort()
                measure_files.sort()

                poses = []
                goals = []
                for frame in range(len(measure_files)):
                    with open(measure_dir + measure_files[frame], "r") as read_file:
                        measurement = json.load(read_file)
                    this_pose = get_rt(measurement['x'], measurement['y'], measurement['theta'])
                    this_goal = get_rt(measurement['x_command'], measurement['y_command'], 0)
                    poses.append(this_pose)
                    goals.append(this_goal)

                for frame in range(len(lidar_files)):
                    lidar_path = lidar_dir + lidar_files[frame]
                    yaw = random.randint(-10, 10)
                    yaw = math.radians(yaw)
                    if not is_training:
                        yaw = 0
                    pcd_data = self.get_train_pcd(lidar_path, self.num_points, self.box_2d, yaw)
                    this_direction_params, this_goal = fit_traj(poses[frame:], horizon_max=self.horizon_max, yaw=yaw,
                                                                vis=False, goal=goals[frame])
                    if this_direction_params is None:
                        continue
                    pcds.append(pcd_data)
                    directions.append(this_direction_params)
                    goals.append(this_goal)
                    counts += 1
                    print(f"counts = {counts}")

        dataset = tf.data.Dataset.from_tensor_slices((pcds, directions, goals))
        return dataset


if __name__ == '__main__':
    import yaml
    from utils.pointcloud_utils import read_bin, show_trajectorys, fit_traj
    from data_augument import random_flip

    # config_file = '../config/carla_config.yaml'
    config_file = '../config/carla_goal_config.yaml'
    if os.path.isfile(config_file):
        print("using config file:", config_file)
        with open(config_file) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)


        class ConfigClass:
            def __init__(self, **entries):
                self.__dict__.update(entries)


        cfg = ConfigClass(**config_dict)  # convert python dict to class for ease of use

    else:
        print("=> no config file found at '{}'".format(config_file))

    print("train_sequences:", cfg.train_sequences)
    print("val_sequences:", cfg.val_sequences)
    start = time.time()
    dataset = GoalDataset(cfg=cfg)
    train_data = dataset.generate_d_sampling_data(is_training=True).batch(4)
    end = time.time()
    print(f"data time: {end - start:.1f}s")
    # train_data = dataset.get_dataset(True)
    # train_data = train_data.batch(cfg.batch_size, drop_remainder=True)
    #
    for idx, data in enumerate(train_data):
        batch_sz = data[0].shape[0]
        pcds, goals, gts = data
        pcds, goals, gts = random_flip(pcds, gts, goals=goals)
        for i in range(batch_sz):
            # pcd = pcds[i].numpy()
            # goal = goals[i].numpy()
            # direction = gts[i].numpy()
            show_trajectorys(pcds[i], pred=gts[i], show_frame=i, goal_point=goals[i])
            print(f'idx:{idx}',
                  data[0][i].numpy().shape,
                  data[1][i].numpy().shape,
                  data[2][i].numpy().shape)
