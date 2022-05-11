# generate and get dataset from tfrecord files
import math
import os
import json
import random
import cv2
import numpy as np
import tensorflow as tf

from utils.pointcloud_utils import read_bin, extract_pc_in_box2d, random_sample_numpy, fit_traj, load_pcd_data
from utils.map_utils import get_carla_pose, get_navg_maps

def get_rt(x, y, yaw):
    rt = np.array([[math.cos(yaw), -math.sin(yaw), 0, x],
                   [math.sin(yaw), math.cos(yaw), 0, y],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
    return rt


class GoalTFRecordDataset:
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
            tfrecords_dir = self.tfrecords_train_dir
        else:
            sequences = self.val_sequences
            tfrecords_dir = self.tfrecords_val_dir
        dirs = []
        for town in sequences:
            dirs.append(os.path.join(self.root_dir, town + '_short'))

        town_num = 0
        for sub_root in dirs:
            root_files = os.listdir(sub_root)
            routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root, folder))]

            town_name = sequences[town_num]
            town_num += 1

            route_idx = 0
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

                tfrecord_file = tfrecords_dir + town_name + f'_{route_idx}_0.tfrecord'
                writer = tf.io.TFRecordWriter(tfrecord_file)

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
                    # # tobytes
                    pcd_data = pcd_data.tobytes()
                    this_direction_params = this_direction_params.tobytes()
                    this_goal = this_goal.tobytes()
                    #
                    if frame % self.save_every_frames == 0 and frame > 0:
                        tfrecord_file = tfrecords_dir + town_name + f'_{route_idx}_{frame // self.save_every_frames}.tfrecord'
                        writer = tf.io.TFRecordWriter(tfrecord_file)

                    # 建立 tf.train.Example dict
                    carla_feature = {
                        'points': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pcd_data])),
                        'this_goal': tf.train.Feature(bytes_list=tf.train.BytesList(value=[this_goal])),
                        'this_direction_params': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[this_direction_params]))
                    }

                    # 通过字典建立Example
                    example = tf.train.Example(features=tf.train.Features(feature=carla_feature))
                    # 将Example序列化
                    serialized = example.SerializeToString()
                    # write
                    writer.write(serialized)
                    tfrecord_name = tfrecord_file.split('/')[-1]
                    print(
                        tfrecord_name + f' {frame % self.save_every_frames}/{self.save_every_frames}')

                route_idx += 1

    def get_dataset(self, train=True):
        # 读取TFRecord文件
        # 定义Feature结构，告诉编码器每个Feature的类型是什么
        carla_feature_description = {
            'points': tf.io.FixedLenFeature([], tf.string),
            'this_goal': tf.io.FixedLenFeature([], tf.string),
            'this_direction_params': tf.io.FixedLenFeature([], tf.string)
        }

        # 解码序列化的Example
        def carla_parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, carla_feature_description)
            feature_dict['points'] = tf.io.decode_raw(feature_dict['points'], out_type=tf.float32)
            feature_dict['this_goal'] = tf.io.decode_raw(feature_dict['this_goal'], out_type=tf.float32)
            feature_dict['this_direction_params'] = tf.io.decode_raw(feature_dict['this_direction_params'],
                                                                     out_type=tf.float32)

            feature_dict['points'] = tf.reshape(feature_dict['points'], shape=[-1, 4])
            feature_dict['this_goal'] = tf.reshape(feature_dict['this_goal'], shape=[2])
            direction_shape = self.horizon_max * 2 + 1
            feature_dict['this_direction_params'] = tf.reshape(feature_dict['this_direction_params'],
                                                               shape=[direction_shape])

            return feature_dict['points'], feature_dict['this_goal'], feature_dict['this_direction_params']

        if train:
            train_files = os.listdir(self.tfrecords_train_dir)
            train_files.sort()
            train_tfrecord_files = [self.tfrecords_train_dir + one_dir for one_dir in train_files]
            # shuffle
            random.shuffle(train_tfrecord_files)
            train_dataset = tf.data.TFRecordDataset(train_tfrecord_files)
            train_dataset = train_dataset.map(carla_parse_example)
            # lidar_file = train_dataset.batch(self.batch_size)  # drop_remainder = False
            data = train_dataset
        else:
            val_files = os.listdir(self.tfrecords_val_dir)
            val_files.sort()
            val_tfrecord_files = [self.tfrecords_val_dir + one_dir for one_dir in val_files]
            val_dataset = tf.data.TFRecordDataset(val_tfrecord_files)
            val_dataset = val_dataset.map(carla_parse_example)
            data = val_dataset
        return data


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

    dataset = GoalTFRecordDataset(cfg=cfg)
    dataset.generate_d_sampling_data(is_training=False)
    # train_data = dataset.get_dataset(True)
    # train_data = train_data.batch(cfg.batch_size, drop_remainder=True)
    #
    # for idx, lidar_file in enumerate(train_data):
    #     batch_sz = lidar_file[0].shape[0]
    #     pcds, goals, gts = lidar_file
    #     pcds, goals, gts = random_flip(pcds, gts, goals=goals)
    #     for i in range(batch_sz):
    #         # pcd = pcds[i].numpy()
    #         # goal = goals[i].numpy()
    #         # direction = gts[i].numpy()
    #         show_trajectorys(pcds[i], pred=gts[i], show_frame=i, goal_point=goals[i])
    #         print(f'idx:{idx}',
    #               lidar_file[0][i].numpy().shape,
    #               lidar_file[1][i].numpy().shape,
    #               lidar_file[2][i].numpy().shape)
