# generate and get dataset from tfrecord files
import sys

sys.path.append('..')
import math
import os
import random
import cv2
import numpy as np
import tensorflow as tf

from utils.pointcloud_utils import read_bin, extract_pc_in_box2d, random_sample_numpy, fit_traj
from utils.map_utils import get_navg_maps

from utils.kitti_utils import show_bev


def get_navg_maps(files_path):
    nav_maps = []
    files = os.listdir(files_path)
    files.sort()
    frames = len(files)
    for frame in range(frames):
        nav_maps.append(files_path + files[frame])
    return nav_maps


class RealWorldDataset:
    def __init__(self, cfg):
        self.env = cfg.env
        self.train_sequences = cfg.train_sequences
        self.val_sequences = cfg.val_sequences
        self.root_dir = cfg.root_dir

        self.lidar_height = cfg.lidar_height
        self.horizon_max = cfg.in_horizon_max
        self.tfrecords_val_dir = cfg.tfrecords_val_dir
        self.tfrecords_train_dir = cfg.tfrecords_train_dir
        self.num_points = cfg.in_num_points
        self.save_every_frames = cfg.save_every_frames

    def generate_d_sampling_data(self, is_training=True):
        # get points and trajectory points
        if is_training:
            sequences = self.train_sequences
            tfrecords_dir = self.tfrecords_train_dir
        else:
            sequences = self.val_sequences
            tfrecords_dir = self.tfrecords_val_dir

        for sequence in sequences:
            lidar_dir = self.root_dir + f"{sequence:0>2d}/lidar/"
            map_dir = self.root_dir + f"{sequence:0>2d}/local_map/"
            waypoints_dir = self.root_dir + f"{sequence:0>2d}/waypoints/"
            navg_maps = get_navg_maps(map_dir)

            total_lidar_files = os.listdir(lidar_dir)
            total_lidar_files.sort()

            waypoints_files = os.listdir(waypoints_dir)
            waypoints_files.sort()

            total_frames = min(min(len(navg_maps), len(waypoints_files)), len(total_lidar_files))
            tfrecord_file = tfrecords_dir + f'{sequence}_0.tfrecord'
            writer = tf.io.TFRecordWriter(tfrecord_file)

            for frame in range(total_frames):
                lidar_path = lidar_dir + total_lidar_files[frame]
                waypoints_path = waypoints_dir + waypoints_files[frame]

                single_pcd_data = read_bin(lidar_path, self.lidar_height)
                single_pcd_data = random_sample_numpy(single_pcd_data, N=self.num_points)
                this_direction_params = np.loadtxt(waypoints_path)

                if this_direction_params is None:
                    continue
                # tobytes
                single_pcd_data = single_pcd_data.astype(np.float32)
                single_pcd_data = single_pcd_data.tobytes()
                this_direction_params = this_direction_params.astype(np.float32)
                this_direction_params = this_direction_params.tobytes()

                this_navg_map = cv2.imread(navg_maps[frame])
                this_navg_map = this_navg_map.astype(np.float32)
                this_navg_map = this_navg_map.tobytes()

                if frame % self.save_every_frames == 0 and frame > 0:
                    tfrecord_file = tfrecords_dir + f'{sequence}_{frame // self.save_every_frames}.tfrecord'
                    writer = tf.io.TFRecordWriter(tfrecord_file)

                carla_feature = {
                    'single_pcd_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[single_pcd_data])),
                    'navg_maps': tf.train.Feature(bytes_list=tf.train.BytesList(value=[this_navg_map])),
                    'this_direction_params': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[this_direction_params]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=carla_feature))
                # 将Example序列化
                serialized = example.SerializeToString()
                # write
                writer.write(serialized)
                print(
                    f'{sequence}_{frame // self.save_every_frames}.tfrecord  {frame % self.save_every_frames}/{self.save_every_frames}')

    def get_dataset(self, train=True):
        # 读取TFRecord文件
        # 定义Feature结构，告诉编码器每个Feature的类型是什么
        carla_feature_description = {
            'single_pcd_data': tf.io.FixedLenFeature([], tf.string),
            'navg_maps': tf.io.FixedLenFeature([], tf.string),
            'this_direction_params': tf.io.FixedLenFeature([], tf.string)
        }

        # 解码序列化的Example
        def carla_parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, carla_feature_description)
            feature_dict['single_pcd_data'] = tf.io.decode_raw(feature_dict['single_pcd_data'], out_type=tf.float32)
            feature_dict['navg_maps'] = tf.io.decode_raw(feature_dict['navg_maps'], out_type=tf.float32)
            feature_dict['this_direction_params'] = tf.io.decode_raw(feature_dict['this_direction_params'],
                                                                     out_type=tf.float32)

            feature_dict['single_pcd_data'] = tf.reshape(feature_dict['single_pcd_data'], shape=[-1, 4])
            feature_dict['navg_maps'] = tf.reshape(feature_dict['navg_maps'], shape=[100, 100, 3])

            return feature_dict['single_pcd_data'], feature_dict['navg_maps'], feature_dict['this_direction_params']

        if train:
            train_tfrecord_files = [self.tfrecords_train_dir + one_dir for one_dir in
                                    os.listdir(self.tfrecords_train_dir)]
            # shuffle
            random.shuffle(train_tfrecord_files)
            train_dataset = tf.data.TFRecordDataset(train_tfrecord_files)
            train_dataset = train_dataset.map(carla_parse_example)
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
    # config_file = '../config/realworld.yaml'
    config_file = '../config/remote_realworld.yaml'
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

    dataset = RealWorldDataset(cfg=cfg)
    dataset.generate_d_sampling_data(is_training=False)
    if (0):
        train_data = dataset.get_dataset(False)
        train_data = train_data.batch(cfg.batch_size, drop_remainder=True)

        for idx, data in enumerate(train_data):
            batch_sz = data[0].shape[0]
            single_pcds, maps, gts = data
            single_pcds = single_pcds.numpy()
            maps = maps.numpy()
            gts = gts.numpy()
            # maps = dilate_map(maps)
            # single_pcds, maps, gts = random_flip(single_pcds, gts=gts, maps=maps)
            for i in range(batch_sz):
                pcd = single_pcds[i]
                direction = gts[i]
                map = maps[i]
                show_trajectorys(pcd, pred=direction, show_frame=i, navg_map=map)
                print(f'idx:{idx}',
                      data[0][i].numpy().shape,
                      data[1][i].numpy().shape,
                      data[2][i].numpy().shape,
                      )
