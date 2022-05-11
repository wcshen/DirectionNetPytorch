# generate and get dataset from tfrecord files
import sys

# sys.path.append('..')
import math
import os
import random
import cv2
import numpy as np
import tensorflow as tf

from utils.pointcloud_utils import read_bin, extract_pc_in_box2d, random_sample_numpy, fit_traj, load_pcd_data
from utils.map_utils import get_carla_pose, get_navg_maps
from DA.downsample_normal import DAPreprocess
from utils.cross_sensor_utils import get_poses
from utils.kitti_utils import show_bev


# all lidar global sample_poses
def get_kitti_poses(file_path):
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


class TFRecordDataset:
    def __init__(self, cfg):
        self.env = cfg.env
        self.train_towns = cfg.train_towns
        self.train_sequences = cfg.train_sequences
        self.val_towns = cfg.val_towns
        self.val_sequences = cfg.val_sequences
        self.root_dir = cfg.root_dir

        self.lidar_height = cfg.lidar_height
        # FIXME(swc)
        self.horizon_max = 20  # cfg.horizon_max

        self.tfrecords_val_dir = cfg.tfrecords_val_dir
        self.tfrecords_train_dir = cfg.tfrecords_train_dir
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

        self.da_preprocess = DAPreprocess(max_frames=1, beam=64)

    def get_train_pcd(self, file_path, this_pose, num_points, box_2d, yaw):
        # NOTE(swc): z+=lidar_height
        single_pcd_data = np.load(file_path)
        self.da_preprocess.set_pcds(single_pcd_data)

        lidar16, lidar32, lidar40 = self.da_preprocess.downsample_64()

        rotate_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                               [math.sin(yaw), math.cos(yaw), 0],
                               [0, 0, 1]])

        r_points = np.transpose(lidar16[:, :3])
        r_points = rotate_mat @ r_points
        r_points = np.transpose(r_points)
        lidar16[:, :3] = r_points
        lidar16 = lidar16.astype(np.float32)

        r_points = np.transpose(lidar32[:, :3])
        r_points = rotate_mat @ r_points
        r_points = np.transpose(r_points)
        lidar32[:, :3] = r_points
        lidar32 = lidar32.astype(np.float32)

        r_points = np.transpose(lidar40[:, :3])
        r_points = rotate_mat @ r_points
        r_points = np.transpose(r_points)
        lidar40[:, :3] = r_points
        lidar40 = lidar40.astype(np.float32)

        r_points = np.transpose(single_pcd_data[:, :3])
        r_points = rotate_mat @ r_points
        r_points = np.transpose(r_points)
        single_pcd_data[:, :3] = r_points
        single_pcd_data = single_pcd_data.astype(np.float32)

        lidar16 = extract_pc_in_box2d(lidar16, box_2d)
        lidar16 = random_sample_numpy(lidar16, num_points)

        lidar32 = extract_pc_in_box2d(lidar32, box_2d)
        lidar32 = random_sample_numpy(lidar32, num_points)

        lidar40 = extract_pc_in_box2d(lidar40, box_2d)
        lidar40 = random_sample_numpy(lidar40, num_points)

        single_pcd_data = extract_pc_in_box2d(single_pcd_data, box_2d)
        single_pcd_data = random_sample_numpy(single_pcd_data, num_points)

        single_pcd_data[:, 2] += self.lidar_height
        lidar16[:, 2] += self.lidar_height
        lidar32[:, 2] += self.lidar_height
        lidar40[:, 2] += self.lidar_height

        return lidar16, lidar32, lidar40, single_pcd_data

    def generate_d_sampling_data(self, is_training=True):
        # get points and trajectory points
        if is_training:
            towns = self.train_towns
            sequences = self.train_sequences
            tfrecords_dir = self.tfrecords_train_dir
        else:
            towns = self.val_towns
            sequences = self.val_sequences
            tfrecords_dir = self.tfrecords_val_dir

        for town, this_sequences in zip(towns, sequences):
            for sequence in this_sequences:
                lidar_dir = self.root_dir + town + f"/{sequence:0>2d}/lidar64/"
                pose_path = self.root_dir + town + f"/poses/{sequence:0>2d}.txt"
                map_dir = self.root_dir + town + f"/{sequence:0>2d}/local_map/"
                navg_maps = get_navg_maps(map_dir)
                sample_poses = get_poses(pose_path)

                total_lidar_files = os.listdir(lidar_dir)
                total_lidar_files.sort()
                total_frames = len(navg_maps)
                tfrecord_file = tfrecords_dir + f'{town}_{sequence}_0.tfrecord'
                writer = tf.io.TFRecordWriter(tfrecord_file)

                for frame in range(total_frames):
                    lidar_path = lidar_dir + total_lidar_files[frame]
                    this_pose = sample_poses[frame]
                    # lidar_file augument
                    yaw = random.randint(-20, 20)
                    yaw = math.radians(yaw)
                    if not is_training:
                        yaw = 0
                    lidar16, lidar32, lidar40, single_pcd_data = self.get_train_pcd(lidar_path, this_pose,
                                                                                    self.num_points, self.box_2d,
                                                                                    yaw)  # (N,4)
                    this_direction_params = fit_traj(sample_poses[frame:], horizon_max=self.horizon_max, yaw=yaw,
                                                     is_negative=False)

                    if this_direction_params is None:
                        continue
                    # tobytes
                    single_pcd_data = single_pcd_data.tobytes()
                    lidar16 = lidar16.tobytes()
                    lidar32 = lidar32.tobytes()
                    lidar40 = lidar40.tobytes()
                    this_direction_params = this_direction_params.tobytes()
                    this_navg_map = None
                    if self.env == 'carla':
                        this_navg_map = cv2.imread(navg_maps[frame])
                        this_navg_map = this_navg_map.astype(np.float32)
                        this_navg_map = this_navg_map.tobytes()

                    if frame % self.save_every_frames == 0 and frame > 0:
                        tfrecord_file = tfrecords_dir + f'{town}_{sequence}_{frame // self.save_every_frames}.tfrecord'
                        writer = tf.io.TFRecordWriter(tfrecord_file)

                    carla_feature = {
                        'lidar16': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lidar16])),
                        'lidar32': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lidar32])),
                        'lidar40': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lidar40])),
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
                        f'{town}_{sequence}_{frame // self.save_every_frames}.tfrecord  {frame % self.save_every_frames}/{self.save_every_frames}')

    def generate_single_sampling_data(self, is_training=True):
        # get points and trajectory points
        if is_training:
            towns = self.train_towns
            sequences = self.train_sequences
            tfrecords_dir = self.tfrecords_train_dir
        else:
            towns = self.val_towns
            sequences = self.val_sequences
            tfrecords_dir = self.tfrecords_val_dir

        for town, this_sequences in zip(towns, sequences):
            for sequence in this_sequences:
                lidar_dir = self.root_dir + town + f"/{sequence:0>2d}/lidar32/"
                pose_path = self.root_dir + town + f"/poses/{sequence:0>2d}.txt"
                map_dir = self.root_dir + town + f"/{sequence:0>2d}/local_map/"
                navg_maps = get_navg_maps(map_dir)
                sample_poses = get_poses(pose_path)

                total_lidar_files = os.listdir(lidar_dir)
                total_lidar_files.sort()
                total_frames = len(navg_maps)
                tfrecord_file = tfrecords_dir + f'{town}_{sequence}_0.tfrecord'
                writer = tf.io.TFRecordWriter(tfrecord_file)

                for frame in range(total_frames):
                    lidar_path = lidar_dir + total_lidar_files[frame]
                    # lidar_file augument
                    yaw = random.randint(-20, 20)
                    yaw = math.radians(yaw)
                    if not is_training:
                        yaw = 0
                    single_pcd_data = np.load(lidar_path)
                    rotate_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                           [math.sin(yaw), math.cos(yaw), 0],
                                           [0, 0, 1]])

                    r_points = np.transpose(single_pcd_data[:, :3])
                    r_points = rotate_mat @ r_points
                    r_points = np.transpose(r_points)
                    single_pcd_data[:, :3] = r_points
                    single_pcd_data = single_pcd_data.astype(np.float32)
                    single_pcd_data = extract_pc_in_box2d(single_pcd_data, self.box_2d)
                    single_pcd_data = random_sample_numpy(single_pcd_data, self.num_points)
                    single_pcd_data[:, 2] += self.lidar_height

                    this_direction_params = fit_traj(sample_poses[frame:], horizon_max=self.horizon_max, yaw=yaw,
                                                     is_negative=False)

                    if this_direction_params is None:
                        continue
                    # tobytes
                    single_pcd_data = single_pcd_data.tobytes()
                    this_direction_params = this_direction_params.tobytes()
                    this_navg_map = None
                    if self.env == 'carla':
                        this_navg_map = cv2.imread(navg_maps[frame])
                        this_navg_map = this_navg_map.astype(np.float32)
                        this_navg_map = this_navg_map.tobytes()

                    if frame % self.save_every_frames == 0 and frame > 0:
                        tfrecord_file = tfrecords_dir + f'{town}_{sequence}_{frame // self.save_every_frames}.tfrecord'
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
                        f'{town}_{sequence}_{frame // self.save_every_frames}.tfrecord  {frame % self.save_every_frames}/{self.save_every_frames}')

    def get_dataset(self, train=True):
        # 读取TFRecord文件
        # 定义Feature结构，告诉编码器每个Feature的类型是什么
        kitti_feature_description = {
            'points': tf.io.FixedLenFeature([], tf.string),
            'this_direction_params': tf.io.FixedLenFeature([], tf.string)
        }

        carla_feature_description = {
            'lidar16': tf.io.FixedLenFeature([], tf.string),
            'lidar32': tf.io.FixedLenFeature([], tf.string),
            'lidar40': tf.io.FixedLenFeature([], tf.string),
            'single_pcd_data': tf.io.FixedLenFeature([], tf.string),
            'navg_maps': tf.io.FixedLenFeature([], tf.string),
            'this_direction_params': tf.io.FixedLenFeature([], tf.string)
        }

        # 解码序列化的Example
        def kitti_parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, kitti_feature_description)
            feature_dict['points'] = tf.io.decode_raw(feature_dict['points'], out_type=tf.float32)
            feature_dict['this_direction_params'] = tf.io.decode_raw(feature_dict['this_direction_params'],
                                                                     out_type=tf.float32)

            feature_dict['points'] = tf.reshape(feature_dict['points'], shape=[-1, 4])
            direction_shape = self.horizon_max * 2 + 1
            feature_dict['this_direction_params'] = tf.reshape(feature_dict['this_direction_params'],
                                                               shape=[direction_shape])

            return feature_dict['points'], feature_dict['this_direction_params']

        def carla_parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, carla_feature_description)
            feature_dict['lidar16'] = tf.io.decode_raw(feature_dict['lidar16'], out_type=tf.float32)
            feature_dict['lidar32'] = tf.io.decode_raw(feature_dict['lidar32'], out_type=tf.float32)
            feature_dict['lidar40'] = tf.io.decode_raw(feature_dict['lidar40'], out_type=tf.float32)
            feature_dict['single_pcd_data'] = tf.io.decode_raw(feature_dict['single_pcd_data'], out_type=tf.float32)
            feature_dict['navg_maps'] = tf.io.decode_raw(feature_dict['navg_maps'], out_type=tf.float32)
            feature_dict['this_direction_params'] = tf.io.decode_raw(feature_dict['this_direction_params'],
                                                                     out_type=tf.float32)

            feature_dict['single_pcd_data'] = tf.reshape(feature_dict['single_pcd_data'], shape=[-1, 4])
            feature_dict['lidar16'] = tf.reshape(feature_dict['lidar16'], shape=[-1, 4])
            feature_dict['lidar32'] = tf.reshape(feature_dict['lidar32'], shape=[-1, 4])
            feature_dict['lidar40'] = tf.reshape(feature_dict['lidar40'], shape=[-1, 4])
            feature_dict['navg_maps'] = tf.reshape(feature_dict['navg_maps'], shape=[100, 100, 3])

            direction_shape = self.horizon_max * 2 + 1
            feature_dict['this_direction_params'] = tf.reshape(feature_dict['this_direction_params'],
                                                               shape=[direction_shape])

            return feature_dict['lidar16'], feature_dict['lidar32'], feature_dict['lidar40'], feature_dict[
                'single_pcd_data'], feature_dict['navg_maps'], feature_dict['this_direction_params']

        if train:
            train_tfrecord_files = [self.tfrecords_train_dir + one_dir for one_dir in
                                    os.listdir(self.tfrecords_train_dir)]
            # shuffle
            random.shuffle(train_tfrecord_files)
            train_dataset = tf.data.TFRecordDataset(train_tfrecord_files)
            if self.env == 'kitti':
                train_dataset = train_dataset.map(kitti_parse_example)
            else:
                train_dataset = train_dataset.map(carla_parse_example)
            # lidar_file = train_dataset.batch(self.batch_size)  # drop_remainder = False
            data = train_dataset
        else:
            val_files = os.listdir(self.tfrecords_val_dir)
            # for i in range(len(val_files)):
            #     val_files[i] = val_files[i].split('.')
            #     val_files[i][0] = int(val_files[i][0])
            # val_files.sort()
            # for i in range(len(val_files)):
            #     val_files[i][0] = str(val_files[i][0])
            #     val_files[i] = val_files[i][0] + '.' + val_files[i][1]
            val_files.sort()
            # print(val_files)
            val_tfrecord_files = [self.tfrecords_val_dir + one_dir for one_dir in val_files]
            val_dataset = tf.data.TFRecordDataset(val_tfrecord_files)
            if self.env == 'kitti':
                val_dataset = val_dataset.map(kitti_parse_example)
            else:
                val_dataset = val_dataset.map(carla_parse_example)
            data = val_dataset
        return data

    def get_single_dataset(self, train=True):
        # 读取TFRecord文件
        # 定义Feature结构，告诉编码器每个Feature的类型是什么
        kitti_feature_description = {
            'points': tf.io.FixedLenFeature([], tf.string),
            'this_direction_params': tf.io.FixedLenFeature([], tf.string)
        }

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

            direction_shape = self.horizon_max * 2 + 1
            feature_dict['this_direction_params'] = tf.reshape(feature_dict['this_direction_params'],
                                                               shape=[direction_shape])

            return feature_dict['single_pcd_data'], feature_dict['navg_maps'], feature_dict['this_direction_params']

        if train:
            train_tfrecord_files = [self.tfrecords_train_dir + one_dir for one_dir in
                                    os.listdir(self.tfrecords_train_dir)]
            # shuffle
            random.shuffle(train_tfrecord_files)
            train_dataset = tf.data.TFRecordDataset(train_tfrecord_files)
            train_dataset = train_dataset.map(carla_parse_example)
            # lidar_file = train_dataset.batch(self.batch_size)  # drop_remainder = False
            data = train_dataset
        else:
            val_files = os.listdir(self.tfrecords_val_dir)
            val_files.sort()
            print(val_files)
            val_tfrecord_files = [self.tfrecords_val_dir + one_dir for one_dir in val_files]
            val_dataset = tf.data.TFRecordDataset(val_tfrecord_files)
            val_dataset = val_dataset.map(carla_parse_example)
            data = val_dataset
        return data


if __name__ == '__main__':
    import yaml
    from utils.pointcloud_utils import read_bin, show_trajectorys, fit_traj
    from data_augument import random_flip, dilate_map

    config_file = '../config/carla_config.yaml'
    # config_file = '../config/remote_carla_config.yaml'
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

    dataset = TFRecordDataset(cfg=cfg)
    # dataset = NoramlTFRecordDataset(cfg=cfg)
    dataset.generate_single_sampling_data(is_training=False)
    # dataset.generate_d_sampling_data(is_training=False)
    if (0):
        train_data = dataset.get_dataset(False)
        train_data = train_data.batch(cfg.batch_size, drop_remainder=True)

        for idx, data in enumerate(train_data):
            batch_sz = data[0].shape[0]
            lidar16, lidar32, lidar40, single_pcds, maps, gts = data
            single_pcds = lidar32.numpy()
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
                      data[3][i].numpy().shape,
                      data[4][i].numpy().shape,
                      data[5][i].numpy().shape,
                      )
