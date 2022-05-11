import os
import random
import numpy as np
import math

from utils.pointcloud_utils import fit_traj

# mk dir

# root_dir = '/media/swc/swc/study/LidarData/real_world/data/'
root_dir = '/media/wrc/a6a69390-c76b-42fa-8d81-6ef8474db793/home/a/swc/data/real_world/data/'


def exist_make_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def get_yd_rt(x, y, yaw):
    # lcoal_pose from yandan
    # forward:y yaw from y
    yaw = yaw * (math.pi / 180.0)
    rt = np.array([[math.cos(yaw), -math.sin(yaw), 0, y],
                   [math.sin(yaw), math.cos(yaw), 0, -x],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
    return rt


def get_poses(file_path):
    sample_poses = []
    f = open(file_path)
    lines = f.readlines()
    total_frames = 0

    for line in lines:
        total_frames += 1
        line = line.strip('').split(' ')
        local_x = float(line[1]) * 0.01
        local_y = float(line[2]) * 0.01
        local_h = float(line[3]) * 0.01

        rt = get_yd_rt(local_x, local_y, local_h)
        sample_poses.append(rt)
    f.close()
    return sample_poses


def generate_waypoints_txt():
    # get points and trajectory points
    sequences = [0, 1, 2, 3, 4]
    horizon_max = 30
    horizon_res = 0.2
    for sequence in sequences:
        lidar_dir = root_dir + f"{sequence:0>2d}/lidar/"
        pose_path = root_dir + f"poses/{sequence:0>2}.txt"

        waypoints_dir = root_dir + f"{sequence:0>2d}/waypoints/"

        exist_make_dir(waypoints_dir)

        sample_poses = get_poses(pose_path)

        total_lidar_files = os.listdir(lidar_dir)
        total_lidar_files.sort()
        total_frames = len(total_lidar_files)

        for frame in range(total_frames):
            yaw = 0
            this_direction_params = fit_traj(sample_poses[frame:], horizon_max=horizon_max, horizon_res=horizon_res,
                                             yaw=yaw,
                                             vis=False,
                                             is_negative=False)

            if this_direction_params is None:
                continue

            direction_txt = waypoints_dir + f"{frame:0>6d}.txt"
            np.savetxt(direction_txt, X=this_direction_params)
            print(f"save {direction_txt}")


if __name__ == '__main__':
    generate_waypoints_txt()
    # path = '/media/swc/swc/study/LidarData/real_world/data/03/waypoints/000000.txt'
    # a = np.loadtxt(path)
    # print(a.shape)