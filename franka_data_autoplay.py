import numpy as np
import rospy
import pickle 
import signal
import os
import argparse
import tf
import cv2
import h5py
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
# from franka_data_util import quaternion_distance_threshold, get_file_path, play_trajectory, get_rl_pipeline

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    rospy.signal_shutdown("You pressed Ctrl+C!")
    exit(0)

def load_trajectory(path):
    # take the last recorded trajectory
    f_list = os.listdir(path)
    file_path = path + '/' + str(len(f_list)-1) + '.h5'
    print('load', file_path)
    trans_list, quat_list = [], []
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    data = h5py.File(file_path, 'r')
    return data

# def get_trajectory(path, sample_type='last', return_raw=False):
def get_trajectory(path, idx=0, return_raw=False):
    # take the last recorded trajectory
    f_list = os.listdir(path)
    f_num = len(f_list)
    # if sample_type == 'first':
    #     f_idx = 0   
    # elif sample_type == 'last':
    #     f_idx = f_num-1
    # else:
    #     f_idx = np.random.randint(1, f_num)
    if idx == -2:
        f_idx = np.random.randint(1, f_num)
    elif idx == -1:
        f_idx = f_num-1
    else:
        f_idx = min(idx, f_num-1)
    print('selected file id', f_idx, f_num-1)
    file_path = path + '/' + str(f_idx) + '.h5'

    trans_list, quat_list = [], []
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    #     trans_list, quat_list = np.array(data['translation']), np.array(data['rotation'])

    data = h5py.File(file_path, 'r')
    if return_raw:
        return data
    else:
        trans_list, quat_list = np.array(f['translation']), np.array(f['rotation'])
        return trans_list, quat_list

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("franka_data_generation")
    rate = rospy.Rate(10)  # 10 Hz loop rate

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm_name', default='left_arm', type=str)  # left_arm   right_arm
    parser.add_argument('--item_name', default='xiaomi', type=str)   # xiaomi, lv, ubag, ibag
    # parser.add_argument('--robot_traj_path', default='./dataset/robot/left_arm/open/xiaomi', type=str)    # Logging directory
    parser.add_argument("--use_rgb", default=True, type=bool) 
    parser.add_argument("--use_robot", default=False, type=bool) 
    parser.add_argument("--move_robot", default=False, type=bool) 
    parser.add_argument("--mode", default='rgb', type=str) 
    parser.add_argument("--idx", default=-1, type=int) 
    parser.add_argument("--type", default='grasp', type=str) 

    args = parser.parse_args()
    robot_traj_path = './dataset_hdf5/robot/' + args.type + '/' + args.arm_name + '/open/' + args.item_name
    support_start_path = './dataset_hdf5/support/' + args.arm_name + '/start/' + args.item_name
    support_open_path = './dataset_hdf5/support/' + args.arm_name + '/open/' + args.item_name

    # # init threshold
    # d_threshod = 0.01
    # q_threshold = quaternion_distance_threshold(1)

    if args.mode == 'rgb':
        # arm.set_default_pose()
        # data = get_trajectory(robot_traj_path, sample_type='uniform', return_raw=True)
        data = get_trajectory(robot_traj_path, idx=args.idx, return_raw=True)

        for keys in data:
            print(keys, len(data[keys]))

        trans_list, quat_list, gripper_list = np.array(data['translation']), np.array(data['rotation']), np.array(data['gripper_w'])
        rgb_list = np.array(data['rgb'])
        len_list = len(rgb_list)
        idx = 0
        # for frame in rgb_list:
        while True:
            frame = rgb_list[idx]
            frame = cv2.resize(frame, (224, 224))
            cv2.imshow('Align Example', frame)
            key = cv2.waitKey(0)
            # print(key)
            if key == 81:  # Enter key
                idx = max(0, idx-1)
            elif key == 83:
                idx = min(len_list-1, idx+1)
            elif key == 27:  # Escape key
                break  # Exit the program
        print(rgb_list.shape)
    if args.mode == 'depth':
        # arm.set_default_pose()
        data = get_trajectory(robot_traj_path, sample_type='uniform', return_raw=True)
        for keys in data:
            print(keys, len(data[keys]))

        depth_list = np.array(data['depth'])
        len_list = len(depth_list)
        idx = 0
        # for frame in rgb_list:
        while True:
            frame = depth_list[idx]
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.convertScaleAbs(frame, alpha=0.03)

            # max_distance = 2000  # Set maximum valid depth to 2000 mm
            # frame[frame > max_distance] = 0  # Set far values to 0

            cv2.imshow('Align Example', frame)
            key = cv2.waitKey(0)
            # print(key)
            if key == 81:  # Enter key
                idx = max(0, idx-1)
            elif key == 83:
                idx = min(len_list-1, idx+1)
            elif key == 27:  # Escape key
                break  # Exit the program
        print(depth_list.shape)        
