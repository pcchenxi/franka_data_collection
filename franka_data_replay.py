import numpy as np
# import rospy, tf
import signal
import os
import argparse
import cv2
import h5py
from franka_robot.franka_dual_arm import FrankaLeft, FrankaRight

# from scipy.spatial.transform import Rotation

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    # rospy.signal_shutdown("You pressed Ctrl+C!")
    exit(0)

def get_trajectory(path, idx=0):
    # take the last recorded trajectory
    f_list = os.listdir(path)
    f_num = len(f_list)
    f_idx = min(idx, f_num-1)
    print('selected file id', f_idx, f_num-1)
    file_path = path + '/' + str(f_idx) + '.h5'

    data = h5py.File(file_path, 'r')
    return data

def get_data_list(traj_path, mode, idx):
    data = get_trajectory(traj_path, idx=idx)
    for keys in data:
        print(keys, len(data[keys]))

    trans_list, quat_list, gripper_list = np.array(data['translation']), np.array(data['rotation']), np.array(data['gripper_w'])
    if mode == 'rgb':
        img_list = np.array(data['rgb'])
    elif mode == 'depth':
        img_list = np.array(data['depth'])

    return trans_list, quat_list, gripper_list, img_list

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    # rospy.init_node("franka_data_generation")
    # rate = rospy.Rate(10)  # 10 Hz loop rate

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', default='./paper_hdf5/human', type=str)
    parser.add_argument('--arm', default='left', type=str)
    parser.add_argument('--cam', default='left', type=str)
    parser.add_argument('--zip', default='left', type=str)
    parser.add_argument('--item', default='small_bag', type=str)
    parser.add_argument('--idx', default=0, type=int)

    parser.add_argument('--mode', default='rgb', type=str) # rgb, depth, trajectory
    args = parser.parse_args()

    traj_path = os.path.join(args.base_path, args.zip, args.item)
    f_list = os.listdir(traj_path)
    f_num = len(f_list)    

    if args.mode == 'rgb' or args.mode == 'depth':
        base_idx = args.idx
        trans_list, quat_list, gripper_list, img_list = get_data_list(traj_path, mode=args.mode, idx=base_idx)
        len_list = len(img_list)
        idx = 0

        while True:
            frame = img_list[idx]
            # frame = cv2.resize(frame, (224, 224))
            cv2.imshow('Align Example', frame)
            key = cv2.waitKey(0)
            # print(key)
            if key == 110: # next trajectory
                base_idx = min(f_num-1, base_idx + 1)
                print('\n -- idx', base_idx)
                trans_list, quat_list, gripper_list, img_list = get_data_list(traj_path, mode=args.mode, idx=base_idx)
                len_list = len(img_list)
                idx = 0
            elif key == 112: # previous trajectory
                base_idx = max(0, base_idx - 1)
                print('\n -- idx', base_idx)
                trans_list, quat_list, gripper_list, img_list = get_data_list(traj_path, mode=args.mode, idx=base_idx)
                len_list = len(img_list)
                idx = 0
            elif key == 81:  # arrow left
                idx = max(0, idx-1)
            elif key == 83:  # arrow right
                idx = min(len_list-1, idx+1)
            elif key == 27:  # Escape key
                break  # Exit the program
        print(img_list.shape)

    elif args.mode == 'trajectory':
        if 'left' in args.arm:
            arm = FrankaLeft() 
        else:
            arm = FrankaRight()

        arm.open_gripper()
        arm.set_default_pose()

        data = get_trajectory(traj_path, idx=args.idx)
        trans_list, quat_list, gripper_list = np.array(data['translation']), np.array(data['rotation']), np.array(data['gripper_w'])

        for trans, rot, grip in zip(trans_list, quat_list, gripper_list):
            arm.set_ee_pose(trans, rot, asynchronous=False)
            arm.set_gripper_opening(grip)
