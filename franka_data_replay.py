# from juliacall import Main as jl, convert as jlconvert

# jl.seval("using MeshCat")
# jl.seval("using Rotations")
# jl.seval("using TORA")

import numpy as np
import rospy, tf
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
    rospy.init_node("franka_data_generation")
    rate = rospy.Rate(10)  # 10 Hz loop rate

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', default='./paper_hdf5_v4/human', type=str)
    parser.add_argument('--arm', default='left', type=str)    # left, right        
    parser.add_argument('--cam', default='cam_up', type=str)    # up, down        initial grassper position
    parser.add_argument('--zip', default='zip_top', type=str)    # top, bottom     zipper position
    parser.add_argument('--item', default='small_box', type=str)    # Logging directory
    parser.add_argument('--data_mode', default='grasp', type=str) # grasp, open,  grasp_noise, open_noise

    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--mode', default='rgb', type=str) # rgb, depth, trajectory
    args = parser.parse_args()

    traj_path = os.path.join(args.base_path, args.data_mode, args.item, args.zip, args.cam)
    f_list = os.listdir(traj_path)
    f_num = len(f_list)    

    if args.mode == 'rgb' or args.mode == 'depth':
        base_idx = args.idx
        trans_list, quat_list, gripper_list, img_list = get_data_list(traj_path, mode=args.mode, idx=base_idx)
        len_list = len(img_list)
        idx = 0
        print('last trans', trans_list[-1])

        while True:
            frame = img_list[idx]
            print(idx, frame.shape, gripper_list[idx])
            col_shift = 200
            frame = frame[:300, 320-col_shift:320+col_shift, :]  # Crop top 300 rows

            # frame = cv2.resize(frame, (224, 224))
            cv2.imshow('Align Example', frame)
            # print(trans_list[0],quat_list[0])
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
        # arm.set_default_pose()
        # shift = np.array([-0.003, 0.313, 0.0])
        # trans_up, quat_up = np.array([0.41554661, 0.12623907, 0.55202587])+shift, np.array([0.12179858,  0.23606472, -0.4982947, 0.82531264])
        # trans_down, quat_down = np.array([0.46696117, 0.15896487, 0.20866235])+shift, np.array([0.11838443, 0.17352301, -0.56927033, 0.79486237])
        joint_up = np.array([-0.97788733, -1.04903993,  1.31520369, -1.58949637, -0.26875838,  1.36971498, 2.23423306])
        joint_down = np.array([-1.12243027, -1.2869527, 1.72586445, -2.25379698,  0.18903419, 2.15440121, 2.43160574])
        arm.set_joint_pose(joint_up, asynchronous=False)
        # print('set joint up', joint_up)
        # arm.set_ee_pose(trans_up, quat_up, asynchronous=False)
        # arm.set_ee_pose(trans_down, quat_down, asynchronous=False)
        # arm.set_ee_pose_relative(np.array([-0.15, 0.0, 0.0]))
        input("Press Enter to start the trajectory playback...")

        # if args.cam == 'up':
        #     arm.set_default_pose()
        # elif args.cam == 'down':
        #     down_default_trans =[0.24859943,0.45291657,0.19324586]
        #     down_default_rot =  [0.05746551 ,0.08996469,-0.39378176 ,0.91298412]
        #     arm.set_ee_pose(down_default_trans,down_default_rot, asynchronous=False)

        data = get_trajectory(traj_path, idx=args.idx)
        trans_list, quat_list, gripper_list = np.array(data['translation']), np.array(data['rotation']), np.array(data['gripper_w'])

        arm.set_ee_pose(trans_list[0], quat_list[0], asynchronous=False)
        input("Press Enter to start the trajectory playback...")
        # shift = np.array([-0.1, 0, 0])
        shift = np.array([0.0, 0.0, 0.0])
        for trans, rot, grip in zip(trans_list, quat_list, gripper_list):
            arm.set_ee_pose(trans+shift, rot, asynchronous=False)
            # arm.set_gripper_opening(grip)
            rate.sleep()
