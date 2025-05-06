import numpy as np
import rospy
import pickle 
import signal
import os, cv2
import argparse
import h5py
from scipy.spatial.transform import Rotation as R
from franka_robot.franka_dual_arm import FrankaLeft, FrankaRight
from franka_data_util import quaternion_distance_threshold, get_file_path, play_trajectory, get_rl_pipeline, get_RGBDframe
# from franka_robot.tora import TORA


def publish_static_tf(broadcaster):
    """Publish a static transform from franka_table to franka_base."""
    broadcaster.sendTransform(
        (-1.0, 0.0, 0.015),  # Translation
        (0.0, 0.0, 0.0, 1.0),  # Quaternion (no rotation)
        rospy.Time.now(),
        "franka_base",  # Child frame
        "/vicon/franka_table/franka_table"  # Parent frame
    )

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    rospy.signal_shutdown("You pressed Ctrl+C!")
    exit(0)

def interpolate_pose(robot_pose, target_pose, ref_point, step_length=0.03):
    ref_x, ref_y, ref_z = ref_point

    # Calculate total distance and number of steps based on step length
    start = np.array(robot_pose[:3])
    end = np.array(target_pose[:3])
    total_distance = np.linalg.norm(end - start)
    steps = int(np.ceil(total_distance / step_length))

    # Linear interpolation for position
    x_vals = np.linspace(robot_pose[0], target_pose[0], steps)
    y_vals = np.linspace(robot_pose[1], target_pose[1], steps)
    z_vals = np.linspace(robot_pose[2], target_pose[2], steps)
    # print(robot_pose, ref_point, target_pose)

    trans_list, quat_list = [], []

    for i in range(steps):
        x, y, z = x_vals[i], y_vals[i], z_vals[i]

        # Calculate direction vector from current position to reference point
        dir_vec = np.array([ref_x - x, ref_y - y, ref_z - z])
        dir_vec /= np.linalg.norm(dir_vec)  # Normalize the direction vector

        # Calculate roll, pitch, yaw to point to the reference
        yaw = np.arctan2(dir_vec[1], dir_vec[0])
        pitch = np.arcsin(-dir_vec[2])  # Negative sign due to coordinate conventions
        roll = 0  # Assuming roll remains constant

        quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        # print(roll, pitch*180/np.pi, yaw*180/np.pi)

        trans_list.append([x,y,z])
        quat_list.append(quat)

    return trans_list, quat_list

def ee_to_base(trans_base, quat_base, trand_in_ee):
    rot_base = R.from_quat(quat_base)  # Base rotation as a Rotation object

    # Compute new translation in the base frame
    trans_base_target = trans_base + rot_base.apply(trand_in_ee)

    # # Compute new rotation in the base frame
    # rot_relative = R.from_euler('xyz', euler_relative, degrees=False)  # Relative rotation from Euler angles
    # quat_base_target = (rot_base * rot_relative).as_quat()  # Combine rotations
    return trans_base_target

def interactive_record(arm, file_root, only_pose=False):
    data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
    while True:
        file_path = get_file_path(file_root)
        a = input('press enter or q to quite, o or c to open or close gripper')

        if a == 'o':
            arm.open_gripper()
        elif a == 'c':
            arm.close_gripper()
        elif a == 'q':
            if not only_pose:
                # with open(file_path, 'wb') as f:
                #     pickle.dump(data, f)
                with h5py.File(file_path, 'w') as hdf5_file:
                    for key, value in data.items():
                        hdf5_file.create_dataset(key, data=value)
                    print('----- save to', file_path, len(data['translation']))

            # arm.open_gripper()
            break
        else:
            ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
            if arm.get_gripper_width() < 0.02:
                gripper_w = 0.0
            else:
                gripper_w = 0.04
            data['translation'].append(ee_trans)
            data['rotation'].append(ee_quat)
            data['gripper_w'].append(gripper_w)

            print('trajectory length', len(data['translation']))
            if only_pose:
                # with open(file_path, 'wb') as f:
                #     pickle.dump(data, f)
                with h5py.File(file_path, 'w') as hdf5_file:
                    for key, value in data.items():
                        hdf5_file.create_dataset(key, data=value)
                    print('------ save to', file_path, len(data['translation']))

                data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}


def get_trajectory(path, need_gripper = True, sample_type='last'):
    # take the last recorded trajectory
    f_list = os.listdir(path)
    f_num = len(f_list)
    if sample_type == 'first':
        f_idx = 0   
    elif sample_type == 'last':
        f_idx = f_num-1
    else:
        f_idx = np.random.randint(1, f_num)
    print('selected file id', sample_type, f_idx)
    file_path = path + '/' + str(f_idx) + '.h5'

    trans_list, quat_list, gripperw_list = [], [], []
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    #     trans_list, quat_list = np.array(data['translation']), np.array(data['rotation'])
    #     if need_gripper:
    #         gripperw_list = np.array(data['gripper_w'])
    with h5py.File(file_path, 'r') as f:
        trans_list, quat_list = np.array(f['translation']), np.array(f['rotation'])
        if need_gripper:
            gripperw_list = np.array(f['gripper_w'])
    return trans_list, quat_list, gripperw_list

def repeat_last(data, num_repeat=10):
    for key in data:
        if len(data[key]) > 0:
            for i in range(num_repeat):
                data[key].append(data[key][-1])
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm_name', default='left_arm', type=str)  # left_arm   right_arm
    parser.add_argument('--item_name', default='xiaomi', type=str)   # xiaomi, lv, ubag, ibag
    parser.add_argument("--use_rgb", action="store_true", help="Use RGB")
    # parser.add_argument("--use_robot", action="store_true", help="Use RGB")
    # parser.add_argument("--use_robot", default=True, type=bool) 
    parser.add_argument("--mode", default='run', type=str) # traj_open, traj_close  # pose_grasp

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("franka_data_replay")
    rate = rospy.Rate(50)  # 10 Hz loop rate

    robot_traj_path = './dataset_hdf5/robot/' + args.arm_name + '/open/' + args.item_name
    support_grasp_path = './dataset_hdf5/support/' + args.arm_name + '/grasp/' + args.item_name
    support_start_path = './dataset_hdf5/support/' + args.arm_name + '/start/' + args.item_name
    support_open_path = './dataset_hdf5/support/' + args.arm_name + '/open/' + args.item_name
    support_close_path = './dataset_hdf5/support/' + args.arm_name + '/close/' + args.item_name

    print('use rgb', args.use_rgb)

    arm, rs_pl = None, None

    # init threshold
    d_threshod = 0.015
    q_threshold = quaternion_distance_threshold(1.5)

    # robot related init
    # if args.use_robot:
    # tora = TORA()
    if 'left' in args.arm_name:
        arm = FrankaLeft()
    else:
        arm = FrankaRight()
    # arm.set_default_pose()
    # arm.open_gripper()
    # arm.close_gripper()

    print('use rgb', args.use_rgb)
    # if args.use_rgb:
    if 'left' in args.arm_name:
        arm = FrankaLeft()
        rs_pl = get_rl_pipeline('315122271073')
    elif 'right' in args.arm_name:
        arm = FrankaRight()
        rs_pl = get_rl_pipeline('419122270338')

    if args.mode == 'traj_open':
        arm.open_gripper()
        arm.set_default_pose()
        start_trans, start_quat, _ = get_trajectory(support_grasp_path, need_gripper=False, sample_type='last')
        start_trans, start_quat = start_trans[0], start_quat[0]
        arm.set_ee_pose(start_trans, start_quat, asynchronous=False)
        input('move zipper')
        arm.close_gripper()        

        interactive_record(arm, support_open_path, only_pose=False)
    elif args.mode == 'traj_close':
        # trans_list, quat_list, gripperw_list = get_trajectory(support_open_path, sample_type='last')
        # arm.set_ee_pose(trans_list[-1], quat_list[-1], asynchronous=False)
        interactive_record(arm, support_close_path, only_pose=False)
    elif args.mode == 'pose_grasp':
        interactive_record(arm, support_grasp_path, only_pose=True)     
    elif args.mode == 'play_open':
        arm.open_gripper()
        arm.set_default_pose()
        start_trans, start_quat, _ = get_trajectory(support_grasp_path, need_gripper=False, sample_type='last')
        start_trans, start_quat = start_trans[0], start_quat[0]
        arm.set_ee_pose(start_trans, start_quat, asynchronous=False)

        # trans_list, quat_list, gripperw_list = get_trajectory(support_open_path, sample_type='first')
        # start_trans[1] = trans_list[0][1]
        # arm.set_ee_pose(start_trans, start_quat, asynchronous=False)

        input('move zipper')
        arm.close_gripper()

        ##########################################
        trans_list, quat_list, gripperw_list = get_trajectory(support_open_path, sample_type='last')
        arm.set_soft(70)
        ##########################################
        print('play data', len(trans_list))
        # # print(trans_list[0], quat_list[0])
        tra_seg = play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripper_list=None, record_trajectory=True)
        file_path = get_file_path(support_open_path)

        print(len(tra_seg['translation']))
        input('press enter to save data')
        with h5py.File(file_path, 'w') as hdf5_file:
            for key, value in tra_seg.items():
                hdf5_file.create_dataset(key, data=value)            
        # tra_seg = play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripper_list = None, record_trajectory=True, rs_pl=rs_pl)
    elif args.mode == 'play_close':
        # arm.open_gripper()
        # arm.set_default_pose()
        # print('at default')

        # # arm.set_ee_pose(trans_list[0], quat_list[0], asynchronous=False)
        # arm.set_ee_pose(trans_list[0]+np.array([-0.02, 0, 0.01]), quat_list[0], asynchronous=False)
        # ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
        # print(trans_list[0]+np.array([-0.02, 0, 0.01]), ee_trans)

        # input('move zipper')
        # arm.set_ee_pose(trans_list[0], quat_list[0], asynchronous=False)
        # arm.close_gripper()

        ##########################################
        trans_list, quat_list, gripperw_list = get_trajectory(support_close_path, sample_type='last')
        arm.set_soft(60)
        ##########################################

        print('play data', len(trans_list))
        # # print(trans_list[0], quat_list[0])
        tra_seg = play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripper_list=None, record_trajectory=True)
        file_path = get_file_path(support_close_path)

        print(len(tra_seg['translation']))
        input('press enter to save data')
        with h5py.File(file_path, 'w') as hdf5_file:
            for key, value in tra_seg.items():
                hdf5_file.create_dataset(key, data=value)            
    elif args.mode == 'play_traj_reverse':
        # arm.set_soft()
        arm.open_gripper()
        # # arm.set_default_pose()

        # # trans_list, quat_list, _ = get_trajectory(support_open_path, sample_type='first')
        # # # print(stop_idx, trans_list.shape, trans_list[stop_idx:].shape)
        # # # trans_list, quat_list = trans_list[33:][::-1], quat_list[33:][::-1]
        # # trans_list, quat_list = trans_list[1:][::-1], quat_list[1:][::-1]

        trans_list, quat_list, _ = get_trajectory(support_open_path, sample_type='last')

        arm.set_ee_pose(trans_list[-1], quat_list[-1], asynchronous=False)
        input('move gripper')
        arm.close_gripper()

        # trans_list, quat_list, _ = get_trajectory(support_open_path, sample_type='last')
        # arm.set_ee_pose(trans_list[-1], quat_list[-1], asynchronous=False)

        trans_list, quat_list, _ = get_trajectory(support_close_path, sample_type='first')
        traj = play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripper_list=None, record_trajectory=True)
        print(len(traj['translation']))
        input('press enter to save data')
        save_path = get_file_path(support_close_path)
        # with open(save_path, 'wb') as f:
        #     pickle.dump(traj, f)
        with h5py.File(save_path, 'w') as hdf5_file:
            for key, value in traj.items():
                hdf5_file.create_dataset(key, data=value)
        arm.open_gripper()

    elif args.mode == 'replay':
        arm.open_gripper()
        arm.set_default_pose()

        d_threshod = 0.005
        q_threshold = quaternion_distance_threshold(0.5)

        print('in replay')
        # sample a random graping pose, and add some noise
        for i in range(10):
            trans_list, quat_list, gripperw_list = get_trajectory(robot_traj_path, need_gripper=True, sample_type='uniform')
            # for trans, quat, grepw in zip(trans_list, quat_list, gripperw_list):

            #     color_img, depth_img = get_RGBDframe(rs_pl)
            #     depth_visual = cv2.convertScaleAbs(depth_img, alpha=0.03)

            #     # cv2.imshow('rgb', color_img)
            #     # cv2.imshow('depth', depth_visual)
            #     # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     #     break

            #     arm.set_ee_pose(trans, quat, asynchronous=True)
            #     arm.set_gripper_opening(grepw)

            play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripperw_list, record_trajectory=True, rs_pl=rs_pl)