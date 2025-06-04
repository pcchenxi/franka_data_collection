import numpy as np
import rospy
import pickle 
import signal
import os
import argparse
import h5py
from scipy.spatial.transform import Rotation as R
from franka_robot.franka_dual_arm import FrankaLeft, FrankaRight
from franka_data_util import quaternion_distance_threshold, get_file_path, play_trajectory, get_rl_pipeline, find_next_target


def publish_static_tf(broadcaster):
    """Publish a static transform from franka_table to franka_base."""
    broadcaster.sendTransform(
        (-1.0+0.025, -0.0125, 0.015),  # Translation
        (0.0, 0.0, 0.0, 1.0),  # Quaternion (no rotation)
        rospy.Time.now(),
        "franka_base",  # Child frame
        "/vicon/franka_table/franka_table"  # Parent frame
    )
    broadcaster.sendTransform(
        (0.16, 0.0, 0.0),  # Translation
        (0.0, 0.0, 0.0, 1.0),  # Quaternion (no rotation)
        rospy.Time.now(),
        "franka_human_ee",  # Child frame
        "vicon/franka_human/franka_human"  # Parent frame
    )

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    rospy.signal_shutdown("You pressed Ctrl+C!")
    exit(0)

def interpolate_pose(robot_pose, target_pose, ref_point, step_length=0.01):
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
    file_path = path + '/' + str(f_idx) + '.h5'
    print('selected file id', sample_type, f_idx)
    print(file_path)

    trans_list, quat_list, gripperw_list = [], [], []
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)    
    with h5py.File(file_path, 'r') as f:
        trans_list, quat_list = np.array(f['translation']), np.array(f['rotation'])
        if need_gripper:
            gripperw_list = np.array(f['gripper_w'])
    return trans_list, quat_list, gripperw_list

def fix_pose(path):
    data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
    # take the last recorded trajectory
    f_list = os.listdir(path)
    f_num = len(f_list)
    for i in range(f_num):
        file_path = path + '_mani/' + str(i) + '.h5'
        with h5py.File(file_path, 'r') as f:
            trans_list, quat_list, gripperw_list = np.array(f['translation']), np.array(f['rotation']), np.array(f['gripper_w'])
            for j in range(len(trans_list)):
                trans_list[j][2] -= 0.004
            data['translation'] = trans_list
            data['rotation'] = quat_list
            data['gripper_w'] = gripperw_list

        file_path_s = path + '/' + str(i) + '.h5'
        with h5py.File(file_path_s, 'w') as hdf5_file:
            for key, value in data.items():
                hdf5_file.create_dataset(key, data=value)
        print(i, file_path_s)
    print('done')

def repeat_last(data, num_repeat=10):
    for key in data:
        if len(data[key]) > 0:
            for i in range(num_repeat):
                data[key].append(data[key][-1])
    return data

def play_close(support_close_path):
    trans_list, quat_list, _ = get_trajectory(support_close_path, sample_type='last')
    # stop_idx = np.random.randint(10, 33)
    # print(stop_idx, trans_list.shape, trans_list[stop_idx:].shape)
    # trans_list, quat_list = trans_list[stop_idx:][::-1], quat_list[stop_idx:][::-1]
    play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list)

def ee_to_base(trans_base, quat_base, trand_in_ee):
    rot_base = R.from_quat(quat_base)  # Base rotation as a Rotation object

    # Compute new translation in the base frame
    trans_base_target = trans_base + rot_base.apply(trand_in_ee)

    # # Compute new rotation in the base frame
    # rot_relative = R.from_euler('xyz', euler_relative, degrees=False)  # Relative rotation from Euler angles
    # quat_base_target = (rot_base * rot_relative).as_quat()  # Combine rotations
    return trans_base_target

def add_pose_noise(trans, quat, trans_range=0.05, euler_range=10):
    # Add translation noise
    translation_noise = np.random.uniform(-trans_range, trans_range, size=3)
    noisy_translation = trans + translation_noise

    # Add rotational noise
    euler_noise = np.random.uniform(-euler_range, euler_range, size=3)  # in degrees
    rotation_noise = R.from_euler('xyz', euler_noise, degrees=True).as_quat()

    # Combine original quaternion with rotation noise
    original_rotation = R.from_quat(quat)
    noisy_rotation = original_rotation * R.from_quat(rotation_noise)

    # Convert noisy rotation back to quaternion
    noisy_quaternion = noisy_rotation.as_quat()

    return noisy_translation, noisy_quaternion

def play_trajectory_seg(arm, trans_list, quat_list, gripperw_list=None, seg_length=10):
    d_threshod = 0.01
    q_threshold = quaternion_distance_threshold(1)

    tra_len = len(trans_list)
    start_idx = np.random.randint(seg_length*2)
    trans_start_list, quat_start_list = trans_list[:start_idx+1], quat_list[:start_idx+1]
    # print('start idx', start_idx)
    play_trajectory(arm, rate, d_threshod, q_threshold, 
                            trans_start_list, quat_start_list, gripperw_list,
                            record_trajectory=False, rs_pl=rs_pl)
    
    for i in range(start_idx, tra_len, seg_length):
        # print('rand idx', i)
        # -----------------------------
        arm.set_soft()
        ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
        ee_trans_noise, ee_quat_noise = add_pose_noise(ee_trans, ee_quat, 0.05, 10)        
        arm.set_ee_pose(ee_trans_noise, ee_quat_noise, asynchronous=False)
        arm.set_hard()
        # -----------------------------
        trans_list_seg, quat_list_seg = trans_list[i:i+seg_length], quat_list[i:i+seg_length]
        tra_seg = play_trajectory(arm, rate, d_threshod, q_threshold, 
                                trans_list_seg, quat_list_seg, gripperw_list,
                                record_trajectory=True, rs_pl=rs_pl)
        
        save_path = get_file_path(robot_traj_seg_path)
        with h5py.File(save_path, 'w') as hdf5_file:
            for key, value in tra_seg.items():
                hdf5_file.create_dataset(key, data=value)
        
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("franka_data_replay")
    rate = rospy.Rate(50)  # 10 Hz loop rate

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm_name', default='left_arm', type=str)  # left_arm   right_arm
    parser.add_argument('--item_name', default='xiaomi', type=str)   # xiaomi, lv, ubag, ibag
    parser.add_argument("--use_rgb", default=True, type=bool) 
    parser.add_argument("--use_robot", default=True, type=bool) 
    # parser.add_argument("--mode", default='run', type=str) # run, run_seg

    args = parser.parse_args()
    robot_traj_path = './dataset_hdf5/robot/' + args.arm_name + '/open/' + args.item_name
    robot_traj_seg_path = './dataset_hdf5/robot_seg/' + args.arm_name + '/open/' + args.item_name

    support_grasp_path_ = './dataset_hdf5/support/' + args.arm_name + '/grasp/' + args.item_name #+ '_'
    support_grasp_path = './dataset_hdf5/support/' + args.arm_name + '/grasp/' + args.item_name

    support_open_path = './dataset_hdf5/support/' + args.arm_name + '/open/' + args.item_name
    support_close_path = './dataset_hdf5/support/' + args.arm_name + '/close/' + args.item_name
    support_start_path = './dataset_hdf5/support/' + args.arm_name + '/start/' + args.item_name

    arm, rs_pl = None, None

    # init threshold
    d_threshod = 0.01
    q_threshold = quaternion_distance_threshold(1.0)

    # fix_pose(support_grasp_path)

    # robot related init
    if args.use_robot:
        if 'left' in args.arm_name:
            arm = FrankaLeft()
        else:
            arm = FrankaRight()
        arm.set_default_pose()
        # arm.open_gripper()
        # arm.close_gripper()

    # if args.use_rgb:
    #     rs_pl = get_rl_pipeline()

    arm.open_gripper()
    # arm.set_default_pose()
    # take the last recorded trajectory
    f_list = os.listdir(support_grasp_path_)
    f_num = len(f_list)
    # for i in range(f_num):
    idx = 0

    file_path = support_grasp_path_ + '/' + str(0) + '.h5'
    with h5py.File(file_path, 'r') as f:
        trans_list, quat_list = np.array(f['translation']), np.array(f['rotation'])
    fixed_quat = quat_list[0]

    z_shift, x_shift = 0.0, 0.0
    while True:
        a = input('press n for next pose, press s for save the current pose')

        if a == 'n':
            idx = min(f_num-1, idx+1)
        if a == 'p':
            idx = max(0, idx-1)
        if a == 'u':
            z_shift += 0.002
        if a == 'd':
            z_shift -= 0.002
        if a == 'f':
            x_shift += 0.002
        if a == 'b':
            x_shift -= 0.002                     
        elif a == 's':
            data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
            ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
            if arm.get_gripper_width() < 0.02:
                gripper_w = 0.0
            else:
                gripper_w = 0.04
            data['translation'].append(ee_trans)
            data['rotation'].append(ee_quat)
            data['gripper_w'].append(gripper_w)

            file_path = get_file_path(support_grasp_path)
            with h5py.File(file_path, 'w') as hdf5_file:
                for key, value in data.items():
                    hdf5_file.create_dataset(key, data=value)
                print('------ save to', file_path, len(data['translation']))

            continue
        

        file_path = support_grasp_path_ + '/' + str(idx) + '.h5'
        with h5py.File(file_path, 'r') as f:
            trans_list, quat_list = np.array(f['translation']), np.array(f['rotation'])

        ziper_loc = np.random.uniform(-0.02, 0.02)
        arm.set_ee_pose(trans_list[0]+np.array([x_shift, 0.0, z_shift]), fixed_quat,asynchronous=False)
        print(file_path)
