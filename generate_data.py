import numpy as np
import rospy, time
import pickle 
import signal
import os
import argparse
import h5py
from scipy.spatial.transform import Rotation as R
from franka_robot.franka_dual_arm import FrankaLeft, FrankaRight
from franka_data_util import quaternion_distance_threshold, get_file_path, play_trajectory, get_rl_pipeline, find_next_target, append_state


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

def play_close(arm, rate, d_threshod, q_threshold, support_close_path):
    trans_list, quat_list, gripperw_list = get_trajectory(support_close_path, sample_type='last')

    # stop_idx = np.random.randint(10, 33)
    # print(stop_idx, trans_list.shape, trans_list[stop_idx:].shape)
    # trans_list, quat_list = trans_list[stop_idx:][::-1], quat_list[stop_idx:][::-1]
    play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripperw_list)

def ee_to_base(trans_base, quat_base, trand_in_ee):
    rot_base = R.from_quat(quat_base)  # Base rotation as a Rotation object

    # Compute new translation in the base frame
    trans_base_target = trans_base + rot_base.apply(trand_in_ee)

    # # Compute new rotation in the base frame
    # rot_relative = R.from_euler('xyz', euler_relative, degrees=False)  # Relative rotation from Euler angles
    # quat_base_target = (rot_base * rot_relative).as_quat()  # Combine rotations
    return trans_base_target

def add_pose_noise(trans, quat, trans_range=0.05, euler_range=10, type='uniform'):
    # Add translation noise
    if type == 'uniform':
        translation_noise = np.random.uniform(-trans_range, trans_range, size=3)
        noisy_translation = trans + translation_noise
        # Add rotational noise
        euler_noise = np.random.uniform(-euler_range, euler_range, size=3)  # in degrees
        rotation_noise = R.from_euler('xyz', euler_noise, degrees=True).as_quat()
    elif type == 'gaussian':
        translation_noise = np.random.normal(0, trans_range/3, size=3).clip(-trans_range, trans_range)
        noisy_translation = trans + translation_noise
        euler_noise = np.random.normal(0, euler_range/3, size=3).clip(-euler_range, euler_range)  # in degrees
        rotation_noise = R.from_euler('xyz', euler_noise, degrees=True).as_quat()

    # Combine original quaternion with rotation noise
    original_rotation = R.from_quat(quat)
    noisy_rotation = original_rotation * R.from_quat(rotation_noise)

    # Convert noisy rotation back to quaternion
    noisy_quaternion = noisy_rotation.as_quat()

    # if noisy_translation[2] > 0.75:
    #     noisy_translation[2] = 0.75
    # if noisy_translation[2] < 0.6:
    #     noisy_translation[2] = 0.6        
    return noisy_translation, noisy_quaternion

def init_grasp(arm, support_grasp_path):
    arm.open_gripper()
    arm.set_default_pose()

    # sample a random graping pose, and add some noise
    grasp_trans, grasp_quat, _ = get_trajectory(support_grasp_path, need_gripper=False, sample_type='uniform')
    grasp_trans, grasp_quat = grasp_trans[0], grasp_quat[0]
    grasp_trans += np.array([0.0, 0.0, -0.003])
    # arm.set_ee_pose(grasp_trans, grasp_quat, asynchronous=False)
    # pre_grasp_trans = ee_to_base(grasp_trans, grasp_quat, np.array([-0.02, 0, 0.0]))
    pre_grasp_trans = ee_to_base(grasp_trans, grasp_quat, np.array([-0.02, 0.0, 0.0]))
    arm.set_ee_pose(pre_grasp_trans, grasp_quat, asynchronous=False)
    input('move the zipper')

    # input('press enter when ready')
    arm.set_ee_pose(grasp_trans, grasp_quat, asynchronous=False)
    # -----------------------------
    arm.set_soft(60)
    # -----------------------------
    post_grasp_trans = ee_to_base(grasp_trans, grasp_quat, np.array([0.015, 0.0, 0.0]))
    arm.set_ee_pose(post_grasp_trans, grasp_quat, asynchronous=False)
    grasp_trans, grasp_quat, _ = arm.get_ee_pose()
    # grasp_trans = grasp_trans + np.array([0.005, 0.0, 0.0])
    # -----------------------------
    arm.set_hard()
    # -----------------------------
    arm.close_gripper()
    return grasp_trans, grasp_quat

def move_to_new_zip_pose(arm, support_grasp_path):
    grasp_trans, grasp_quat, _ = get_trajectory(support_grasp_path, need_gripper=False, sample_type='uniform')
    grasp_trans, grasp_quat = grasp_trans[0], grasp_quat[0]
    grasp_trans += np.array([0.0, 0.0, -0.003])

    # sample a random y pose, and replace it
    ziper_loc = np.random.uniform(-0.02, 0.02)
    grasp_trans[1] = grasp_trans[1] + ziper_loc
    # ee_trans[1] = grasp_trans[1]

    # pre_grasp_trans = grasp_trans + np.array([-0.01, 0.0, -0.01])
    # arm.set_ee_pose(pre_grasp_trans, grasp_quat, asynchronous=False)
    arm.set_ee_pose(grasp_trans, grasp_quat, asynchronous=False)

    arm.open_gripper(0.04)
    pre_grasp_trans = ee_to_base(grasp_trans, grasp_quat, np.array([-0.02, 0.0, 0.0]))
    arm.set_ee_pose(pre_grasp_trans, grasp_quat, asynchronous=False)
    arm.set_ee_pose(grasp_trans, grasp_quat, asynchronous=False)
    # -----------------------------
    arm.set_soft(60)
    # -----------------------------
    post_grasp_trans = ee_to_base(grasp_trans, grasp_quat, np.array([0.015, 0.0, 0.0]))
    arm.set_ee_pose(post_grasp_trans, grasp_quat, asynchronous=False)
    # -----------------------------
    arm.set_hard()
    # -----------------------------
    arm.close_gripper()
    grasp_trans, grasp_quat, _ = arm.get_ee_pose()
    # grasp_trans = grasp_trans + np.array([0.005, 0.0, 0.0])
    
    arm.open_gripper()
    arm.set_ee_pose_relative(np.array([-0.05, 0.0, 0.0]))

    return grasp_trans, grasp_quat

def get_interpolate_trajectory(current_trans, grasp_trans, grasp_quat):
    pregrasp_trans = ee_to_base(grasp_trans, grasp_quat, np.array([-0.05, 0.0, 0.0]))
    ref_trans = ee_to_base(grasp_trans, grasp_quat, np.array([0.0, 0.0, 0.0]))
    # ref_trans = np.array(grasp_trans + [0.0, 0, 0.01]) # where the ee pointing to while moving

    trans_list, quat_list = interpolate_pose(current_trans, pregrasp_trans, ref_trans, step_length=0.01)

    trans_list.append(pregrasp_trans)
    quat_list.append(grasp_quat)    
    trans_list.append(grasp_trans)
    quat_list.append(grasp_quat)

    return trans_list, quat_list

def run_grasp(arm, rs_pl, grasp_trans, grasp_quat, use_full, use_noise, d_threshod, q_threshold):
    if use_full:
        print('--- use default start pose') 
        arm.set_default_pose()
        ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
        trans_list, quat_list = get_interpolate_trajectory(ee_trans, grasp_trans, grasp_quat)           
        use_start = False
    else:
        print('--- move to noise start pose') 
        start_trans = ee_to_base(grasp_trans, grasp_quat, np.array([-0.1, 0.0, 0.0]))
        start_trans_noise, start_quat_noise = add_pose_noise(start_trans, grasp_quat, 0.07, 10.0, type='uniform')
        arm.set_soft(60)
        arm.set_ee_pose(start_trans_noise, start_quat_noise, asynchronous=False)
        print('    at noisy start pose', )
        arm.set_hard()
    
        trans_list, quat_list = [], []
        pregrasp_trans = ee_to_base(grasp_trans, grasp_quat, np.array([-0.05, 0.0, 0.0]))
        trans_list.append(pregrasp_trans)
        quat_list.append(grasp_quat)    
        trans_list.append(grasp_trans + np.array([0.0, 0.0, 0.0]))
        quat_list.append(grasp_quat)
        use_start = True 

    trans_list = np.array(trans_list)
    quat_list = np.array(quat_list)
    tra_seg_grasp = play_trajectory(arm, rate, d_threshod, q_threshold, 
                            trans_list, quat_list, 
                            record_trajectory=True, rs_pl=rs_pl, use_noise=use_noise, use_start=use_start)
    arm.close_gripper()
    append_state(rs_pl, arm, tra_seg_grasp)
    return tra_seg_grasp

def run_open(arm, rs_pl, use_full, use_noise, d_threshod, q_threshold):
    if use_full:
        print('-----open full')
        trans_list, quat_list, gripperw_list = get_trajectory(support_open_path, sample_type='last')
    else:
        print("-----open from noisy pose")
        ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
        trans_list, quat_list, gripperw_list = get_trajectory(support_open_path, sample_type='last')
        target_idx = find_next_target(ee_trans, ee_quat, trans_list, quat_list, d_threshod, q_threshold)
        print('   target_idx', target_idx, len(trans_list))

        trans_list, quat_list = trans_list[target_idx:], quat_list[target_idx:]
        len_traj = len(trans_list)-5
        random_idx = np.random.randint(len_traj) + 2
        print('   rand_idx', random_idx, trans_list[random_idx:].shape)
        play_trajectory(arm, rate, d_threshod, q_threshold,
                                trans_list[:random_idx], quat_list[:random_idx], gripperw_list[:random_idx])

        ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
        ee_trans_noise, ee_quat_noise = add_pose_noise(ee_trans, ee_quat, 0.03, 15, type='uniform')
        arm.set_soft(50)
        arm.set_ee_pose(ee_trans_noise, ee_quat_noise, asynchronous=False)
        arm.set_hard()
        print('   at moisy pose', random_idx, len(trans_list[random_idx:]))

    tra_open = play_trajectory(arm, rate, d_threshod, q_threshold, 
                            trans_list, quat_list, gripperw_list, 
                            record_trajectory=True, rs_pl=rs_pl, use_noise=use_noise)
    return tra_open

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("franka_data_replay")
    rate = rospy.Rate(20)  # 10 Hz loop rate

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm_name', default='left_arm', type=str)  # left_arm   right_arm
    parser.add_argument('--item_name', default='xiaomi', type=str)   # xiaomi, lv, ubag, ibag
    parser.add_argument("--use_rgb", default=True, type=bool) 
    parser.add_argument("--use_robot", default=True, type=bool) 
    parser.add_argument("--mode", default='run_full', type=str) # run, run_seg

    args = parser.parse_args()
    robot_traj_path = './dataset_hdf5/robot/' + args.arm_name + '/open/' + args.item_name
    robot_traj_seg_path = './dataset_hdf5/robot_seg/' + args.arm_name + '/open/' + args.item_name
    robot_traj_grasp_path = './dataset_hdf5/robot/grasp/' + args.arm_name + '/open/' + args.item_name
    robot_traj_open_path = './dataset_hdf5/robot/open/' + args.arm_name + '/open/' + args.item_name
    robot_traj_full_path = './dataset_hdf5/robot/full/' + args.arm_name + '/open/' + args.item_name

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
            rs_pl = get_rl_pipeline('315122271073')
        elif 'right' in args.arm_name:
            arm = FrankaRight()
            rs_pl = get_rl_pipeline('419122270338')
        # arm.set_default_pose()
        # arm.open_gripper()
        # arm.close_gripper()

    if args.mode == 'run_full':
        d_threshod = 0.01
        q_threshold = quaternion_distance_threshold(1.0)

        grasp_trans, grasp_quat = init_grasp(arm, support_grasp_path)
        arm.open_gripper()
        arm.set_default_pose()
        init_ee_trans, init_ee_quat, _ = arm.get_ee_pose()
        for i in range(10):      
            print('------------------------', i, '-----------------------------')
            use_noise = False
            use_full = True

            print('use nosie', use_noise, 'use full', use_full)
            arm.open_gripper()
            tra_seg_grasp = run_grasp(arm, rs_pl, grasp_trans, grasp_quat, use_full, use_noise, d_threshod, q_threshold)

            save_path = get_file_path(robot_traj_grasp_path)
            with h5py.File(save_path, 'w') as hdf5_file:
                for key, value in tra_seg_grasp.items():
                    hdf5_file.create_dataset(key, data=value)

            # --------------------------------------------------------------------------------------------
            # move to open
            tra_open = run_open(arm, rs_pl, use_full, use_noise, d_threshod, q_threshold)
            repeat_last(tra_open, num_repeat=5)

            save_path = get_file_path(robot_traj_open_path)
            with h5py.File(save_path, 'w') as hdf5_file:
                for key, value in tra_open.items():
                    hdf5_file.create_dataset(key, data=value)

            # tra_full = {key: tra_seg_grasp[key] + tra_open[key] for key in tra_seg_grasp}
            # repeat_last(tra_full, num_repeat=5)

            # for keys in tra_full:
            #     print(keys, len(tra_full[keys]))

            # save_path = get_file_path(robot_traj_full_path)
            # with h5py.File(save_path, 'w') as hdf5_file:
            #     for key, value in tra_full.items():
            #         hdf5_file.create_dataset(key, data=value)

            # arm.close_gripper()
            play_close(arm, rate, d_threshod, q_threshold, support_close_path)
            grasp_trans, grasp_quat = move_to_new_zip_pose(arm, support_grasp_path)

    elif args.mode == 'run_open':
        d_threshod = 0.01
        q_threshold = quaternion_distance_threshold(1.0)

        grasp_trans, grasp_quat = init_grasp(arm, support_grasp_path)
        arm.open_gripper()
        arm.set_default_pose()
        init_ee_trans, init_ee_quat, _ = arm.get_ee_pose()
        for i in range(10):          
            print('------------------------', i, '-----------------------------')
            arm.open_gripper()         
            if i %5 == 0:
                print('--- use default start pose') 
                arm.set_default_pose()
                ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
                trans_list, quat_list = get_interpolate_trajectory(ee_trans, grasp_trans, grasp_quat)
                # trans_list.append(grasp_trans + np.array([0.0, 0.0, -0.005]))
                # quat_list.append(grasp_quat)
                trans_list = np.array(trans_list)
                quat_list = np.array(quat_list)
                play_trajectory(arm, rate, d_threshod, q_threshold, 
                                        trans_list, quat_list, 
                                        record_trajectory=False, rs_pl=rs_pl)

            arm.set_ee_pose(grasp_trans + np.array([0.0, 0.0, -0.002]), grasp_quat, asynchronous=False)
            arm.close_gripper()

            # --------------------------------------------------------------------------------------------
            # move to open
            use_noise = True
            use_full = False
            print('use nosie', use_noise, 'use full', use_full)
            tra_open = run_open(arm, rs_pl, use_full, use_noise, d_threshod, q_threshold)

            save_path = get_file_path(robot_traj_open_path)
            with h5py.File(save_path, 'w') as hdf5_file:
                for key, value in tra_open.items():
                    hdf5_file.create_dataset(key, data=value)

            # arm.close_gripper()
            play_close(arm, rate, d_threshod, q_threshold, support_close_path)
            grasp_trans, grasp_quat = move_to_new_zip_pose(arm, support_grasp_path)

    elif args.mode == 'run_grasp':
        d_threshod = 0.01
        q_threshold = quaternion_distance_threshold(1.0)

        grasp_trans, grasp_quat = init_grasp(arm, support_grasp_path)
        arm.open_gripper()
        arm.set_default_pose()
        init_ee_trans, init_ee_quat, _ = arm.get_ee_pose()
        for i in range(10):            
            print('------------------------', i, '-----------------------------')
            use_noise = True
            use_full = False
            print('use nosie', use_noise, 'use full', use_full)
            arm.open_gripper()
            tra_seg_grasp = run_grasp(arm, rs_pl, grasp_trans, grasp_quat, use_full, use_noise, d_threshod, q_threshold)
            for keys in tra_seg_grasp:
                print(keys, len(tra_seg_grasp[keys]))
            # print('last gripper w:', tra_seg_grasp['gripper_w'][-1])

            # chart = input('press enter to save data, q to discard the data')
            # if chart != 'q':
            save_path = get_file_path(robot_traj_grasp_path)
            with h5py.File(save_path, 'w') as hdf5_file:
                for key, value in tra_seg_grasp.items():
                    hdf5_file.create_dataset(key, data=value)

            ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
            trans_list, quat_list, gripperw_list = get_trajectory(support_open_path, sample_type='last')
            start_idx = find_next_target(ee_trans, ee_quat, trans_list, quat_list, d_threshod, q_threshold)
            end_idx = start_idx + 2
            play_trajectory(arm, rate, d_threshod, q_threshold,
                                    trans_list[start_idx:end_idx], quat_list[start_idx:end_idx], gripperw_list[start_idx:end_idx])
            grasp_trans, grasp_quat = move_to_new_zip_pose(arm, support_grasp_path)

    arm.robot.join_motion()