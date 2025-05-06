import numpy as np
import cv2, os
import pyrealsense2 as rs
from utils import ARUCO_DICT, aruco_display, get_center
from scipy.spatial.transform import Rotation as R

# Device: Intel RealSense D405, Serial Number: "419122270338"
# Device: Intel RealSense D405, Serial Number: "315122271073"
# Device: Intel RealSense D435I, Serial Number: "109622072337"

def get_rl_pipeline(selected_serial):
    # camera init
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)  # Align depth to color

    get_RGBframe(pipeline)
    return pipeline, align

def get_RGBframe(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color: 
        return None
    else:
        color_np = np.asanyarray(color.get_data())
        color = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        # cv2.imshow("RGB Image", color)
        # cv2.waitKey(1)

        return color

def get_RGBDframe(rs_pl):
    pipeline, align = rs_pl
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)  # Align depth with color

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        # print('return none', color_np.shape, depth_np.shape)
        return None, None
    else:
        # print(depth_np.max(), depth_np.min())
        # depth_np[depth_np > max_distance] = 0  # Set far values to 0
        color_np = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)

        # max_distance = 2000  # Set maximum valid depth to 2000 mm
        depth_np = np.asanyarray(depth_frame.get_data())
        # depth_visual = cv2.convertScaleAbs(depth_np, alpha=0.03)
        # # # print(depth_np)
        # cv2.imshow("Depth Image", depth_visual)
        # cv2.imshow("RGB Image", color)
        # cv2.waitKey(1)

        return color, depth_np

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
    return noisy_translation, noisy_quaternion

def append_state(rs_pl, arm, data):
    if rs_pl is not None:
        # rgb = get_RGBframe(rs_pl)
        rgb, depth = get_RGBDframe(rs_pl)
        data['rgb'].append(rgb)
        data['depth'].append(depth*1)

    ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
    gripper_w = arm.get_gripper_width()
    data['translation'].append(ee_trans)
    data['rotation'].append(ee_quat)
    data['gripper_w'].append(gripper_w)

    return data

def play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripper_list=None, record_trajectory=False, rs_pl=None, use_noise=False, use_start=False):
    target_idx = 0
    done = False
    data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}

    while not done:
        done = False
        ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
        target_idx = find_next_target(ee_trans, ee_quat, trans_list, quat_list, d_threshod, q_threshold, target_idx, use_start=use_start)
        target_trans, targeta_quat = trans_list[target_idx], quat_list[target_idx]
        q_error = 1 - abs(np.dot(ee_quat, targeta_quat)) # quaternion distance
        d_error = np.linalg.norm(ee_trans - target_trans)
        if target_idx == len(trans_list)-1 and d_error < d_threshod and q_error < q_threshold:
            done = True
        
        # target_rpy = R.from_quat(targeta_quat).as_euler('xyz')
        gripper_w = arm.get_gripper_width()
        # print(target_idx, len(trans_list), target_trans, ee_trans, target_rpy*180/np.pi, gripper_w)
        # print(target_idx, len(trans_list), d_error, d_error < d_threshod, q_error, q_error < q_threshold, gripper_w)

        # if use_noise:
        if use_noise and target_idx < len(trans_list)-4:
            target_trans, targeta_quat = add_pose_noise(target_trans, targeta_quat, 0.005, 2.0, type='gaussian')

        # move gripper
        if gripper_list is not None:
            arm.set_gripper_opening(gripper_list[target_idx])
        arm.set_ee_pose(target_trans, targeta_quat)

        joint_vel = arm.robot.current_joint_velocities
        max_joint_vel = np.max(abs(joint_vel))
        if max_joint_vel > 0.3:
            arm.speed_down()
            # print('slower !!!')
        else:
            arm.speed_normal()

        rate.sleep()

        if record_trajectory:
            data = append_state(rs_pl, arm, data)

    arm.robot.join_motion()
    arm.set_ee_pose(trans_list[-1], quat_list[-1], asynchronous=False)

    if record_trajectory:
        data = append_state(rs_pl, arm, data)

    return data

def get_file_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f_list = os.listdir(dir_path)
    file_path = dir_path + '/' + str(len(f_list)) + '.h5'
    print('get_file_path', file_path)

    return file_path

def quaternion_distance_threshold(max_angle_deg):
    # Convert maximum angle to radians
    max_angle_rad = np.radians(max_angle_deg)
    
    # Compute the dot product threshold
    dot_product_threshold = np.cos(max_angle_rad / 2)
    
    # Compute the quaternion distance threshold
    distance_threshold = 1 - dot_product_threshold
    
    return distance_threshold

def find_next_target(current_trans, current_quat, trans_list, quat_list, trans_threshold, quat_threshold, start_idx=0, use_start=False):
    # Compute distances to all trajectory points
    traj_len = len(trans_list)
    distances = np.linalg.norm(trans_list - current_trans, axis=1)
    
    if use_start:
        closest_idx = start_idx
    else:
        # Find the closest point
        start_idx = min(start_idx, traj_len-1)
        closest_idx = np.argmin(distances) # max(0, np.argmin(distances)-1)
        closest_idx = max(start_idx, closest_idx)
        # print('start idx', start_idx)
        for i in range(closest_idx, len(distances), 1):
            dist = distances[i]
            if dist < trans_threshold*1.5:
                closest_idx = i

    # print('closest idx', closest_idx)

    if closest_idx > traj_len - 1:
        closest_idx = traj_len - 1

    # closest_idx = np.argmin(distances)
    closest_quat = quat_list[closest_idx]

    t_error = distances[closest_idx]
    q_error = 1 - abs(np.dot(current_quat, closest_quat)) # quaternion distance

    # print('error', t_error, q_error, trans_threshold, quat_threshold)

    # Determine if the robot is "between" two points
    if closest_idx < traj_len - 1:
        # Vector from closest point to current position
        to_next = trans_list[closest_idx + 1] - trans_list[closest_idx]
        to_robot = current_trans - trans_list[closest_idx]

        # print('closest idx', closest_idx, 'dot product', np.dot(to_next, to_robot), 't_th', trans_threshold, t_error, 'q_th', quat_threshold, q_error)
        # Check if robot has passed the current point
        if np.dot(to_next, to_robot) > 0.0005:
            # Move to the next point
            return closest_idx + 1
        elif t_error < trans_threshold*1 and q_error < quat_threshold*1:
            return closest_idx + 1
    
    # Otherwise, stay on the current point
    # print('closest idx', closest_idx)

    return closest_idx

def get_gripper_state(rgb, detector, gripper_state, marker_left, marker_right):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    
    if len(corners) > 1:
        for corner, id in zip(corners, ids):
            cx, cy = get_center(corner)
            if id == 0:
                marker_left = cx
            if id == 1:
                marker_right = cx

    if marker_left is not None and marker_right is not None:
        pix_diff = abs(marker_right-marker_left)
        if pix_diff < 200:
            gripper_state = 0
        else:
            gripper_state = 1

    return gripper_state, marker_left, marker_right