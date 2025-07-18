import numpy as np
import pyrealsense2 as rs
import os, rospy, signal, argparse
import tf, cv2, collections, h5py
from scipy.spatial.transform import Rotation
from utils import ARUCO_DICT, aruco_display, get_center
# from franka_robot.franka_dual_arm import FrankaLeft, FrankaRight
from scipy.spatial.transform import Rotation as R

DROP_FRAME_NUM = 10
LEFT_CAM_ID = '315122271073'#'419122270338' # 315122271073
RIGHT_CAM_ID = '419122270338'#'419122270338' # 315122271073

def publish_static_tf(broadcaster):
    """Publish a static transform from franka_table to franka_base."""
    broadcaster.sendTransform(
        (-1.0+0.025, 0.0, 0.015),  # Translation
        (0.0, 0.0, 0.0, 1.0),  # Quaternion (no rotation)
        rospy.Time.now(),
        "franka_base",  # Child frame
        "/vicon/franka_table/franka_table"  # Parent frame
    )

def get_rl_pipeline(selected_serial):
    # camera init
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)  # Align depth to color

    # get_RGBframe(pipeline)
    return pipeline, align

def get_RGBframe(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color: 
        return None
    else:
        color_np = np.asanyarray(color.get_data())
        color = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        return color

def get_RGBDframe(rs_pl):
    pipeline, align = rs_pl
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)  # Align depth with color

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None
    else:
        # print(depth_np.max(), depth_np.min())
        # depth_np[depth_np > max_distance] = 0  # Set far values to 0
        color_np = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)

        # max_distance = 2000  # Set maximum valid depth to 2000 mm
        depth_np = np.asanyarray(depth_frame.get_data())

        # visulize camera image
        depth_visual = cv2.convertScaleAbs(depth_np, alpha=0.03)
        cv2.imshow("Depth Image", depth_visual)
        cv2.imshow("RGB Image", color)
        cv2.waitKey(1)

        return color, depth_np

def get_gripper_state(rgb, detector, gripper_state, marker_left, marker_right):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    event = False
    # cv2.imshow("RGB Image", gray)
    # cv2.waitKey(1)
    # print(len(corners))

    if len(corners) > 1:
        for corner, id in zip(corners, ids):
            cx, cy = get_center(corner)
            if id == 0:
                marker_left = cx
            if id == 1:
                marker_right = cx
    elif len(corners) == 0:
        event = True

    if marker_left is not None and marker_right is not None:
        pix_diff = abs(marker_right-marker_left)
        if pix_diff < 200:
            gripper_state = 0.0
        else:
            gripper_state = 0.05

    return gripper_state, marker_left, marker_right, event

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    rospy.signal_shutdown("You pressed Ctrl+C!")

def get_file_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f_list = os.listdir(dir_path)
    file_path = dir_path + '/' + str(len(f_list)) + '.h5'
    print('get_file_path', file_path)

    return file_path

def init_devices(camera_name):
    if 'left' in camera_name:
        # arm = FrankaLeft() 
        cam_id = LEFT_CAM_ID
    else:
        # arm = FrankaRight()
        cam_id = RIGHT_CAM_ID

    # camera init
    pipeline = get_rl_pipeline(cam_id)

    # marked related
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT['DICT_4X4_50'])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)    
    return pipeline, detector
    
def save_data(file_path):
    file_path = get_file_path(file_path)
    with h5py.File(file_path, 'w') as hdf5_file:
        for key, value in data.items():
            value = value[DROP_FRAME_NUM:-DROP_FRAME_NUM]
            hdf5_file.create_dataset(key, data=value)
            print(key, len(value))
        print('----- save to', file_path, len(data['translation']))


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

def add_text(img, zip_pos):
    # Text settings
    text = zip_pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 0, 0)  # White

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate bottom-right position
    x = img.shape[1] - text_width - 10  # 10 px from right
    y = img.shape[0] - 10  # 10 px from bottom

    # Put text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Image with Bottom-Right Text", img)
    cv2.waitKey(1)

def get_euler_difference(quat1, quat2):
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)

    # Compute relative rotation: r_rel applied to quat1 gives quat2
    r_rel = r2 * r1.inv()

    angle_rad = r_rel.magnitude()
    angle_deg = np.degrees(angle_rad)

    return angle_deg 


if __name__ == '__main__':
    # ros and system related init
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("franka_data_recording")
    broadcaster = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    rate = rospy.Rate(10)  # 10 Hz loop rate

    # recording related init
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', default='left', type=str)    # left, right        
    parser.add_argument('--cam', default='cam_up', type=str)    # up, down        initial grassper position
    parser.add_argument('--zip', default='zip_top', type=str)    # top, bottom     zipper position
    parser.add_argument('--item', default='small_box', type=str)    # Logging directory
    parser.add_argument('--data_mode', default='grasp', type=str) # grasp, open,  grasp_noise, open_noise

    parser.add_argument('--base_path', default='./paper_hdf5_v4/human', type=str)    # Logging directory
    args = parser.parse_args()

    human_traj_save_path = os.path.join(args.base_path, args.data_mode, args.item, args.zip, args.cam)
    if not os.path.exists(human_traj_save_path):
        os.makedirs(human_traj_save_path)    

    pipeline, detector = init_devices(args.arm)
    # reference_path = '/home/xi/xi_space/franka_manipulation/franka_data_clooection/paper_hdf5_v3/human/grasp/small_box/zip_top/cam_up'
    # _, _, _, img_list = get_data_list(reference_path, mode='rgb', idx=0)
    # reference_img = img_list[0]

    # main loop
    data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
    event_list = collections.deque(maxlen=5)
    marker_left, marker_right = None, None
    gripper_state = 0.04 # open
    start, event_tirgger, ready = False, False, False

    g_count = 0
    pre_trans, pre_quat = None, None

    while not rospy.is_shutdown():
        publish_static_tf(broadcaster)

        rgb, depth = get_RGBDframe(pipeline)
        if rgb is None:
            continue

        gripper_state, marker_left, marker_right, event = get_gripper_state(rgb, detector, gripper_state, marker_left, marker_right)
        try:
            (human_trans, human_rot) = listener.lookupTransform('franka_base', '/vicon/franka_human/franka_human', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Failed to lookup transform from gripper to franka_base.")
            continue

        event_list.append(event)
        event_tirgger = len(event_list) == 5 and all(event_list)  # check if all events are True in the last 5 frames
        if event_tirgger:
            ready = True

        if ready and not event_tirgger and not start: # trigger recording
            start = True
            ready = False
            print('start recording')

            if g_count % 2 == 0:
                zip_pos = 'zip_bottom'
            else:
                zip_pos = 'zip_top'
            human_traj_save_path = os.path.join(args.base_path, args.data_mode, args.item, zip_pos, args.cam)
            if not os.path.exists(human_traj_save_path):
                os.makedirs(human_traj_save_path)

            add_text(rgb, zip_pos)

            print('----------- current zip pose', zip_pos)
        elif event_tirgger and start: # trigger skip frame
            print('skip frame', len(data['gripper_w']))
            continue
        elif ready and not event_tirgger and start: # trigger stop recording
            start = False
            ready = False
            print('stop recording', len(data['gripper_w']))

            save_data(human_traj_save_path)
            data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
            g_count += 1
            pre_trans, pre_quat = None, None

        if start:
            if pre_trans is None:
                pre_trans, pre_quat = human_trans, human_rot

            trans_diff_pre = np.linalg.norm(np.array(human_trans) - np.array(pre_trans))
            euler_diff_pre = get_euler_difference(human_rot, pre_quat)

            if trans_diff_pre > 0.05 or euler_diff_pre > 15.0:
                print('!!!!!!!!!!!!!!!!!!! skip frame jump', trans_diff_pre, euler_diff_pre)
                continue

            data['translation'].append(human_trans)
            data['rotation'].append(human_rot)
            data['gripper_w'].append(gripper_state)
            data['rgb'].append(rgb)
            data['depth'].append(depth*1)

            pre_trans, pre_quat = human_trans, human_rot


        rate.sleep()


# cam_top
# [0.36638517 0.30260678 0.57729757]
# degree: [ 0.03939009  0.30290007 -1.27132275] [  2.25688617  17.35489556 -72.84142787]
# quat [ 0.10522243  0.10982125 -0.58919091  0.79355   ]
# Joints:  [-0.97788733, -1.04903993,  1.31520369, -1.58949637, -0.26875838,  1.36971498, 2.23423306]
# Elbow:  ElbowState(joint_3_pos=1.3152, joint_4_flip=FlipDirection.Negative)