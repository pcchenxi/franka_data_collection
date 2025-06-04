import numpy as np
import rospy, time
import signal, h5py
import sys, os
import argparse
import tf, cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
from utils import ARUCO_DICT, aruco_display, get_center
from franka_robot.franka_dual_arm import FrankaLeft, FrankaRight
# BASE_PATH = './dataset_hdf5/human/'
# BASE_PATH = './dataset/human/grasp_init/'
# BASE_PATH = './dataset/human/grasp_noise/'
# BASE_PATH = './dataset/human/full' #全程，不带噪声
BASE_PATH = './dataset/human/open_noise_full' #拉拉链过程中有噪声
DROP_LAST_FRAME = 3
data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
file_root = 'dataset_hdf5/human/junchen_record'
support_path = 'dataset_hdf5/human/junchen_record'
_loop = True

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
    # cv2.imshow("RGB Image", gray)
    # cv2.waitKey(1)

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
            gripper_state = 0.0
        else:
            gripper_state = 0.05

    return gripper_state, marker_left, marker_right

def get_file_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f_list = os.listdir(dir_path)
    file_path = dir_path + '/' + str(len(f_list)) + '.h5'
    print('get_file_path', file_path)

    return file_path

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    # file_path = get_file_path(file_root)
    # with h5py.File(file_path, 'w') as hdf5_file:
    #     for key, value in data.items():
    #         hdf5_file.create_dataset(key, data=value)
    #     print('----- save to', file_path, len(data['translation']))
    sys.exit(0)
    
def save_data(file_path):
    print('Start Saving data')
    file_path = get_file_path(file_path)
    with h5py.File(file_path, 'w') as hdf5_file:
        for key, value in data.items():
            # 删掉最后两帧
            if args.drop_last_frame:
                value = value[:-DROP_LAST_FRAME]
                if args.start_trigger == 'close':
                    value = value[DROP_LAST_FRAME*2:]# if recoding is triggered by close, drop the first two frames to ensure the gripper is closed
                if args.start_trigger == 'key':
                    value = value[1:]# if recoding is triggered by key, drop the first frame
            hdf5_file.create_dataset(key, data=value)
        print('----- save to', file_path, len(data['translation']))
    # clear data
    for key in data.keys():
        data[key] = []
    _loop = False
    print('Data saved to', file_path)
def save_data_or(file_path):
    print('Start Saving data')
    file_path = get_file_path(file_path)
    with h5py.File(file_path, 'w') as hdf5_file:
        for key, value in data.items():
            hdf5_file.create_dataset(key, data=value)
        print('----- save to', file_path, len(data['translation']))
    # clear data
    _loop = False
    print('Data saved to', file_path)
def get_trajectory_by_idx(path, need_gripper = True, idx=-1):
    # take the last recorded trajectory
    f_list = os.listdir(path)
    f_num = len(f_list)
    if idx < 0 or idx >= f_num:
        print('invalid index')
        idx = f_num-1
    print('selected file id',idx)
    file_path = path + '/' + str(idx) + '.h5'

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

if __name__ == '__main__':
    # ros related init
    rospy.init_node("franka_data_collection")
    broadcaster = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    rate = rospy.Rate(10)  # 10 Hz loop rate

    # recording related init
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='left_arm', type=str)    # Logging directory
    parser.add_argument('--item_name', default='xiaomi', type=str)
    parser.add_argument("--use_rgb", default=True, type=bool) 
    parser.add_argument('--mode', default='open_traj', type=str)#open_traj, play_traj, back_to_default
    parser.add_argument('--loop', default=False, type=bool)
    parser.add_argument('--drop_last_frame', default=True, type=bool) # drop the last frame
    parser.add_argument('--start_trigger', default='open', type=str) # open, close
    # if loop==True, will record next trajectory once the previous one is finished(still need to close-and-open the gripper to start recording)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    
# initialize saving path
    human_traj_save_path = os.path.join(BASE_PATH, args.name, args.item_name)
    if not os.path.exists(human_traj_save_path):
        os.makedirs(human_traj_save_path)
    _loop = True
    # robot related init
    if 'left' in args.name:
        arm = FrankaLeft() 
        cam_id = '315122271073'#'419122270338' # 315122271073
        # arm_shift = [0, -0.27, 0]
    else:
        arm = FrankaRight()
        cam_id = '419122270338'
        # arm_shift = [0, -0.27, 0]

    # camera init
    if args.use_rgb:
        pipeline = get_rl_pipeline(cam_id)

        # marked related
        arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT['DICT_4X4_50'])
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # main loop
    marker_left, marker_right = None, None
    gripper_state = 0.04 # open

    # what we need to recored
    # data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
    if args.mode == 'open_traj':
        have_been_closed = False
        start = False
        while not rospy.is_shutdown() and _loop:
            publish_static_tf(broadcaster)
            if args.use_rgb:
                rgb, depth = get_RGBDframe(pipeline)
                if rgb is not None:
                    gripper_state, marker_left, marker_right = get_gripper_state(rgb, detector, gripper_state, marker_left, marker_right)
                    # was not recording and now gripper is open-> change signal
                    if args.start_trigger == "open":
                        if have_been_closed and gripper_state!=0.0:
                            if not start:
                                start = True
                                print('start recording')
                            else:
                                start = False
                                print('stop recording')
                                save_data(human_traj_save_path)
                    elif args.start_trigger == "close" and have_been_closed:
                        if gripper_state==0.0:
                            start = True
                            print('start recording')
                        else:
                            start = False
                            print('stop recording')
                            save_data(human_traj_save_path)
                    elif have_been_closed: # if args.start_trigger == "key":
                        if gripper_state==0.0:
                            if not start:
                                _ = input('press enter to start recording')
                                start = True
                                print('start recording')
                        else:
                            start = False
                            print('stop recording')
                            save_data(human_traj_save_path)
                    have_been_closed = (gripper_state == 0.0) # record last gripper state, true means closed
            try:
                (human_trans, human_rot) = listener.lookupTransform('franka_base', '/vicon/franka_human/franka_human', rospy.Time(0))
            
                human_ee_rpy = Rotation.from_quat(human_rot).as_euler('xyz')
                # print("human", human_trans, human_ee_rpy*180/np.pi)
                if not start:
                    continue
                data['translation'].append(human_trans)
                print(human_rot)
                data['rotation'].append(human_rot)
                data['gripper_w'].append(gripper_state)
                if args.use_rgb:
                    data['rgb'].append(rgb)
                    data['depth'].append(depth*1)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Failed to lookup transform from gripper_r to franka_base.")


            # do not use this two line
            # arm.set_ee_pose(np.array(human_trans)+arm_shift, human_rot)
            # arm.set_gripper_opening(gripper_state)

            rate.sleep()
            
    if args.mode == "play_traj":
        arm.open_gripper()
        arm.set_default_pose()
        idx = int(input('input the index of the trajectory you want to play:'))
        human_trans, human_rot, gripper_state = get_trajectory_by_idx(human_traj_save_path, need_gripper = True, idx=idx)
        # start_trans, start_quat = start_trans[0], start_quat[0]
        
        for i in range(len(human_rot)):
            if args.use_rgb:
                _,_ = get_RGBDframe(pipeline)
            arm.set_ee_pose(np.array(human_trans[i]), human_rot[i], asynchronous=False)
            arm.set_gripper_opening(gripper_state[i])
            print('gripper:', gripper_state[i])
    if args.mode == "back_to_default":# whf added
        arm.open_gripper()
        arm.set_default_pose()
            
        arm.robot.join_motion()
