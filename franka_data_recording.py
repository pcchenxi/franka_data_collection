import numpy as np
import pyrealsense2 as rs
import os, rospy, signal, argparse
import tf, cv2, collections, h5py
from scipy.spatial.transform import Rotation
from utils import ARUCO_DICT, aruco_display, get_center
# from franka_robot.franka_dual_arm import FrankaLeft, FrankaRight

DROP_FRAME_NUM = 5
LEFT_CAM_ID = '419122270338'#'419122270338' # 315122271073
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

if __name__ == '__main__':
    # ros and system related init
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("franka_data_recording")
    broadcaster = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    rate = rospy.Rate(10)  # 10 Hz loop rate

    # recording related init
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', default='left', type=str)    # Logging directory
    parser.add_argument('--zip', default='left', type=str)    # Logging directory
    parser.add_argument('--item', default='small_bag', type=str)    # Logging directory
    # parser.add_argument('--mode', default='open_traj', type=str)#open_traj, play_traj, back_to_default

    parser.add_argument('--base_path', default='./paper_hdf5/human', type=str)    # Logging directory
    args = parser.parse_args()

    human_traj_save_path = os.path.join(args.base_path, args.zip, args.item)
    if not os.path.exists(human_traj_save_path):
        os.makedirs(human_traj_save_path)    

    pipeline, detector = init_devices(args.cam)

    # main loop
    data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}
    event_list = collections.deque(maxlen=5)
    marker_left, marker_right = None, None
    gripper_state = 0.04 # open
    start, event_tirgger, ready = False, False, False

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
        elif event_tirgger and start: # trigger skip frame
            print('skip frame', len(data['gripper_w']))
            continue
        elif ready and not event_tirgger and start: # trigger stop recording
            start = False
            ready = False
            print('stop recording', len(data['gripper_w']))

            save_data(human_traj_save_path)
            data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}

        if start:
            data['translation'].append(human_trans)
            data['rotation'].append(human_rot)
            data['gripper_w'].append(gripper_state)
            data['rgb'].append(rgb)
            data['depth'].append(depth*1)

        rate.sleep()