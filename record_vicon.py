#!/usr/bin/env python
import rospy, time
from geometry_msgs.msg import TransformStamped
import numpy as np
import pickle 
from copy import deepcopy
import signal
import sys, os
import argparse


gripper_translation, gripper_rotation = None, None
data = {'path':'', 'rgb':[], 'depth':[], 'translation':[], 'rotation':[]}

def callback(data):
    global gripper_translation, gripper_rotation
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    # print(data.transform.translation)
    # print(data.transform.rotation)
    gripper_translation = data.transform.translation
    gripper_rotation = data.transform.rotation
    
def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/vicon/gripper_r/gripper_r", TransformStamped, callback)


def record_data(path):
    global data
    data['path'] = path
    while True:
        if gripper_translation is None or gripper_rotation is None:
            time.sleep(0.1)
            continue

        data['translation'].append([gripper_translation.x, gripper_translation.y, gripper_translation.z])
        data['rotation'].append([gripper_rotation.x, gripper_rotation.y, gripper_rotation.z, gripper_rotation.w])
        # data['translation'].append((0,0,0))
        # data['rotation'].append((0,0,0,0))

        time.sleep(0.1)   

def play_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    trans_list, rot_list = data['translation'], data['rotation']
    for trans, rot in zip(trans_list, rot_list):
        print(trans)
        print(rot)
        print()

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    path = data['path']
    with open(path, 'wb') as f:
        pickle.dump(data, f)   

    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='open_ccw', type=str)    # Logging directory
    parser.add_argument('--mode', default='record', type=str)    # Logging directory
    args = parser.parse_args()

    dir_path = './'+args.name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f_list = os.listdir(dir_path)
    print(len(f_list))

    listener()
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')

    if args.mode == 'record':
        path = './' + args.name + '/' + str(len(f_list)) + '.pkl'
        record_data(path)
    else:
        path = './' + args.name + '/' + str(len(f_list)-1) + '.pkl'
        play_data(path)

    # signal.pause()