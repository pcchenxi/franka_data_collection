from scipy.spatial.transform import Rotation
from franky import Robot, Gripper, Measure
from franky import Affine, JointWaypointMotion, JointWaypoint, CartesianMotion, ReferenceType, CartesianPoseStopMotion, CartesianWaypointMotion, CartesianWaypoint
# from franka_robot.tora import TORA
import numpy as np

class Franka:
    def __init__(self, robot_ip, relative_dynamics_factor=0.05) -> None:
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)
        # self.tora = TORA()

        self.robot.relative_dynamics_factor = relative_dynamics_factor
        # self.robot.velocity_rel = 0.05
        self.robot.acceleration_rel = 0.5
        self.robot.jerk_rel = 0.5

        self.relative_df = relative_dynamics_factor

        self.pose_shift = [0, 0, 0]
        self.start_joint_pose = []
        self.sup_joint_pose = []

        imp_value = 1000
        torque_threshold = 50
        force_threshold = 60
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])
        self.robot.set_collision_behavior(
            [torque_threshold, torque_threshold, torque_threshold, torque_threshold, torque_threshold, torque_threshold, torque_threshold],  # Torque thresholds (Nm)
            [force_threshold, force_threshold, force_threshold, force_threshold, force_threshold, force_threshold]       # Force thresholds (N)
        )

        self.robot.recover_from_errors()

    def set_soft(self, imp_value=150):
        imp_value = imp_value
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

    def set_hard(self, imp_value=1000):
        imp_value = imp_value
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

    def speed_down(self):
        self.robot.relative_dynamics_factor = self.relative_df * 0.5

    def speed_normal(self):
        self.robot.relative_dynamics_factor = self.relative_df

    def set_default_pose(self):
        self.robot.relative_dynamics_factor = 0.1
        motion = CartesianMotion(Affine([-0.04, 0.0, 0.0]), ReferenceType.Relative)
        self.robot.move(motion)    

        robot_pose = self.robot.current_pose
        ee_trans = robot_pose.end_effector_pose.translation
        ee_quat = robot_pose.end_effector_pose.quaternion
        # motion = CartesianMotion(Affine(ee_trans+np.array([-0.05, 0.0, 0.1]), ee_quat))
        # self.robot.move(motion)
        if ee_trans[0] > 0.5:
            self.set_joint_pose(self.sup_joint_pose)

        self.set_joint_pose(self.start_joint_pose)
        self.robot.relative_dynamics_factor = self.relative_df

    def open_gripper(self, asynchronous=True):
        # if asynchronous:
        #     self.gripper.move_async(0.04, 0.02)
        # else:
        #     self.gripper.open(0.04)
        success = self.gripper.move(0.04, 0.02)

    def close_gripper(self, asynchronous=True):
        # if asynchronous:
        #     self.gripper.move_async(0.0, 0.03)
        # else:
        #     self.gripper.move(0.0, 0.03)
        # success = self.gripper.move(0.05, 0.03)
        self.gripper.grasp(0.0, 0.05, 20, epsilon_outer=1.0)

    def set_gripper_opening(self, width, asynchronous=True):
        current_width = self.gripper.width
        # if asynchronous:
        #     self.gripper.move_async(width, 0.02)
        # else:
        if width > 0.01:
            width = 0.04
        else:
            width = 0.0

        if abs(current_width - width) > 0.01:
            self.gripper.move(width, 0.03)
        # success = self.gripper.move(0.0, 0.02)

    def set_ee_pose(self, translation, quaternion, asynchronous=True):
        # print('set ee')
        shifted_translation = translation - self.pose_shift
        motion = CartesianMotion(Affine(shifted_translation, quaternion))
        self.robot.move(motion, asynchronous=asynchronous)

    def set_ee_pose_relative(self, translation, asynchronous=False):
        # shifted_translation = translation - self.pose_shift
        motion = CartesianMotion(Affine(translation), ReferenceType.Relative)
        self.robot.move(motion, asynchronous=asynchronous)

    def set_sora_pose(self, ee_trans, ee_quat, asynchronous=False):
        target_ee_trans = ee_trans - self.pose_shift
        target_ee_quat = ee_quat
        current_ee_trans = self.robot.current_pose.end_effector_pose.translation
        current_ee_quat = self.robot.current_pose.end_effector_pose.quaternion
        joint_pose = self.robot.state.q
        joint_v = self.robot.current_joint_velocities

        target_joint, _, _ = self.tora.plan(joint_pose, joint_v, current_ee_trans, current_ee_quat, target_ee_trans, target_ee_quat)
        # return target_joint
        self.set_joint_pose(target_joint[-1], asynchronous=asynchronous)

    def set_joint_pose(self, joint_pose, asynchronous=False):
        assert len(joint_pose) == 7
        m1 = JointWaypointMotion([JointWaypoint(joint_pose)])
        self.robot.move(m1, asynchronous=asynchronous)

    def set_joint_trajectories(self, joint_trajectory, asynchronous=False):
        waypoints = []
        if len(joint_trajectory) == 1:
            joint_trajectory = np.array([joint_trajectory])

        print(joint_trajectory)
        for i in range(len(joint_trajectory)):
            js = joint_trajectory[i]
            wpoint = JointWaypoint(js)
            # wpoint = JointWaypoint(JointState(position=js, velocity=jv))
            waypoints.append(wpoint)
        motion = JointWaypointMotion(waypoints)
        # m1 = JointWaypointMotion([JointWaypoint(joint_pose)])
        self.robot.move(motion, asynchronous=asynchronous)

    def get_ee_pose(self):
        robot_pose = self.robot.current_pose
        ee_trans = robot_pose.end_effector_pose.translation
        ee_quat = robot_pose.end_effector_pose.quaternion
        ee_rpy = Rotation.from_quat(ee_quat).as_euler('xyz')

        shifted_ee_trans = ee_trans + self.pose_shift
        return shifted_ee_trans, ee_quat, ee_rpy

    def get_joint_pose(self):
        state = self.robot.state
        joint_pose = state.q
        return joint_pose

    def get_elbow_pose(self):
        state = self.robot.state
        elbow_pose = state.elbow
        return elbow_pose
    
    def get_joint_vel(self):
        return self.robot.current_joint_velocities
    
    def get_gripper_width(self):
        return self.gripper.width
    
    def get_ee_force(self):
        fx, fy, fz = self.robot.state.O_F_ext_hat_K[0:3]
        normal_force = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
        # print(self.robot.state.O_F_ext_hat_K, normal_force)
        # normal_force = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
        return fx, fy, fz, normal_force