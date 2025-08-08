"""
@Author: Prince Wang
@Date: 2024-02-22
@Last Modified by:   Prince Wang
@Last Modified time: 2023-10-24 23:04:04
"""

import os
import pybullet
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random
from reward import grasp_reward
from stable_baselines3 import PPO

path = os.path.join(os.path.dirname(__file__), "../")
model1 = PPO.load(path + "models/PPO1/best_model.zip")
model2 = PPO.load(path + "models/PPO2/best_model.zip")
model3 = PPO.load(path + "models/PPO3/best_model.zip")
model4 = PPO.load(path + "models/PPO4/best_model.zip")
# model5 = PPO.load(path + "models/PPO5/best_model.zip")
# model6 = PPO.load(path + "models/PPO6/best_model.zip")
# model7 = PPO.load(path + "models/PPO7/best_model.zip")
# model8 = PPO.load(path + "models/PPO8/best_model.zip")
models = [model1, model2, model3, model4]
print("模型载入完毕")


class FR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        gui=False,
        use_guide_model=True,
        test=False,
        stage_skip_rate=0,
        guide_rate=1.0,
        online=False,
        info="",
    ):
        super(FR5_Env).__init__()
        self.use_stage_skip = False  # 是否启用阶段跳过以实现数据采集
        self.stage_skip_rate = stage_skip_rate  # 阶段跳过的概率
        self.test = test
        self.step_num = 0
        self.stage = 0
        self.use_guide_model = use_guide_model  # 是否以一定概率使用指导模型动作
        self.guide_rate = guide_rate if test is False else 0  # 指导模型动作的概率
        self.info = info
        self.online = online

        self.terminated = False
        self.truncated = False
        self.success = False
        self.reward = 0
        self.observation = np.zeros(16, dtype=np.float32)
        self.guide_observation = np.zeros(15, dtype=np.float32)
        self.target = np.zeros(4, dtype=np.float32)
        self.targets = np.zeros((5, 3), dtype=np.float32)

        self.grasp_zero = [0, 0]
        self.grasp_effort = [0.03, 0.03]
        self.grasp_zero_sym = [0.075, 0.075]
        self.grasp_effort_sym = [0.045, 0.045]
        self.grasp_effort_ori = [0.03, 0.03]
        self.grasp_center_dis = 0.169
        self.grasp_edge_dis = 0.180
        # 设置最小的关节变化量
        low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(
            low=low_action, high=high_action, dtype=np.float32
        )

        low = np.zeros((1, 16), dtype=np.float32)
        high = np.ones((1, 16), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # lower limits for null space
        self.ll = [-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543]
        # upper limits for null space
        self.ul = [3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543]
        # joint ranges for null space
        self.jr = [
            6.28318530718,
            6.28318530718,
            5.6558,
            6.28318530718,
            6.28318530718,
            6.28318530718,
        ]
        # restposes for null space
        self.rp = [1.19826176, -1.2064331, 1.85829957, -0.72282605, 1.44937236, 0.0]

        # 初始化pybullet环境
        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        # self.p.setTimeStep(1/240)
        # print(self.p)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 初始化环境
        self.init_env()

    def init_env(self):
        """仿真环境初始化"""
        # 创建机械臂
        fr5_path = path + "fr5_description/urdf/fr5v6.urdf"
        drawer_path = path + "fr5_description/urdf/drawer2.urdf"
        # print(fr5_path)
        self.fr5 = self.p.loadURDF(
            fr5_path,
            useFixedBase=True,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            flags=p.URDF_USE_SELF_COLLISION,
        )

        self.machine = self.p.loadURDF(
            "fr5_description/urdf/machine.urdf",
            useFixedBase=True,
            basePosition=[0, 2, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
        )

        # 创建桌子
        self.table = p.loadURDF(
            "table/table.urdf",
            basePosition=[0, 0.5, -0.63],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        )

        # 创建抽屉
        self.drawer = self.p.loadURDF(
            drawer_path,
            useFixedBase=True,
            basePosition=[0, 1, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            flags=p.URDF_USE_SELF_COLLISION,
        )
        p.changeDynamics(
            self.drawer,  # 抽屉的 ID
            -1,  # 链接索引（-1 表示基座）
            lateralFriction=10,  # 设置侧向摩擦力
            spinningFriction=5,  # 设置旋转摩擦力
            rollingFriction=5,  # 设置滚动摩擦力
        )

        # 添加红色方形目标物
        self.cube_size = 0.04  # 方形目标物的边长
        col_cube_id = self.p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_size / 3, self.cube_size / 3, self.cube_size],
        )
        vis_cube_id = self.p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_size / 3, self.cube_size / 3, self.cube_size],
            rgbaColor=[1, 0, 0, 0.5],
        )
        self.target_cube = self.p.createMultiBody(
            baseMass=5,
            baseCollisionShapeIndex=col_cube_id,
            baseVisualShapeIndex=vis_cube_id,
            basePosition=[0, 1, 0.3],
        )

        # 设置目标物的摩擦力
        p.changeDynamics(
            self.target_cube,  # 目标物的 ID
            -1,  # 链接索引（-1 表示基座）
            lateralFriction=10,  # 设置侧向摩擦力
            spinningFriction=5,  # 设置旋转摩擦力
            rollingFriction=5,  # 设置滚动摩擦力
        )

    def step(self, action):
        """step"""
        info = {}
        # Execute one time step within the environment
        guide_action, _ = models[self.stage].predict(
            observation=self.guide_observation, deterministic=True
        )

        # 比较两个动作的差异
        diff = np.sum(abs(guide_action - action))

        # 初始化关节角度列表
        joint_angles = []

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        if self.use_guide_model and np.random.uniform(0, 1) < self.guide_rate:
            action = guide_action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action) / 180 * np.pi)

        anglenow = np.hstack([Fr5_joint_angles])
        p.setJointMotorControlArray(
            self.fr5,
            [1, 2, 3, 4, 5, 6],
            p.POSITION_CONTROL,
            targetPositions=anglenow,
        )

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self, diff)

        # observation计算
        self.get_observation()

        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def flashUR5(self):
        """夹爪干涉出现时，反置机械臂"""
        if self.grasp_zero[0] == 0:
            self.grasp_zero = self.grasp_zero_sym
            self.grasp_effort = self.grasp_effort_sym
        elif self.grasp_zero[0] == self.grasp_zero_sym[0]:
            self.grasp_zero = [0, 0]
            self.grasp_effort = self.grasp_effort_ori

    def move(self, position):
        Euler = [-1.57079, 3.14159, 0]
        orn = p.getQuaternionFromEuler(Euler)
        pos = [i for i in position]
        pos[1] -= 0.24
        for _ in range(3):
            joint_angles = p.calculateInverseKinematics(
                self.fr5,
                6,
                pos,
                orn,
                lowerLimits=self.ll,
                upperLimits=self.ul,
                jointRanges=self.jr,
                restPoses=self.rp,
            )
            joint_angles = list(joint_angles[:6])
            p.setJointMotorControlArray(
                self.fr5,
                [1, 2, 3, 4, 5, 6],
                p.POSITION_CONTROL,
                targetPositions=joint_angles,
            )
            for _ in range(100):
                self.p.stepSimulation()
                # time.sleep(1./240)

    def reset(self, seed=None, options=None):
        if (
            self.use_stage_skip == True
            and np.random.uniform(0, 1) < self.stage_skip_rate
        ):
            self.stage = (self.stage + 1) % 4
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        self.success_ = False
        self.contact = False
        if self.stage == 0:
            # 重新设置机械臂的位置
            neutral_angle = [30, -137, 128, 9, 30, 0, 0, 0]
            neutral_angle = [x * math.pi / 180 for x in neutral_angle]

            # 1.48354575 -1.20812401  2.08497612 -0.85981021  1.66050315  0.
            # neutral_angle = [1.48354575, -1.20812401, 2.08497612, -0.85981021, 1.66050315, 0., 0, 0]
            p.setJointMotorControlArray(
                self.fr5,
                [1, 2, 3, 4, 5, 6, 8, 9],
                p.POSITION_CONTROL,
                targetPositions=neutral_angle,
            )
            for _ in range(20):
                self.p.stepSimulation()

            error_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.fr5)
            for contact_point in error_contact_points:
                link_index = contact_point[3]
                if link_index == 7 or link_index == 8:
                    logger.info("夹爪干涉出现！")
                    self.contact = True
                    self.flashUR5()
                    for _ in range(10):
                        self.p.stepSimulation()
                    break

            # 随机生成初始位置
            euler_z = np.random.uniform(-0.1, 0.1, 1)
            Euler = [-1.57079, 3.14159, euler_z[0]]
            orn = p.getQuaternionFromEuler(Euler)
            pos = [0, 0.35, 0.3]
            # 关节六实际位置[0,0.42,0.3],y+=0.07

            # 循环以使结果逼近
            for _ in range(3):
                joint_angles = p.calculateInverseKinematics(
                    self.fr5,
                    6,
                    pos,
                    orn,
                    lowerLimits=self.ll,
                    upperLimits=self.ul,
                    jointRanges=self.jr,
                    restPoses=self.rp,
                )
                joint_angles = list(joint_angles[:6]) + self.grasp_zero
                p.setJointMotorControlArray(
                    self.fr5,
                    [1, 2, 3, 4, 5, 6, 8, 9],
                    p.POSITION_CONTROL,
                    targetPositions=joint_angles,
                )
                for _ in range(100):
                    self.p.stepSimulation()

            # 设置初始位置
            Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
            Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
            Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
            relative_position = np.array([0, 0, self.grasp_center_dis])

            # 固定夹爪相对于机械臂末端的相对位置转换
            rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
            rotated_relative_position = rotation.apply(relative_position)
            self.gripper_centre_pos = [
                Gripper_posx,
                Gripper_posy,
                Gripper_posz,
            ] + rotated_relative_position

            # 重置抽屉位置
            self.drawer_xpos = np.random.uniform(-0.05, 0.05, 1)[0]
            self.drawer_ypos = np.random.uniform(0.93, 1, 1)[0]
            p.resetBasePositionAndOrientation(
                self.drawer, [self.drawer_xpos, self.drawer_ypos, 0], [0, 0, 1, 0]
            )
            num_joints = p.getNumJoints(self.drawer)  # 获取抽屉的关节数量
            for joint_index in range(num_joints):
                p.resetJointState(
                    self.drawer, joint_index, targetValue=0.0
                )  # 将关节位置复位为 0

            # 重置方形目标物的位置
            self.goalx = self.drawer_xpos + np.random.uniform(-0.05, 0.05, 1)[0]
            self.goaly = self.drawer_ypos + np.random.uniform(-0.1, -0.05, 1)[0]
            self.goalz = 0.3
            self.target_position = np.array([self.goalx, self.goaly, self.goalz])
            p.resetBasePositionAndOrientation(
                self.target_cube, self.target_position, [0, 0, 0, 1]
            )

            for _ in range(100):
                self.p.stepSimulation()

            # 抽屉把手位置
            drawer_handle_position = np.array(p.getLinkState(self.drawer, 14)[0])
            # 目标物位置
            self.target_position = np.array(
                p.getBasePositionAndOrientation(self.target_cube)[0]
            )

            self.target1 = [i for i in drawer_handle_position]
            self.target2 = [i for i in self.target1]
            self.target2[1] -= 0.27
            # self.target3 = [i for i in self.target2]
            # self.target3[2] += 0.18
            self.target4 = [i for i in self.target_position]
            self.target4[1] += 0.02
            self.target5 = [i for i in self.target2]
            self.target5[1] += 0.15
            self.target5[2] = self.target_position[2]
            # self.target6 = [i for i in self.target3]
            # self.target7 = [i for i in self.target2]
            # self.target7[1] += 0.02
            # self.target8 = [i for i in self.target1]
            self.targets = [self.target1, self.target2,self.target4,self.target5]

        if self.stage == 1:
            self.move(self.target1)
            self.close_gripper()
        if self.stage == 2:
            self.move(self.target2)
            self.open_gripper()
        if self.stage == 3:
            # self.move((self.target2[0], self.target2[1], self.target2[2]+0.18))
            self.move(self.target4)
            self.close_gripper()
            # self.move(self.target5)
            # self.open_gripper()

        self.target[:3] = self.targets[self.stage]
        self.target[3] = self.stage / 3
        self.get_observation()

        infos = {}
        infos["is_success"] = False
        infos["reward"] = 0
        infos["step_num"] = 0
        return self.observation, infos

    def close_gripper(self):
        """合并夹抓"""
        p.setJointMotorControlArray(
            self.fr5, [8, 9], p.POSITION_CONTROL, targetPositions=[0.03, 0.03]
        )
        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1 / 240.)

    def open_gripper(self):
        """打开夹抓"""
        p.setJointMotorControlArray(
            self.fr5, [8, 9], p.POSITION_CONTROL, targetPositions=[0, 0]
        )
        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1 / 240.)

    def get_gripper_position(self):
        # 获取夹爪中心位置和朝向
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, self.grasp_center_dis])
        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = [
            Gripper_posx,
            Gripper_posy,
            Gripper_posz,
        ] + rotated_relative_position

        gripper_orientation = p.getLinkState(self.fr5, 7)[1]
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler("xyz", degrees=True)

        return gripper_centre_pos, gripper_orientation

    def get_observation(self, add_noise=False):
        """计算observation"""
        gripper_centre_pos, gripper_orientation = self.get_gripper_position()
        obs_gripper_centre_pos = np.array(gripper_centre_pos, dtype=np.float32)
        obs_gripper_orientation = np.array(
            gripper_orientation / 180 * np.pi, dtype=np.float32
        )
        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0]
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(
                    joint_angles[i - 1], range=0, gaussian=True
                )

        # 机械臂各关节角度
        joint_angles = np.array(joint_angles, dtype=np.float32)

        self.observation = np.hstack(
            (joint_angles, obs_gripper_centre_pos, obs_gripper_orientation, self.target)
        )

        # [1,16]
        self.observation = self.observation.reshape(1, -1)
        self.guide_observation = self.observation.copy()[:, :-1]
        return self.observation

    def render(self):
        """设置观察角度"""
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=90,
            cameraPitch=-7.6,
            cameraTargetPosition=[0.39, 0.45, 0.42],
        )

    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        """添加噪声"""
        if gaussian:
            angle += np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            angle += random.uniform(-5, 5)
        return angle


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    Env = FR5_Env(gui=True)
    for i in range(100):
        for j in range(4):
            Env.stage = j
            Env.reset()
            time.sleep(1)

    # # x 轴
    # frame_start_postition, frame_posture = p.getBasePositionAndOrientation(Env.drawer)
    # frame_start_postition = [0, 0.8, 0.3]
    # R_Mat = np.array(p.getMatrixFromQuaternion(frame_posture)).reshape(3, 3)
    # x_axis = R_Mat[:, 0]
    # x_end_p = (np.array(frame_start_postition) + np.array(x_axis * 5)).tolist()
    # x_line_id = p.addUserDebugLine(frame_start_postition, x_end_p, [1, 0, 0])

    # # y 轴
    # y_axis = R_Mat[:, 1]
    # y_end_p = (np.array(frame_start_postition) + np.array(y_axis * 5)).tolist()
    # y_line_id = p.addUserDebugLine(frame_start_postition, y_end_p, [0, 1, 0])

    # # z轴
    # z_axis = R_Mat[:, 2]
    # z_end_p = (np.array(frame_start_postition) + np.array(z_axis * 5)).tolist()
    # z_line_id = p.addUserDebugLine(frame_start_postition, z_end_p, [0, 0, 1])

    # time.sleep(10)
    # check_env(Env, warn=True)

    for i in range(100):
        p.stepSimulation()

    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
