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
from interval import Interval


def cal_success_reward(self, distance):
    success_reward = 0

    # 夹爪中心和目标之间距离小于一定值，则任务成功
    if self.success == True and self.step_num <= 120:
        success_reward = 1
        self.terminated = True
        self.success = True

    # 机械臂执行步数过多
    if self.step_num > 120:
        success_reward = -1
        self.terminated = True
        logger.info(
            "失败！执行步数过多！当前阶段 %s    执行步数：%s    距离目标:%s"
            % (self.stage, self.step_num, distance)
        )
    # elif self.stage in [1,3]:
    #     pos,_ = self.get_gripper_position()
    #     if pos[1] - self.target[1] <= 0:
    #         success_reward = -100
    #         self.terminated = True
    #         if self.success_ == True:
    #             self.success = True
    #         else:
    #             logger.info(
    #                 "失败！偏离目标！ 执行步数：%s    距离目标:%s"
    #                 % (self.step_num, distance)
    #             )
    # elif self.stage in [0,2]:
    #     pos,_ = self.get_gripper_position()
    #     if pos[1] - self.target[1] >= 0.01:
    #         success_reward = -100
    #         self.terminated = True
    #         if self.success_ == True:
    #             self.success = True
    #         else:
    #             logger.info(
    #                 "失败！偏离目标！ 执行步数：%s    距离目标:%s"
    #                 % (self.step_num, distance)
    #             )


    if self.terminated:
        if self.success:
            logger.info(
                "成功抓取！！！！！！！！！！当前阶段:%s  执行步数：%s  距离目标:%s"
                % (self.stage, self.step_num, distance)
            )
            self.stage = (self.stage + 1) % 4
        else:
            self.stage = 0
        # self.truncated = True

    return success_reward


def cal_pose_reward(self):
    """姿态奖励"""
    # 计算夹爪的朝向
    gripper_orientation = p.getLinkState(self.fr5, 7)[1]
    gripper_orientation = R.from_quat(gripper_orientation)
    gripper_orientation = gripper_orientation.as_euler("xyz", degrees=True)
    # 计算夹爪的姿态奖励
    pose_reward = -(
        pow(gripper_orientation[0] + 90, 2)
        + pow(gripper_orientation[1], 2)
        + pow(gripper_orientation[2], 2)
    )
    # logger.debug("姿态奖励：%f"%pose_reward)
    return pose_reward * 0.01


def cal_dis_reward(self, distance):
    """计算距离奖励"""
    if self.step_num == 0:
        distance_reward = 0
    else:
        distance_reward = 1000 * (self.distance_last - distance)
    # 保存上一次的距离
    self.distance_last = distance
    return distance_reward

def grasp_reward(self, diff=0):
    """获取奖励"""
    info = {}
    # stage需要在结算之前记录
    info["stage"] = self.stage
    total_reward = 0

    distance = get_distance(self, self.targets[self.stage])
    pose_reward = cal_pose_reward(self)
    distance_reward = cal_dis_reward(self, distance)
    judge_success(self, distance, pose_reward, success_dis=0.02, success_pose=-100)
    # judge_success_(self, distance, pose_reward, success_dis=0.04, success_pose=-100)

    # 计算奖励
    success_reward = cal_success_reward(self, distance)
    # 现有模型与目标模型的差异惩罚
    diff_reward = -diff / 10
    total_reward = success_reward + diff_reward
    #3 total_reward = success_reward + diff_reward + pose_reward + 0.1*distance_reward
    #2 total_reward = diff_reward #2

    # reward1 59
    # reward2 60
    # reward3 6

    self.truncated = False
    self.reward = total_reward
    info["reward"] = self.reward
    info["is_success"] = self.success
    info["step_num"] = self.step_num

    info["success_reward"] = 1 if self.success else 0
    info["distance_reward"] = diff_reward
    info["pose_reward"] = pose_reward

    return total_reward, info


def judge_success(self, distance, pose, success_dis, success_pose):
    """判断成功或失败"""
    if distance < success_dis:
        if pose > success_pose:
            self.success = True
        else:
            self.success = False
    else:
        self.success = False
        # total_reward = total_reward + (0.3 - distance)


def judge_success_(self, distance, pose, success_dis, success_pose):
    """判断成功或失败"""
    if distance < success_dis:
        if pose > success_pose:
            self.success_ = True
        else:
            self.success_ = False
    else:
        self.success_ = False


def get_distance(self, target_position):
    """判断机械臂与夹爪的距离"""
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

    distance = math.sqrt(
        (gripper_centre_pos[0] - target_position[0]) ** 2
        + (gripper_centre_pos[1] - target_position[1]) ** 2
        + (gripper_centre_pos[2] - target_position[2]) ** 2
    )
    # logger.debug("distance:%s"%str(distance))
    return distance
