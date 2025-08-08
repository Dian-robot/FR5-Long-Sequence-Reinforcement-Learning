'''
@Author: Prince Wang 
@Date: 2024-02-22 
@Last Modified by:   Prince Wang 
@Last Modified time: 2023-10-24 23:04:04 
'''
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from stable_baselines3 import A2C, PPO, DDPG, TD3

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from FR_Gym import FR5_Env
import time
from utils.arguments import get_args


path = os.path.join(os.path.dirname(__file__), "../../")
model1 = PPO.load(path + "models/PPO1/best_model.zip")
model2 = PPO.load(path + "models/PPO2/best_model.zip")
model3 = PPO.load(path + "models/PPO3/best_model.zip")
model4 = PPO.load(path + "models/PPO4/best_model.zip")
guide_models = [model1, model2, model3, model4]
print("模型载入完毕")

if __name__ == '__main__':
    args, kwargs = get_args()
    env = FR5_Env(gui=True, guide_rate=0)
    env.render()
    model = PPO.load(
        "FR_Gym/FR5_Reinforcement-learning_half_expert_half_explore/models/PPO/best_model.zip")

    success_rate = []

    # for test_stage in range(5):
    #     info = {"is_success": True}
    #     test_num = args.test_num  # 测试次数
    #     test_num = 5  # 测试次数
    #     success_num = 0  # 成功次数
    #     error_num = 0  # 前置任务失败次数
    #     print("测试次数：", test_num)
    #     for i in range(test_num):
    #         env.stage = 0
    #         info = {"is_success": True}
    #         print("guide_model:")
    #         for j in range(test_stage):
    #             # time.sleep(1)
    #             done = False
    #             score = 0
    #             # time.sleep(3)
    #             step = 0
    #             env.stage = j
    #             state, _ = env.reset()
    #             while not done:
    #                 step += 1
    #                 action, _ = guide_models[j].predict(observation=env.guide_observation, deterministic=True)
    #                 state, reward, done, _, info = env.step(action)
    #                 score += reward
    #                 # time.sleep(0.02)
    #         print("test_model:")
    #         if info['is_success']:
    #             done = False
    #             score = 0
    #             # time.sleep(3)
    #             step = 0
    #             env.stage = test_stage
    #             state, _ = env.reset()
    #             while not done:
    #                 step += 1
    #                 action, _ = model.predict(observation=state, deterministic=True)
    #                 state, reward, done, _, info = env.step(action=action)
    #                 score += reward
    #                 # time.sleep(0.02)
    #                 if info['is_success']:
    #                     success_num += 1
    #             print("奖励：", score/step)
    #         else:
    #             print("前置任务失败！不计入总次数")
    #             error_num += 1
    #     if test_num - error_num == 0:
    #         success_rate.append(0)
    #     else:
    #         success_rate.append(success_num / (test_num - error_num))
    #     print("阶段：", test_stage, "成功率：", success_rate[test_stage])
    # print("成功率：", success_rate)
    #不使用指导模型
    test_num = args.test_num  # 测试次数
    test_num = 100  # 测试次数
    success_num = 0  # 成功次数
    for i in range(test_num):
        for test_stage in range(4):
            done = False
            score = 0
            # time.sleep(3)
            step = 0
            env.stage = test_stage
            state, _ = env.reset()
            while not done:
                step += 1
                action, _ = model.predict(observation=state, deterministic=True)
                state, reward, done, _, info = env.step(action=action)
                score += reward
                # time.sleep(0.005)
                if info['is_success']:
                    if test_stage == 3:
                        env.open_gripper()
                        env.open_gripper()
                        success_num += 1
                        print("成功！")
            if env.success == False:
                print("失败！")
                break
            if test_stage == 0 or test_stage == 2:
                env.close_gripper()
            elif test_stage == 1:
                env.open_gripper()

    env.close()
    print(f"accuracy:{success_num/test_num*100}%")
