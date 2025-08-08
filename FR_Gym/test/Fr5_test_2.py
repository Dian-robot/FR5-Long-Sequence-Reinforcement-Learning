import sys
import os
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from stable_baselines3 import PPO

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

if __name__ == '__main__':
    args, kwargs = get_args()
    online = False
    env = FR5_Env(gui=False, test=True, online=online)
    env.render()
    success_rate = []
    model_dir = "paths/to/your/model/directory"  # 替换为你的模型目录路径

    # 创建TensorBoard的SummaryWriter对象，在名称中添加model_dir最后一个文件夹的名称和当前时间
    log_dir = "./test_logs_new/" + model_dir.split("/")[-2] + "/" + time.strftime('%m%d-%H%M%S', time.localtime())
    writer = SummaryWriter(log_dir)
    for _ in range(100):
        episode = (_) * 5 + 0
        model_path = os.path.join(model_dir, f"PPO-run-eposide{episode}.zip")
        print("model_path:", model_path)
        while True:
            if not os.path.exists(model_path):
                print("模型", episode, "未就绪，等待")
                time.sleep(60)
            else:
                break

        model_path = os.path.splitext(model_path)[0]
        success_rate_episode = []
        model = PPO.load(model_path)
        for test_stage in range(4):
            test_num = args.test_num  # 测试次数
            test_num = 50  # 测试次数
            error_num = 0  # 前置任务失败次数
            info = {"is_success": True}
            success_num = 0  # 成功次数
            print("测试次数：", test_num)
            for i in range(test_num):
                env.stage = 0
                info = {"is_success": True}
                for j in range(test_stage):
                    # time.sleep(1)
                    done = False
                    score = 0
                    # time.sleep(3)
                    step = 0
                    env.stage = j
                    state, _ = env.reset()
                    while not done:
                        step += 1
                        action, _ = guide_models[j].predict(observation=env.guide_observation, deterministic=True)
                        state, reward, done, _, info = env.step(action=action)
                        score += reward
                        # time.sleep(0.02)
                if not info['is_success']:
                    error_num += 1
                    print("前置任务失败！不计入总次数")
                    continue
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
                    # time.sleep(0.02)
                    if info['is_success']:
                        success_num += 1
                print("模型：", episode, "阶段：", test_stage, "奖励：", score)
            if test_num - error_num == 0:
                success_rate_episode.append(0)
            else:
                success_rate_episode.append(success_num / (test_num - error_num))
            print("模型：", episode, "阶段：", test_stage, "成功率：", success_rate_episode[test_stage])
        success_rate.append(success_rate_episode)
        print("模型：", episode, "成功率：", success_rate[-1], "平均成功率：", sum(success_rate[-1]) / 4)
        # 将成功率写入TensorBoard

        for i, rate in enumerate(success_rate_episode):
            writer.add_scalar(f'Success_Rate/Stage_{i}', rate, episode)
        writer.add_scalar(f'Success_Rate/Average', sum(success_rate_episode) / 4, episode)

    print("成功率：", success_rate)
    env.close()

    # 关闭TensorBoard的SummaryWriter对象
    writer.close()
