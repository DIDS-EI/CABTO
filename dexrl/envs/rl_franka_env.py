import os
os.environ["SIM_MODE"] = "Standalone"
import gymnasium as gym
import numpy as np
import torch as th

from dexrl.envs._cfg import EnvCfg
from dexrl.envs.app_launcher import AppLauncher
from dexrl.utils.print import print_yellow
from dexrl.envs.rl_env import Env as RLEnv
def print_mix_pose(x):
    # 前三维黄色,后三位橘色
    print(f"\033[93m{'mixed_pose:':<10} {x[:3]}\033[0m \033[38;5;208m{x[3:6]}\033[0m")


class Env(RLEnv):
    
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22,)) # 31 有旋转
    action_space_singe = gym.spaces.Box(low=-1, high=1, shape=(4,))
    
    action_space = action_space_singe

    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True, headless:bool=False):
        super().__init__(cfg, launch_app, headless)
        # if launch_app:
            # self.set_policy_action_func = self.scenario.set_policy_action
            # self.reach_joint_target_func = self.scenario.robot.right_arm_hand.reach_joint_target

            # self.set_policy_action_func = self.scenario.set_policy_action
            # self.reach_joint_target_func = self.scenario.robot.left_arm_hand.reach_joint_target

        # 训练的时候不需要考虑 done  
        # self.env_args = {"collect_data":False,"noise":False,"random_init":True}
        
        # 尝试设置 鲁棒性
        self.env_args = {"collect_data":False,"noise":True,"random_init":True}

    def launch_scenario(self):

        # from dexrl.sim.scenarios.rl_franka import RLScenario, RLScenarioCfg
        # self.scenario:RLScenario = RLScenario(RLScenarioCfg())
        
        from dexrl.sim.scenarios.rl_franka_sim import RLFrankaScenario, RLFrankaScenarioCfg
        self.scenario:RLFrankaScenario = RLFrankaScenario(RLFrankaScenarioCfg())

    def update_app(self,action):
        env_step = 0
        # action_valid,_ = self.set_policy_action_func(action)

        # if action_valid:
        #     for i in range(1):  # 从1改为4，增加仿真更新频率
        #         # self.scenario.step(action)
        #         current_pos, success = self.reach_joint_target_func()
        #         # current_pos, success = self.scenario.robot.reach_joint_target(left=True)
        #         # if current_pos is not None:
        #         #     print(f"current_pos: {current_pos}, success: {success}")
        #         self.sim_app.update()
        # else:
        #     print("reach_joint_target failed")
        
        self.scenario.set_policy_action(action)
            
        # 增加仿真更新次数，让机器人移动更流畅,就不太会抖动了
        # for i in range(3):  # 从1次增加到3次
        #     self.sim_app.update()
        self.sim_app.update()
        
        results = self.scenario.get_rl_tuples()

        return results
 

    def print_obs(self,obs):
        pass
        # if self.action_space.shape[0] == 4:
        #     print_mix_pose(obs[:6])
        #     print_yellow(f"hand_angles_1: {obs[6:9]}")
        #     print_yellow(f"current_action_state: {obs[9:13]}")
        #     print_yellow(f"obj_pos: {obs[13:16]}")
        #     print_yellow(f"bbox: {obs[16:22]}")
            # print_yellow(f"task_stage: {obs[22]}")
        # print("---------- 观测 --------------")
        # # print_yellow(f"mixed_pose: {obs[:6]}")
        # print_mix_pose(obs[:6])
        # print_yellow(f"hand_angles_1: {obs[6:9]}")
        # print_yellow(f"current_action_state: {obs[9:14]}")
        # print_yellow(f"obj_pos: {obs[14:17]}")
        # print_yellow(f"bbox: {obs[17:23]}")
        # print_yellow(f"task_stage: {obs[23]}")
        # print("------------------------------")
        
    def reset(self, seed=None, options=None):
        # _,_,obj_pos = self.scenario.reset()
        # 减少初始化时的仿真更新次数
        self.scenario.reset()
        for i in range(10):  # 从20减少到10
            self.sim_app.update()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        # self.task_stage = self.scenario.task_stage #????
        # self.last_real_action = np.array([0,0,0,   -3.1, 2.7 , -3.2   ,0,0,0])
        return self.current_observation, {}

    # def reset_and_set_obj_pose(self, obj_pos, obj_quat): # 数据重放的时候需要
    #     self.scenario.reset_and_set_obj_pose(obj_pos, obj_quat)
    #     for i in range(3):  # 从20减少到10
    #         self.sim_app.update()
    #     results = self.scenario.get_rl_tuples()
    #     self.scenario.env_steps = 0
    #     self.current_observation = results[0]
        
    #     return self.current_observation, {"obj_pos": obj_pos}
    
    def step(self, action):
        action = th.tensor(action)
        results = self.update_app(action)
        results = self.scenario.get_rl_tuples()
        # self.scenario.env_steps += 1
        # self.task_stage = self.scenario.task_stage
        return results

    @property
    def single_observation_space(self):
        return self.observation_space

    @property
    def single_action_space(self):
        return self.action_space

    def close(self):
        self.sim_app.close()

if __name__ == "__main__":
    import tqdm
    import cv2

    cfg = EnvCfg(device="cpu")
    env:Env = Env(cfg)
    env.launch_app()
    observation, info = env.reset()

    # env.set_mode("right")

    for i in tqdm.tqdm(range(10000)):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
        

    env.close()