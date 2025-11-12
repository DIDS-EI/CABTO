import os
os.environ["SIM_MODE"] = "Standalone"
import gymnasium as gym
import numpy as np
import torch as th
from dexrl.envs.rl_env import Env as RLEnv

from dexrl.envs._cfg import EnvCfg

class Env(RLEnv):
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,)) # 3+2+ 3+6 + 1 +6
    # action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))
    action_space_singe = gym.spaces.Box(low=-1, high=1, shape=(9,))
    action_space_dual = gym.spaces.Box(low=-1, high=1, shape=(18,))
    action_space = action_space_singe

    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True, headless:bool=False):
        super().__init__(cfg, launch_app, headless)
        self.set_policy_action_func = self.scenario.set_dual_policy_action
        # self.reach_joint_target_func = self.scenario.robot.left_arm.reach_joint_target
        self.reach_joint_target_func = self.scenario.robot.right_arm_hand.reach_joint_target


    def launch_scenario(self):
        # from dexrl.sim.scenarios.bihands_sim import BiHandsScenario, BiHandsScenarioCfg
        # self.scenario:BiHandsScenario = BiHandsScenario(BiHandsScenarioCfg())
        from dexrl.sim.scenarios_dual_arm.rl_dual_arm import RLDualArmScenario, RLDualArmScenarioCfg
        self.scenario:RLDualArmScenario = RLDualArmScenario(RLDualArmScenarioCfg())
        
    def update_app(self,action):
        env_step = 0
        action_valid,_ = self.set_policy_action_func(action)

        if action_valid:
            for i in range(1):  # 从1改为4，增加仿真更新频率
                # self.scenario.step(action)
                current_pos, success = self.reach_joint_target_func()
                # current_pos, success = self.scenario.robot.reach_joint_target(left=True)
                if current_pos is not None:
                    print(f"current_pos: {current_pos}, success: {success}")
                self.sim_app.update()
        else:
            print("reach_joint_target failed")

        results = self.scenario.get_rl_tuples()

        return results


    def set_mode(self,mode:str):
        if mode == "left":
            self.set_policy_action_func = self.scenario.set_left_policy_action
            self.reach_joint_target_func = self.scenario.robot.left_arm_hand.reach_joint_target
            self.action_space = self.action_space_singe
        elif mode == "right":
            self.set_policy_action_func = self.scenario.set_right_policy_action
            self.reach_joint_target_func = self.scenario.robot.right_arm_hand.reach_joint_target
            self.action_space = self.action_space_singe
        elif mode == "dual":
            self.set_policy_action_func = self.scenario.set_dual_policy_action
            self.reach_joint_target_func = self.scenario.robot.dual_arm_hand.reach_joint_target
            self.action_space = self.action_space_dual
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def print_obs(self,obs):
        print("---------- 观测 --------------")
        print(f"mixed_pose: {obs[:3]}")
        print(f"hand_angles_1: {obs[3:5]}")
        print(f"current_action_state: {obs[5:10]}")
        print(f"obj_pos: {obs[10:13]}")
        print(f"obj_rot: {obs[13:19]}")
        print(f"task_stage: {obs[19]}")
        print("------------------------------")
        
    def reset(self, seed=None, options=None):
        _,_,obj_pos = self.scenario.reset()
        # 减少初始化时的仿真更新次数
        for i in range(10):  # 从20减少到10
            self.sim_app.update()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        # self.last_real_action = np.array([0,0,0,   -3.1, 2.7 , -3.2   ,0,0,0])
        return self.current_observation, {"obj_pos": obj_pos}

    def reset_and_set_obj_pose(self, obj_pos, obj_quat): # 数据重放的时候需要
        self.scenario.reset_and_set_obj_pose(obj_pos, obj_quat)
        for i in range(3):  # 从20减少到10
            self.sim_app.update()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        
        return self.current_observation, {"obj_pos": obj_pos}
    
    def step(self, action):
        action = th.tensor(action)
        results = self.update_app(action)
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps += 1
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

    # cfg = DexrlCfg()
    cfg = EnvCfg(device="cpu")
    env:Env = Env(cfg)
    env.launch_app()
    observation, info = env.reset()

    env.set_mode("dual")
    action = np.zeros(18)
    action[0] = 0.5
    action[9] = -0.5
    for i in tqdm.tqdm(range(100)):
        observation, reward, terminated, truncated, info = env.step(action)


    env.set_mode("left")

    action = np.zeros(9)
    action[0] = -0.5
    for i in tqdm.tqdm(range(100)):
        observation, reward, terminated, truncated, info = env.step(action)

    action = np.zeros(9)
    action[2] = 0.5
    for i in tqdm.tqdm(range(100)):

        print(env.scenario.robot.left_arm_hand.arm.get_mixed_pose())
        observation, reward, terminated, truncated, info = env.step(action)



    env.set_mode("right")

    action = np.zeros(9)
    action[0] = 0.5
    for i in tqdm.tqdm(range(100)):
        observation, reward, terminated, truncated, info = env.step(action)

    action = np.zeros(9)
    action[2] = 0.5
    for i in tqdm.tqdm(range(100)):

        print(env.scenario.robot.left_arm_hand.arm.get_mixed_pose())
        observation, reward, terminated, truncated, info = env.step(action)


    mode_list = ["left","right","dual"]

    for mode in mode_list:
        env.set_mode(mode)
        for i in tqdm.tqdm(range(100)):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
        

    env.close()