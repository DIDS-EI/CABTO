import os
os.environ["SIM_MODE"] = "Standalone"
import gymnasium as gym
import numpy as np
import torch as th
from dexrl.envs.rl_env import Env as GraspEnv

from dexrl.envs._cfg import EnvCfg

class Env(GraspEnv):
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,)) # 3+2+ 3+6 + 1 +6
    action_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    # action_space = gym.spaces.Box(low=-1, high=1, shape=(18,))

    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True, headless:bool=False):
        super().__init__(cfg, launch_app, headless)


    def launch_scenario(self):
        # from dexrl.sim.scenarios.bihands_sim import BiHandsScenario, BiHandsScenarioCfg
        # self.scenario:BiHandsScenario = BiHandsScenario(BiHandsScenarioCfg())
        from dexrl.sim.scenarios.rl_bihands_10act_sim import RLBiHandsScenario, RLBiHandsScenarioCfg
        self.scenario:RLBiHandsScenario = RLBiHandsScenario(RLBiHandsScenarioCfg())
        
    def update_app(self,action):
        env_step = 0
        action_valid,_ = self.scenario.set_policy_action(action)

        if action_valid:
            
            # success = False
            # while not success:
            #     env_step+=1
            #     print(f"env_step: {env_step}")
            #     current_pos, success = self.scenario.reach_joint_target()

        # self.scenario.set_ik_target_real(current_pos)

        # for i in range(20):  # 从1改为4，增加仿真更新频率
        # if not success:
        #     print("reach_joint_target failed")
        #     return None, None, None, None, None
        
            for i in range(1):  # 从1改为4，增加仿真更新频率
                # self.scenario.step(action)
                current_pos, success = self.scenario.robot.reach_joint_target()
                # current_pos, success = self.scenario.robot.reach_joint_target(left=True)
                if current_pos is not None:
                    print(f"current_pos: {current_pos}, success: {success}")
                self.sim_app.update()
        
        else:
            print("reach_joint_target failed")

        results = self.scenario.get_rl_tuples()

        return results

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
        
        
        # 如果动作改变了,就重新赋值，重新解算IK，会很慢
        # action = th.tensor(action)
        # if self.current_action is None or self.current_action.all() != action.all(): # 如果动作改变了,就重新赋值
        #     action_valid = self.scenario.set_policy_action(action) # IK 解算
        #     if action_valid:
        #         self.current_action = action
        #     else:
        #         self.current_action = None

        # if self.current_action is not None: # 如果动作改变了,就重新赋值
        #     current_pos, success = self.scenario.reach_joint_target()
        #     if success:
        #         self.current_action = None
        
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

    for i in tqdm.tqdm(range(1000000)):
        
        # mixed_pose = env.scenario.robot.get_mixed_pose(left=True)
        # print(f"mixed_pose: {mixed_pose}")
        
        action = env.action_space.sample()
        # action = action.reshape(1,-1)
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        # env.print_obs(observation)
        # env.reset()
        if terminated or truncated:
            env.reset()
        
        # print(env.scenario.get_sim_info())
        # break 
        # image = env.scenario.get_camera_rgb()
        # print(image.shape)
        # cv2.imshow("image", image)
        # cv2.waitKey(1)

    env.close()