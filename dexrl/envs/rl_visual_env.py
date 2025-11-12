import os
os.environ["SIM_MODE"] = "Standalone"
import gymnasium as gym
import numpy as np
import torch as th

from dexrl.envs._cfg import EnvCfg
from dexrl.envs.app_launcher import AppLauncher

from dexrl.envs.rl_env import Env as RLEnv

class Env(RLEnv):
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,)) # 31 有旋转
    

    # 状态空间维度：6+3+5+1
    state_dim = 15
    
    # 图像尺寸 - 根据实际摄像头分辨率调整
    image_height = 128  # 可以根据实际摄像头调整
    image_width = 128   # 可以根据实际摄像头调整
    
    observation_space = gym.spaces.Dict({
        "state": gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        ),
        "front": gym.spaces.Box(
            low=0,
            high=255,
            shape=(image_height, image_width, 3),
            dtype=np.uint8
        ),
        "wrist": gym.spaces.Box(
            low=0,
            high=255,
            shape=(image_height, image_width, 3),
            dtype=np.uint8
        )

    })
    
    action_space_singe = gym.spaces.Box(low=-1, high=1, shape=(5,))
    # action_space_dual = gym.spaces.Box(low=-1, high=1, shape=(18,))
    action_space = action_space_singe

    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True, headless:bool=False):
        super().__init__(cfg, launch_app, headless)
        self.set_policy_action_func = self.scenario.set_policy_action
        self.reach_joint_target_func = self.scenario.robot.right_arm_hand.reach_joint_target
        # self.reach_joint_target_func = self.scenario.robot.right_arm.reach_joint_target

    def single_observation_space(self):
        """返回状态空间，用于兼容性"""
        return self.observation_space["state"]  

    def launch_scenario(self):
        # from dexrl.sim.scenarios_dual_arm.rl_dual_arm import RLDualArmScenario, RLDualArmScenarioCfg
        # self.scenario:RLDualArmScenario = RLDualArmScenario(RLDualArmScenarioCfg())
        
        # from dexrl.sim.scenarios.rl_insert_lego_sim_3act import RLInsertLegoScenario, RLInsertLegoScenarioCfg
        # self.scenario:RLInsertLegoScenario = RLInsertLegoScenario(RLInsertLegoScenarioCfg())
        
        from dexrl.sim.scenarios.visual_rl_grasp_5act_sim import VisualRLGraspScenario, VisualRLGraspScenarioCfg
        self.scenario:VisualRLGraspScenario = VisualRLGraspScenario(VisualRLGraspScenarioCfg())


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
 

    def print_obs(self,obs):
        pass
        # print("---------- 观测 --------------")
        # print(f"mixed_pose: {obs[:3]}")
        # print(f"hand_angles_1: {obs[3:5]}")
        # print(f"current_action_state: {obs[5:10]}")
        # print(f"obj_pos: {obs[10:13]}")
        # print(f"obj_rot: {obs[13:19]}")
        # print(f"bbox: {obs[19:25]}")
        # print(f"task_stage: {obs[25]}")
        # print("------------------------------")
        
    def reset(self, seed=None, options=None):
        _,_,obj_pos = self.scenario.reset()
        # 减少初始化时的仿真更新次数
        for i in range(10):  # 从20减少到10
            self.sim_app.update()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        self.task_stage = self.scenario.task_stage #????
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
        self.task_stage = self.scenario.task_stage
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
        
        # print(env.scenario.get_sim_info())
        # break 
        
        # rgbd 和 mask
        
        image_dict = env.scenario.get_camera_rgb()
        top_image = image_dict["top_rgb"]
        wrist_image = image_dict["wrist_rgb"]
        print(top_image.shape)
        print(wrist_image.shape)
        color_top_image = cv2.cvtColor(top_image, cv2.COLOR_BGR2RGB)
        color_wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("top_image", color_top_image)
        cv2.imshow("wrist_image", color_wrist_image)
        cv2.waitKey(1)
        
    env.close()