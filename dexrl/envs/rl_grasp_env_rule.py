import os
os.environ["SIM_MODE"] = "Standalone"
import gymnasium as gym
import numpy as np
import torch as th

from dexrl.envs._cfg import EnvCfg
from dexrl.envs.app_launcher import AppLauncher
from dexrl.utils.print import print_yellow
# from dexrl.envs.rl_env import Env as RLEnv
from dexrl.envs.rl_grasp_env import Env as GraspEnv
from dexrl.utils.print import *
def print_mix_pose(x):
    # 前三维黄色,后三位橘色
    print(f"\033[93m{'mixed_pose:':<10} {x[:3]}\033[0m \033[38;5;208m{x[3:6]}\033[0m")


class Env(GraspEnv):
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,)) # 31 有旋转
    # action_space_singe = gym.spaces.Box(low=-1, high=1, shape=(5,))
    # action_space_dual = gym.spaces.Box(low=-1, high=1, shape=(18,))
    
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22,)) # 31 有旋转
    action_space_singe = gym.spaces.Box(low=-1, high=1, shape=(4,))
    
    action_space = action_space_singe

    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True, headless:bool=False):
        super().__init__(cfg, launch_app, headless)
        if launch_app:
            # self.set_policy_action_func = self.scenario.set_policy_action
            # self.reach_joint_target_func = self.scenario.robot.right_arm_hand.reach_joint_target

            self.set_policy_action_func = self.scenario.set_policy_action
            self.reach_joint_target_func = self.scenario.robot.left_arm_hand.reach_joint_target

        # 训练的时候不需要考虑 done  
        # self.env_args = {"collect_data":False,"noise":False,"random_init":True}
        
        # 尝试设置 鲁棒性
        self.env_args = {"collect_data":False,"noise":True,"random_init":True}

    def launch_scenario(self):
        # from dexrl.sim.scenarios_dual_arm.rl_dual_arm import RLDualArmScenario, RLDualArmScenarioCfg
        # self.scenario:RLDualArmScenario = RLDualArmScenario(RLDualArmScenarioCfg())
        
        # from dexrl.sim.scenarios.rl_insert_lego_sim_3act import RLInsertLegoScenario, RLInsertLegoScenarioCfg
        # self.scenario:RLInsertLegoScenario = RLInsertLegoScenario(RLInsertLegoScenarioCfg())
        
        # from dexrl.sim.scenarios.rl_grasp_5act_sim import RLGraspScenario, RLGraspScenarioCfg
        # self.scenario:RLGraspScenario = RLGraspScenario(RLGraspScenarioCfg())
        
        # from dexrl.sim.scenarios.rl_grasp_4act_sim import RLGrasp4Scenario, RLGrasp4ScenarioCfg
        # self.scenario:RLGrasp4Scenario = RLGrasp4Scenario(RLGrasp4ScenarioCfg())

        from dexrl.sim.scenarios.rl_grasp_4act_left_sim import RLGrasp4LeftScenario, RLGrasp4LeftScenarioCfg
        self.scenario:RLGrasp4LeftScenario = RLGrasp4LeftScenario(RLGrasp4LeftScenarioCfg())

    def update_app(self,action):
        env_step = 0
        action_valid,_ = self.set_policy_action_func(action)

        if action_valid:
            for i in range(1):  # 从1改为4，增加仿真更新频率
                # self.scenario.step(action)
                current_pos, success = self.reach_joint_target_func()
                # current_pos, success = self.scenario.robot.reach_joint_target(left=True)
                # if current_pos is not None:
                #     print(f"current_pos: {current_pos}, success: {success}")
                self.sim_app.update()
        else:
            print("reach_joint_target failed")

        results = self.scenario.get_rl_tuples()

        return results
 

    def print_obs(self,obs):
        # pass
        if self.action_space.shape[0] == 4:
            print_mix_pose(obs[:6])
            print_yellow(f"hand_angles_1: {obs[6:9]}")
            print_yellow(f"current_action_state: {obs[9:13]}")
            print_yellow(f"obj_pos: {obs[13:16]}")
            print_yellow(f"bbox: {obs[16:22]}")
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
        x_offset,y_offset,obj_pos = self.scenario.reset()
        # 减少初始化时的仿真更新次数
        for i in range(10):  # 从20减少 到10
            self.sim_app.update()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        self.task_stage = self.scenario.task_stage #????
        # self.last_real_action = np.array([0,0,0,   -3.1, 2.7 , -3.2   ,0,0,0])
        return self.current_observation, {"obj_pos": obj_pos}, {"x_offset": x_offset, "y_offset": y_offset}

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
    observation, info,_ = env.reset()

    # env.set_mode("right")

    # for i in tqdm.tqdm(range(10000)):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         env.reset()
    

    current_replay_action_index = 0
    current_replay_action_step = 0

    transition_data_list = []
    transition_data_num = 0
    max_transition_data_num = 50
            
    # rot = -1
    # y_euler = 1.57 # ok
    # z_euler = -0.6
    # [-1.00,1.57,-0.60]
    
    x_offset_plus = 0.05
    y_offset_plus = 0.01 #0.05
    # x_offset_plus = -0.1
    # y_offset_plus = 0.02
    # approach_distance = 0.05

    while transition_data_num < max_transition_data_num:
        
        _,_,info= env.reset()
        x_offset = info["x_offset"]
        y_offset = info["y_offset"]
        obj_pos = env.scenario.obj.get_world_pose()[0]
        
        env.scenario.policy_steps = 0
        # 空转一段时间
        for i in range(10):
            env.sim_app.update()
        observation = env.scenario.get_rl_tuples()[0]
        
        
        
        action_list = [
            [0.35+x_offset+x_offset_plus,-0.29+y_offset+y_offset_plus,1.02,          1],
            [0.28+x_offset+x_offset_plus,-0.34+y_offset+y_offset_plus,1.02,          1],
            # [0.2+x_offset+x_offset_plus,-0.34+y_offset+y_offset_plus,1.02,          1],
            [0.2+x_offset+x_offset_plus,-0.34+y_offset+y_offset_plus,1.02,         0.5],
            # [0.2+x_offset+x_offset_plus,-0.34+y_offset+y_offset_plus,1.12,         0.5]
            # [-0.2+x_offset+x_offset_plus,-0.34+y_offset+y_offset_plus,1.35,         -0.7]
        ] 
        
        
        # self.action_list = [
        #     [obj_pos[0] + x_offset_plus - 0.12, obj_pos[1] + y_offset_plus, 1, 1, 1],
        #     [obj_pos[0] + x_offset_plus - 0.08, obj_pos[1] + y_offset_plus, 1, 1, 1],
        #     [obj_pos[0] + x_offset_plus + approach_distance, obj_pos[1] + y_offset_plus, 1.02, 1, 1],
        #     [obj_pos[0] + x_offset_plus + approach_distance, obj_pos[1] + y_offset_plus, 1.03, -0.7, 0.2],
        #     [obj_pos[0] + x_offset_plus + approach_distance, obj_pos[1] + y_offset_plus, 1.35, -0.7, 0.2]
        # ]
        action_length = len(action_list)
        
        
        current_transition_data_ls = []
        done = False
        truncated = False
        current_replay_action_index = 0
        current_replay_action_step = 0
        env.scenario.policy_steps = 0
        # print(f"self.action_length: {self.action_length}")

        last_mixed_pose = env.scenario.robot.left_arm_hand.arm.get_mixed_pose()
        last_mixed_pos = last_mixed_pose[:3]
        last_hand_joint_1 = env.scenario.robot.left_arm_hand.hand.get_hand_joint_policy_3()[:1]
        stable_step = 0
        
        while current_replay_action_index < action_length:

            if current_replay_action_step == 0:
                rule_action_4_target = np.array(action_list[current_replay_action_index])
                print(f"rule_action_4_target: {rule_action_4_target}")

            # current_action_5 = self.scenario.robot.right_arm_hand.get_real_action_5()
            current_action_4 = env.scenario.robot.left_arm_hand.get_real_action_4() 
            current_mixed_pos = current_action_4[:3]
            current_hand_joint_1 = current_action_4[3:4]

            delta_action_4 = rule_action_4_target - current_action_4

            reach_threshold =  0.1 #0.01 # 太小会卡主

            action_step_ratio = np.random.rand()*0.5 + 0.5
            # action_step_ratio = np.random.rand()
            
            tmp_max = env.scenario.cfg.policy_act_upper_limit*reach_threshold
            
            if np.all(np.abs(delta_action_4) < tmp_max):
            # if np.all(delta_action_4 > self.scenario.cfg.policy_act_lower_limit*reach_threshold) or np.all(delta_action_4 < self.scenario.cfg.policy_act_upper_limit*reach_threshold):
                print_green(f"reach target ! steps: {current_replay_action_step}")
                current_replay_action_step = 0 # 到达目标后，换下一个动作
                current_replay_action_index += 1
                continue

            # 为什么会卡住，移动不过去？
            # if (np.all(np.abs(current_mixed_pos - last_mixed_pos) < 0.002) and \
            #     np.all(np.abs(current_hand_joint_1 - last_hand_joint_1) < 0.01)) or self.current_replay_action_step > 150:
            if (np.all(np.abs(current_mixed_pos - last_mixed_pos) < 0.002) and \
                np.all(np.abs(current_hand_joint_1 - last_hand_joint_1) < 0.03)) or current_replay_action_step > 150:

                stable_step += 1
                if stable_step > 10:
                    print(f"stucked! {current_replay_action_step}")
                    stable_step = 0
                    current_replay_action_step = 0 # 到达目标后，换下一个动作
                    current_replay_action_index += 1
                    continue
            else:
                stable_step = 0
                
            last_mixed_pos = current_mixed_pos
            last_hand_joint_1 = current_hand_joint_1
            current_replay_action_step += 1
            step_action = np.clip(delta_action_4 , # 认为是网络输出的 policy
                                    env.scenario.cfg.policy_act_lower_limit*action_step_ratio,
                                    env.scenario.cfg.policy_act_upper_limit*action_step_ratio)
            # print(f"self.scenario.cfg.policy_act_lower_limit: {self.scenario.cfg.policy_act_lower_limit}")
            # print(f"self.scenario.cfg.policy_act_upper_limit: {self.scenario.cfg.policy_act_upper_limit}")
            
            step_action = env.scenario.policy_action_normalizer.normalize(step_action) # 成 -1 到 1 之间


            last_observation = env.scenario.get_observation()
            # 每隔3步放进去一次
            ########################################################## 

            # action_valid,real_action = env.scenario.set_policy_action(step_action) 
            # current_pos, status = env.scenario.robot.left_arm_hand.reach_joint_target()
            env.step(step_action)
                
            observation, reward, done, truncated, info = env.scenario.get_rl_tuples()
            print_green(f"policy step {env.scenario.policy_steps}, reward: {reward}, mixed_pos: {current_mixed_pos}")

            # 记录当前数据
            transition_data = dict(
                observations=None,
                actions=None,
                next_observations=None,
                rewards=None,
                masks=None,
                dones=None,
            )

            # joint 达到目标状态视为一个 policy step，在此记录数据
            transition_data["observations"] = last_observation
            transition_data["actions"] = step_action
            transition_data["rewards"] = reward
            transition_data["next_observations"] = observation
            transition_data["masks"] = 1.0 - done
            transition_data["dones"] = done or truncated
            current_transition_data_ls.append(transition_data)

            last_observation = observation

            # print(f"action: {step_action}")
            # print_green(f"reward: {reward}, step: {self.scenario.policy_steps}")
            
            # if done:
            #     break

        # 判断是否成功,成功了才把数据收集下来
        if done:
            transition_data_list.extend(current_transition_data_ls)
            transition_data_num += 1
            print_green(f"done: {done}, truncated: {truncated}")
            print_green(f"transition_data_num: {transition_data_num}")
        if truncated:
            print_green(f"truncated: {truncated}")

    env.close()