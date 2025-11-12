import os
os.environ["SIM_MODE"] = "Standalone"
# os.environ["EXP_PATH"] = "/home/admin01/isaacsim/apps"
# os.environ["CARB_APP_PATH"] = "/home/admin01/isaacsim"
from dexrl._cfg import DexrlCfg
from dexrl.envs.app_launcher import AppLauncher

import gymnasium as gym
import numpy as np
import torch as th

from dexrl.envs._cfg import EnvCfg

class Env(gym.Env):
    # # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(26,)) # eef pose 3+4 + 9 动作 + 3+4目标
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,)) # 6+6+9+3+6+1
    # # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,)) # 01 reward
    # action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))
    
    # 饼干
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(26,)) #20 3+2+ 3+6 + 1
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,)) # stage
    action_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
    
    # 牛奶
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(26,)) 
    # action_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
    
    # 积木
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,)) # 3+2+ 3+6 + 1
    # action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    
    # 双手
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(26,)) 
    # action_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    
    # tree
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(44,)) # q
    # action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))
    
    
    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True, headless:bool=False):
        self.cfg = cfg
        self.sim_app = None
        self.scenario=None
            
        # self.last_real_action = None
        self.current_observation = None
        
        # stage rl
        self.task_stage = None
        self.task_stage_count = None
        # 如果存在task_reset_script方法才赋值
        self.task_reset_script_generator = None

        if hasattr(self.scenario, 'task_reset_script') and \
            callable(getattr(self.scenario, 'task_reset_script')):
            self.task_reset_script_generator = self.scenario.task_reset_script()
        if hasattr(self.scenario, 'task_stage'):
            self.task_stage = self.scenario.task_stage
            self.task_stage_count = self.scenario.task_stage_count
            
        # 训练的时候不需要考虑 done  
        self.env_args = {"collect_data":False,"noise":False,"random_init":True}
       
        if launch_app:
            self.launch_app(headless)
        
       
        
    def launch_app(self,headless:bool=False):
        if self.sim_app : return self.sim_app
        if self.cfg == None:
            self.cfg = EnvCfg(device="cpu")
        self.sim_app = AppLauncher(
            headless=headless,
            # hide_ui=True,
            backend="numpy",
            device=self.cfg.device,
            physics_dt=1.0/60.0,  # 降低物理仿真频率
            rendering_dt=1.0/30.0,  # 降低渲染频率
            physics_substeps=1,  # 减少物理子步数
            render_resolution=(640, 480),  # 降低渲染分辨率
            ).app
        
        from omni.isaac.core import World
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import Sdf, UsdLux

        world:World = World(stage_units_in_meters=1.0,backend="numpy",device=self.cfg.device)
        # world:World = World(stage_units_in_meters=1.0,device=self.cfg.device)
        world.reset()

        self.launch_scenario()

        sphereLight:UsdLux.SphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([0, -6.5, 12])
        # XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

        self.scenario.set_world(world)
        
        self.scenario.load_assets()

        for i in range(20):
            self.sim_app.update()

        self.scenario.robot.articulation.initialize()
        self.scenario.setup()
        
        if hasattr(self.scenario, 'cfg') and hasattr(self.scenario.cfg, 'env_args'):
            self.scenario.cfg.env_args = self.env_args

    def launch_scenario(self):
        # from dexrl.sim.scenarios.rl_grasp_sim import RLGraspScenario, RLGraspScenarioCfg
        # self.scenario:RLGraspScenario = RLGraspScenario(RLGraspScenarioCfg())
        
        # from dexrl.sim.scenarios.rl_grasp_biscuit_sim import RLGraspBiscuitScenario, RLGraspBiscuitScenarioCfg
        # self.scenario:RLGraspBiscuitScenario = RLGraspBiscuitScenario(RLGraspBiscuitScenarioCfg())
        
        from dexrl.sim.scenarios.rl_grasp_biscuit_5act_sim import BTRLGraspBiscuitScenario, BTRLGraspBiscuitScenarioCfg
        self.scenario:BTRLGraspBiscuitScenario = BTRLGraspBiscuitScenario(BTRLGraspBiscuitScenarioCfg())

        # from dexrl.sim.scenarios.rl_insert_lego_sim_3act import RLInsertLegoScenario, RLInsertLegoScenarioCfg
        # self.scenario:RLInsertLegoScenario = RLInsertLegoScenario(RLInsertLegoScenarioCfg())
        
        # from dexrl.sim.scenarios.rl_grasp_milk_5act_sim import BTRLGraspMilkScenario, BTRLGraspMilkScenarioCfg
        # self.scenario:BTRLGraspMilkScenario = BTRLGraspMilkScenario(BTRLGraspMilkScenarioCfg())
        
        # from dexrl.sim.scenarios.rl_grasp_milk_5act_sim_stage import RLGraspMilkStageScenario, RLGraspMilkStageScenarioCfg
        # self.scenario:RLGraspMilkStageScenario = RLGraspMilkStageScenario(RLGraspMilkStageScenarioCfg())

        # 双手
        # from dexrl.sim.scenarios.rl_bihands_10act_sim import RLBiHands10ActScenario, RLBiHands10ActScenarioCfg
        # self.scenario:RLBiHands10ActScenario = RLBiHands10ActScenario(RLBiHands10ActScenarioCfg())
        
        # ============= tree ============= 每个技能单独训练
        # from dexrl.sim.scenarios.tree_pickup_9act_sim import TreePickup9ActScenario, TreePickup9ActScenarioCfg
        # self.scenario:TreePickup9ActScenario = TreePickup9ActScenario(TreePickup9ActScenarioCfg())
        
        # from dexrl.sim.scenarios.tree_handover_9act_sim import TreeHandover9ActScenario, TreeHandover9ActScenarioCfg
        # self.scenario:TreeHandover9ActScenario = TreeHandover9ActScenario(TreeHandover9ActScenarioCfg())
        
        # from dexrl.sim.scenarios.tree_place_9act_sim import TreePlace9ActScenario, TreePlace9ActScenarioCfg
        # self.scenario:TreePlace9ActScenario = TreePlace9ActScenario(TreePlace9ActScenarioCfg())
        
        # from dexrl.sim.scenarios.tree_pour_9act_sim import TreePour9ActScenario, TreePour9ActScenarioCfg
        # self.scenario:TreePour9ActScenario = TreePour9ActScenario(TreePour9ActScenarioCfg())
        
        # ============= tree =============
        
        

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
        
    def reset(self):
        _,_,obj_pose = self.scenario.reset()
        # 减少初始化时的仿真更新次数
        for i in range(50):  # 从20减少到10
            self.sim_app.update()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        # self.last_real_action = np.array([0,0,0,   -3.1, 2.7 , -3.2   ,0,0,0])
        return self.current_observation, {"obj_pos": obj_pose[0]}

    def reset_and_set_obj_pose(self, obj_pos, obj_quat): # 数据重放的时候需要
        self.scenario.reset_and_set_obj_pose(obj_pos, obj_quat)
        for i in range(3):  # 从20减少到10
            self.sim_app.update()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        
        return self.current_observation, {"obj_pos": obj_pos}
    
    def step(self, action,noise=False):
        
        
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
        self.task_stage = self.scenario.task_stage
        self.task_stage_count = self.scenario.task_stage_count
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
        env.print_obs(observation)
        
        if terminated or truncated:
            env.reset()
        
        # print(env.scenario.get_sim_info())
        # break 
        # image = env.scenario.get_camera_rgb()
        # print(image.shape)
        # cv2.imshow("image", image)
        # cv2.waitKey(1)

    env.close()