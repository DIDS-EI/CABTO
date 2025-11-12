import os
os.environ["SIM_MODE"] = "Standalone"
# os.environ["EXP_PATH"] = "/home/admin01/isaacsim/apps"
# os.environ["CARB_APP_PATH"] = "/home/admin01/isaacsim"
from dexrl._cfg import DexrlCfg
from dexrl.envs.app_launcher import AppLauncher

import gymnasium as gym
import numpy as np
import torch as th
from enum import Enum

from dexrl.envs._cfg import EnvCfg
# from dexrl.sim.scenarios_tree.tree_pickup_gc_9act_sim import ScenarioType

class ScenarioType(Enum):
    PICKUP = "pickup"
    HANDOVER = "handover"
    PLACE = "place"
    POUR = "pour"

class Env(gym.Env):
    
    # tree
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(44,)) # q
    # observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,)) # 3+6+3+6
    action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))
    
    
    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True, headless:bool=False, \
                 scenario_type:ScenarioType=ScenarioType.HANDOVER, task_start_stage:int=1,
                 env_args:dict=None):
        self.cfg = cfg
        self.sim_app = None
        self.scenario=None
        self.scenario_type = scenario_type  # 场景类型: ScenarioType 枚举
        self.task_start_stage = task_start_stage  # 开始阶段: 1 或 2
            
        # self.last_real_action = None
        self.current_observation = None
        
        # stage rl
        self.task_stage = None
        self.task_stage_count = None
        # 如果存在task_reset_script方法才赋值
        self.task_reset_script_generator = None
            
        # 训练的时候不需要考虑 done  
        if env_args is None:
            self.env_args = {"collect_data":False,"noise":False,"random_init":True}
        else:
            self.env_args = env_args
       
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
        
        # if self.scenario_type == ScenarioType.PICKUP:
        #     from dexrl.sim.scenarios.tree_pickup_9act_sim import TreePickup9ActScenario, TreePickup9ActScenarioCfg
        #     self.scenario:TreePickup9ActScenario = TreePickup9ActScenario(TreePickup9ActScenarioCfg())
        # elif self.scenario_type == ScenarioType.HANDOVER:
        #     from dexrl.sim.scenarios.tree_handover_9act_sim import TreeHandover9ActScenario, TreeHandover9ActScenarioCfg
        #     self.scenario:TreeHandover9ActScenario = TreeHandover9ActScenario(TreeHandover9ActScenarioCfg())
        # elif self.scenario_type == ScenarioType.PLACE:
        #     from dexrl.sim.scenarios.tree_place_9act_sim import TreePlace9ActScenario, TreePlace9ActScenarioCfg
        #     self.scenario:TreePlace9ActScenario = TreePlace9ActScenario(TreePlace9ActScenarioCfg())
        # elif self.scenario_type == ScenarioType.POUR:
        #     from dexrl.sim.scenarios.tree_pour_9act_sim import TreePour9ActScenario, TreePour9ActScenarioCfg
        #     self.scenario:TreePour9ActScenario = TreePour9ActScenario(TreePour9ActScenarioCfg())
        # else:
        #     raise ValueError(f"Invalid scenario type: {self.scenario_type}")
        
        
        
        # ============= tree =============
        
        if self.scenario_type == ScenarioType.HANDOVER:
            from dexrl.sim.scenarios_tree.tree_handover_9act_sim import TreeHandover9ActScenario, TreeHandover9ActScenarioCfg
            self.scenario:TreeHandover9ActScenario = TreeHandover9ActScenario(TreeHandover9ActScenarioCfg())
        elif self.scenario_type == ScenarioType.PLACE:
            from dexrl.sim.scenarios_tree.tree_place_9act_sim import TreePlace9ActScenario, TreePlace9ActScenarioCfg
            self.scenario:TreePlace9ActScenario = TreePlace9ActScenario(TreePlace9ActScenarioCfg())
        elif self.scenario_type == ScenarioType.PICKUP:
            from dexrl.sim.scenarios_tree.tree_pickup_gc_9act_sim import TreePickupGc9ActScenario, TreePickupGc9ActScenarioCfg
            self.scenario:TreePickupGc9ActScenario = TreePickupGc9ActScenario(TreePickupGc9ActScenarioCfg())
        elif self.scenario_type == ScenarioType.POUR:
            from dexrl.sim.scenarios_tree.tree_pour_9act_sim import TreePour9ActScenario, TreePour9ActScenarioCfg
            self.scenario:TreePour9ActScenario = TreePour9ActScenario(TreePour9ActScenarioCfg())
        else:
            raise ValueError(f"Invalid scenario type: {self.scenario_type}")
        
        # 设置场景配置
        if hasattr(self.scenario, 'cfg'):
            self.scenario.cfg.task_start_stage = self.task_start_stage
        
        
        # 使用统一的tree_9act_sim.py
        # from dexrl.sim.scenarios.tree_9act_sim import Tree9ActScenario, Tree9ActScenarioCfg
    
        
        # 创建配置
        # cfg = Tree9ActScenarioCfg()
        # cfg.scenario_type = self.scenario_type
        
        # 设置开始阶段
        # if hasattr(cfg, 'task_start_stage'):
        #     cfg.task_start_stage = self.task_start_stage
        
        # 创建场景
        # self.scenario: Tree9ActScenario = Tree9ActScenario(cfg)
        
        # ============= tree =============
        
        

    def update_app(self,action):
        env_step = 0
        action_valid,_ = self.scenario.set_policy_action(action)

        if action_valid:

            for i in range(1):  # 从1改为4，增加仿真更新频率
                # self.scenario.step(action)
                current_pos, success = self.scenario.robot.right_arm_hand.reach_joint_target()
                self.sim_app.update()
        
        else:
            print("reach_joint_target failed")

        results = self.scenario.get_rl_tuples()

        return results

    
    def execute_task_reset_script(self):
        def generate_task_reset_script():
            """执行任务重置脚本，如果存在的话"""  
            if hasattr(self.scenario, 'task_reset_script') and \
            callable(getattr(self.scenario, 'task_reset_script')):
                self.task_reset_script_generator = self.scenario.task_reset_script()
        
        if self.task_reset_script_generator is None:
            generate_task_reset_script()
        try:
            if self.task_reset_script_generator is not None:
                # 循环执行所有步骤直到生成器耗尽
                for result in self.task_reset_script_generator:
                    self.sim_app.update()
                self.task_reset_script_generator = None
        except StopIteration:
            return True

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
        _,_,obj_pose = self.scenario.reset(scenario_type=self.scenario_type, stage=self.task_start_stage)
        # 减少初始化时的仿真更新次数
        for i in range(50):  # 从20减少到10
            self.sim_app.update()
        self.execute_task_reset_script()
        results = self.scenario.get_rl_tuples()
        self.scenario.env_steps = 0
        self.current_observation = results[0]
        self.task_stage = self.scenario.task_stage
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
        action = th.tensor(action)
        results = self.update_app(action)
        tuple_args = {"collect_data":True,"noise":noise}
        results = self.scenario.get_rl_tuples(**tuple_args)
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

    # 传入参数，现在是哪个 sim
    # scenario_type = ScenarioType.PICKUP  
    scenario_type = ScenarioType.HANDOVER
    # 可选: ScenarioType.PICKUP, ScenarioType.HANDOVER, ScenarioType.PLACE, ScenarioType.POUR
    
    task_start_stage = 2
    # 可选: 0 或 1；0 代表 没有靠近；1 代表 靠近但没有抬起；2 代表 靠近并抬起
    # 通常选择 1 (一起训) 或者 2 (独立技能)
    # PICKUP 本身没有 2，只有 0  或者 1 (靠近,独立技能)
    
    # 设置这个 sim 开始的阶段是 1 还是 2 并进行设置
    env = Env(scenario_type=scenario_type, task_start_stage=task_start_stage)

    observation, info = env.reset()

    for i in tqdm.tqdm(range(1000000)):
        
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.print_obs(observation)
        
        if terminated or truncated:
            env.reset()
    env.close()