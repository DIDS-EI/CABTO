import os
os.environ["SIM_MODE"] = "Standalone"
from dexrl._cfg import DexrlCfg
from dexrl.envs.app_launcher import AppLauncher

import gymnasium as gym
import numpy as np
import torch as th

from dexrl.envs._cfg import EnvCfg

class Env(gym.Env):
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))

    def __init__(self, cfg: EnvCfg=None, launch_app:bool=True):
        self.cfg = cfg
        self.sim_app = None
        if launch_app:
            self.launch_app()

    def launch_app(self):
        if self.sim_app : return self.sim_app
        
        self.sim_app = AppLauncher(
            # headless=True,
            # hide_ui=True,
            backend="numpy",device=self.cfg.device).app
        
        from omni.isaac.core import World
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import Sdf, UsdLux,UsdGeom

        

        world:World = World(stage_units_in_meters=1.0,backend="numpy",device=self.cfg.device)
        world.reset()

        self.launch_scenario()

        sphereLight:UsdLux.SphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(50000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([-6.5, 0, 12])
        # XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

        self.scenario.set_world(world)
        
        self.scenario.load_assets()

        for i in range(20):
            self.sim_app.update()

        self.scenario.robot.articulation.initialize()
        self.scenario.setup()

    def launch_scenario(self):
        from dexrl.sim.scenarios.grasp import GraspScenario, GraspScenarioCfg

        self.scenario:GraspScenario = GraspScenario(GraspScenarioCfg())

    def update_app(self,action):
        action_valid = self.scenario.set_policy_action(action)

        if action_valid:
            
            success = False
            while not success:
                current_pos, success = self.scenario.reach_joint_target()

        # self.scenario.set_ik_target_real(current_pos)

        # for i in range(20):  # 从1改为4，增加仿真更新频率
        # if not success:
        #     print("reach_joint_target failed")
        #     return None, None, None, None, None
        
        # for i in range(20):  # 从1改为4，增加仿真更新频率
        #     self.scenario.step(action)
                self.sim_app.update()

        results = self.scenario.get_rl_tuples()

        return results

    def reset(self, seed=None, options=None):
        self.scenario.reset()
        return np.zeros(self.observation_space.shape[0]), {}
    
    def step(self, action):
        # action = th.tensor(action)
        results = self.update_app(action)
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
    env.reset()

    for i in tqdm.tqdm(range(1000000)):
        action = env.action_space.sample()
        # action = action.reshape(1,-1)
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        # print(env.scenario.get_sim_info())
        # break

        if i % 100 == 0:
            env.reset()
        
        # image = env.scenario.get_camera_rgb()
        # print(image.shape)
        # cv2.imshow("image", image)
        # cv2.waitKey(1)

    env.close()