import numpy as np
from typing import Dict

from omni.isaac.core.world import World

from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
from omni.isaac.motion_generation import ArticulationMotionPolicy
import omni.replicator.core as rep
from dexrl.sim.objects import BaseObject, Socket, MySocket, Apple_plug

from dexrl.sim.scenarios.franka import Robot, FrankaScenario, FrankaScenarioCfg
from dexrl.sim.utils import Normalizer_N1_1
from dexrl.sim.scenarios._cfg import ScenarioCfg
from dexrl.sim import utils
from dexrl.utils.configclass import configclass
import pxr.Usd


@configclass
class RLFrankaScenarioCfg(FrankaScenarioCfg):
    limit_range = 0.05  # 0.02 #0.05
    policy_act_lower_limit = np.array([
        -limit_range, -limit_range, -limit_range])
    policy_act_upper_limit = np.array([
        limit_range, limit_range, limit_range])

    real_act_ik_lower_limit = np.array([-0.4, -0.6, 0.1])
    real_act_ik_upper_limit = np.array([0.4, -0.3, 0.6])

    obj_cls = Apple_plug
    obj_init_pos = [0, -0.40253, 0.17164]  # [[-0.12822, -0.63934, 0.17668]]
    obj_init_euler = [0, 0, 0]  # [0, 1.57, 0]
    obj_mass = 0.2

    obj_cls2 = Socket
    obj_init_pos2 = [0, -0.6, 0.07]  # -0.0.35
    obj_init_euler2 = [0, 0, 0]  # [0, 1.57, 0]
    obj_mass2 = 0.2

    obj_cls3 = MySocket
    obj_init_pos3 = [0, 0, 0.07]  # -0.0.35
    obj_init_euler3 = [0, 0, 0]  # [0, 1.57, 0]
    obj_mass3 = 0.2


class RLFrankaScenario(FrankaScenario):
    cfg_cls = RLFrankaScenarioCfg  # 去掉类型声明
    cfg: RLFrankaScenarioCfg
    robot_cls: Robot = Robot
    world: World
    stage: pxr.Usd.Stage

    policy_steps = 0

    def __init__(self, cfg: ScenarioCfg):
        super().__init__(cfg)

        self.policy_action_normalizer = Normalizer_N1_1(
            self.cfg.policy_act_lower_limit, self.cfg.policy_act_upper_limit)

    def load_objects(self):
        self.obstacle_list = []
        self.table = FixedCuboid(
                name="table",
                prim_path="/World/objects/obstacle_1",
                scale=np.array([1.2, 0.6, 0.05]),
                position=np.array([0., -0.5, 0.05]),
                color=np.array([0.05, 0.05, 0.05]),
            )
        self.world.scene.add(self.table)
        self.obstacle_list.append(self.table)

        self.red_cube = FixedCuboid(
            name="RedCube",
            position=np.array([-0.043, -0.4, 0.1]),  # np.array([0,-0.4, 0.1]),
            prim_path="/World/objects/red_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
            # mass=50,
        )
        self.world.scene.add(self.red_cube)
        self.obstacle_list.append(self.red_cube)

        self.green_cube = FixedCuboid(
            name="GreenCube",
            position=np.array((0.043, -0.4, 0.1)),
            prim_path="/World/objects/green_cube",
            size=0.05,
            color=np.array([1, 1, 0]),
            # mass=50,
        )
        self.world.scene.add(self.green_cube)
        self.obstacle_list.append(self.green_cube)

        self.blue_cube = DynamicCuboid(
            name="BlueCube",
            position=np.array((0.4, -0.4, 0.1)),
            prim_path="/World/objects/blue_cube",
            size=0.001,
            color=np.array([0, 0, 1]),
        )
        self.world.scene.add(self.blue_cube)
        self.obstacle_list.append(self.blue_cube)

        self.obj_map: Dict[str, DynamicCuboid] = {
            "red_cube": self.red_cube,
            "blue_cube": self.blue_cube,
            "green_cube": self.green_cube,
        }
        for obj in self.obstacle_list:
            with rep.get.prims(obj.prim_path):
                rep.modify.semantics([("class", obj.name)])
        self.plug: BaseObject = Apple_plug(
            position=np.array(self.cfg.obj_init_pos),
            orientation=utils.rot.euler_angles_to_quat(
                self.cfg.obj_init_euler),
            mass=self.cfg.obj_mass,
            )
        self.plug_prim = self.plug.create_prim()
        self.world.scene.add(self.plug_prim)
        self.obstacle_list.append(self.plug_prim)

        self.socket: BaseObject = Socket(
            position=np.array(self.cfg.obj_init_pos2),
            orientation=utils.rot.euler_angles_to_quat(
                self.cfg.obj_init_euler2),
            mass=self.cfg.obj_mass2,
            )
        self.socket_prim = self.socket.create_prim()
        self.world.scene.add(self.socket_prim)
        self.obstacle_list.append(self.socket_prim)

        self.my_socket: BaseObject = MySocket(
            position=np.array(self.cfg.obj_init_pos3),
            orientation=utils.rot.euler_angles_to_quat(
                self.cfg.obj_init_euler3),
            mass=self.cfg.obj_mass3,
            )
        self.my_socket_prim = self.my_socket.create_prim()
        self.world.scene.add(self.my_socket_prim)
        self.obstacle_list.append(self.my_socket_prim)

    def reset(self):
        self.policy_steps = 0
        return

    def update_state_buffers(self):
        pass

    def get_observation(self):
        return np.zeros(22)

    def set_policy_action(self, action):
        # 前三个是位置，后面一个手的开关
        pos = np.array(self.policy_action_normalizer.denormalize(action[:3]))
        quat = utils.rot.euler_angles_to_quat([0, np.pi, 0])

        # 获取当前末端位置
        # current_pos = self.robot.get_mixed_pose_joint()[:3]
        current_pos = self.robot.get_eef_pose_rmpflow()[0]
        next_pos = current_pos + pos
        next_pos = np.clip(next_pos, self.cfg.real_act_ik_lower_limit[:3],
                           self.cfg.real_act_ik_upper_limit[:3])

        # next_pos = np.array([0,-0.4,0.1])
        # self.goto_position_no_yield(next_pos,quat,self.robot.articulation,self.robot.rmpflow)

        # 使用非阻塞的方式，只执行一步移动
        self.robot.rmpflow.set_end_effector_target(
            np.array([-next_pos[1], next_pos[0], next_pos[2]]),
            quat
        )

        # 执行一步RMPflow动作
        articulation_motion_policy = ArticulationMotionPolicy(
            self.robot.articulation, self.robot.rmpflow, 1/20)
        self.robot.rmpflow.update_world()
        action_ik = articulation_motion_policy.get_next_articulation_action(
            1/20)
        self.robot.articulation.apply_action(action_ik)

        if action[3] > 0.5:
            self.open_gripper()
        else:
            self.close_gripper()

    def get_reward(self):
        return 0

    def get_terminated(self):
        return False

    def get_truncated(self):
        return False

    def get_rl_tuples(self):
        self.update_state_buffers()

        observation = self.get_observation()
        reward = self.get_reward()
        terminated = self.get_terminated()
        truncated = self.get_truncated()
        if terminated or truncated:
            infos = {
                "final_observation": observation,
                "sim_episode": {
                    # "r": self.accumulated_reward,
                    "r": reward,
                    "l": self.policy_steps
                }
            }
        else:
            infos = {}
        return observation, reward, terminated, truncated, infos

    def last_return(self):
        observation = self.get_observation()
        reward = self.get_reward()
        self.infos = {"final_observation": [observation],
                      "final_info": [{"sim_episode": {
                          "r": reward,
                          "l": self.policy_steps
                      }}]}
        return True
