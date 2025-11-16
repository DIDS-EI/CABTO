from typing import Dict, List, Optional, Type

import numpy as np
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.world import World
from dexrl.sim.objects import *
from dexrl.sim.scenarios._cfg import ScenarioCfg
from dexrl.sim.scenarios.franka import FrankaScenario, FrankaScenarioCfg, Robot
from dexrl.utils.configclass import configclass
from dexrl.sim.objects import BigLego, Carton, BlueCup, Shrimp, Potato, Apple
from dexrl.sim import utils


class OneFrankaScenario(FrankaScenario):
    """单臂 Franka 场景，单个机器人在原点。"""

    cfg_cls: Type[FrankaScenarioCfg] = FrankaScenarioCfg
    cfg: FrankaScenarioCfg
    robot_cls: Type[Robot] = Robot
    world: World
    stage: object

    def __init__(self, cfg: ScenarioCfg):
        super().__init__(cfg)

    def load_robot(self):
        # 机器人在原点 (0, 0, 0)
        translation = (0.0, 0.0, 0.0)

        self.robot = self.robot_cls(self.cfg, prim_path="/World/Franka")
        self.robot.xform_prim.set_world_pose(
            position=translation,
        )
        self.world.scene.add(self.robot.articulation)  # type: ignore[arg-type]

    def setup(self):
        self.setup_viewport_camera()
        self.setup_camera()
        self.robot.setup()
        self.reset()





####################
# cover
#####################
class CoverScenarioCfg(FrankaScenarioCfg):
    # myx input
    obj_cls5 = Boiler # Boiler  E13
    obj_init_pos5 = [-0.3,-0.4,0.1] #[0.2, -0.45, 0.96]
    obj_init_euler5 = [3.14, 0, 0] # ?
    obj_mass5 = 3  # ?

    obj_cls6 = ChopBoard # ChopBoard E10
    obj_init_pos6 = [0,-0.4,0.1] #[0.2, -0.45, 0.96]
    obj_init_euler6 = [1.57, 0, 0] # ?
    obj_mass6 = 3  # ?

    # Bowl，已有？
    obj_cls7 = Bowl # Bowl E31
    obj_init_pos7 = [0.3,-0.45,0.1] #[0.2, -0.45, 0.96]
    obj_init_euler7 = [0, 0, 0] # ?
    obj_mass7 = 3  # ?


    # 三个要抓起的物体换成 
    obj_cls8 = Shrimp # Apple omzprq
    obj_init_pos8 = [-0.3,-0.4,0.1] #[0.2, -0.45, 0.96]
    obj_init_euler8 = [0, 0, 0] # ?
    obj_mass8 = 3  # ?

    obj_cls9 = Potato # Shrimp mnpgev
    obj_init_pos9 = [0,-0.45,0.1] #[0.2, -0.45, 0.96]
    obj_init_euler9 = [0, 0, 0] # ?
    obj_mass9 = 3  # ?

    obj_cls10 = Apple # Potato bpwohr
    obj_init_pos10 = [0.3,-0.4,0.1] #[0.2, -0.45, 0.96]
    obj_init_euler10 = [0, 0, 0] # ?
    obj_mass10 = 3  # ?



class CoverScenario(FrankaScenario):
    """单臂 Franka 场景，单个机器人在原点。"""

    cfg_cls: Type[CoverScenarioCfg] = CoverScenarioCfg
    cfg: CoverScenarioCfg
    robot_cls: Type[Robot] = Robot
    world: World
    stage: object

    def __init__(self, cfg: ScenarioCfg):
        super().__init__(cfg)

    def load_robot(self):
        # 机器人在原点 (0, 0, 0)
        translation = (0.0, 0.0, 0.0)

        self.robot = self.robot_cls(self.cfg, prim_path="/World/Franka")
        self.robot.xform_prim.set_world_pose(
            position=translation,
        )
        self.world.scene.add(self.robot.articulation)  # type: ignore[arg-type]

    def load_objects(self):
        
        self.obstacle_list = []
        self.table = FixedCuboid(
                name="table",
                prim_path="/World/objects/obstacle_1",
                scale=np.array([0.9, 0.5, 0.05]),
                position=np.array([0.,-0.5, 0.05]),
                color=np.array([0.08, 0.08, 0.08]),
            )
        self.world.scene.add(self.table)
        self.obstacle_list.append(self.table)
        
        # 摆放三个小方块
        self.obj8 = Shrimp(
            position=np.array([-0.3,-0.7,0.1]),
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler8),
            mass=self.cfg.obj_mass8,
            name="obj8",
        )
        self.obj8_prim = self.obj8.create_prim()
        self.world.scene.add(self.obj8_prim)
        self.task_obj_list.append(self.obj8)

        self.obj9 = Potato(
            position=np.array([0,-0.7,0.1]),
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler9),
            mass=self.cfg.obj_mass9,
            name="obj9",
        )
        self.obj9_prim = self.obj9.create_prim()
        self.world.scene.add(self.obj9_prim)
        self.task_obj_list.append(self.obj9)

        self.obj10 = Apple(
            position=np.array([0.3,-0.7,0.1]),
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler10),
            mass=self.cfg.obj_mass10,
            name="obj10",
        )
        self.obj10_prim = self.obj10.create_prim()
        self.world.scene.add(self.obj10_prim)
        self.task_obj_list.append(self.obj10)

        self.plate1 = self.cfg.obj_cls5(
            position=self.cfg.obj_init_pos5,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler5),
            mass=self.cfg.obj_mass5,
            name="obj5",
        )
        self.plate1_prim = self.plate1.create_prim()
        self.world.scene.add(self.plate1_prim)
        self.task_obj_list.append(self.plate1)

        self.plate2 = self.cfg.obj_cls6(
            position=self.cfg.obj_init_pos6,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler6),
            mass=self.cfg.obj_mass6,
            name="obj6",
        )
        self.plate2_prim = self.plate2.create_prim()
        self.world.scene.add(self.plate2_prim)
        self.task_obj_list.append(self.plate2)

        self.plate3 = self.cfg.obj_cls7(
            position=self.cfg.obj_init_pos7,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler7),
            mass=self.cfg.obj_mass7,
            name="obj7",
        )
        self.plate3_prim = self.plate3.create_prim()
        self.world.scene.add(self.plate3_prim)
        self.task_obj_list.append(self.plate3)

        # 摆放三个底座
        # self.plate1 = DynamicCuboid(
        #     name="plate1",  # 粉色，横竖
        #     prim_path="/World/targets/plate1",
        #     position=np.array([-0.3,-0.4,0.1]),
        #     size=0.02,
        #     color=np.array([0.5, 0.2, 0.2], dtype=float), # 粉色
        # )
        # self.world.scene.add(self.plate1)
        # self.obstacle_list.append(self.plate1)

        # self.plate2 = DynamicCuboid(
        #     name="plate2",  # 绿色
        #     prim_path="/World/targets/plate2",
        #     position=np.array([0,-0.4,0.1]),
        #     size=0.02,
        #     color=np.array([0.4, 0.6, 0.3], dtype=float), # 绿色
        # )
        # self.world.scene.add(self.plate2)
        # self.obstacle_list.append(self.plate2)

        # self.plate3 = DynamicCuboid(
        #     name="plate3",  # 紫色
        #     prim_path="/World/targets/plate3",
        #     position=np.array([0.3,-0.4,0.1]),
        #     size=0.02,
        #     color=np.array([0.2, 0.1, 0.7], dtype=float), # 紫色
        # )
        # self.world.scene.add(self.plate3)
        # self.obstacle_list.append(self.plate3)

        # self.obj_map: Dict[str, DynamicCuboid] = {
        #     "cube1": self.obj8,
        #     "cube2": self.obj9,
        #     "cube3": self.obj10,
        #     "plate1": self.plate1,
        #     "plate2": self.plate2,
        #     "plate3": self.plate3,
        # }

    def setup(self):
        self.setup_viewport_camera()
        self.setup_camera()
        self.robot.setup()
        self.reset()



####################
# BlocksScenario
#####################
class BlocksScenario(FrankaScenario):
    """单臂 Franka 场景，单个机器人在原点。"""

    cfg_cls: Type[FrankaScenarioCfg] = FrankaScenarioCfg
    cfg: FrankaScenarioCfg
    robot_cls: Type[Robot] = Robot
    world: World
    stage: object

    def __init__(self, cfg: ScenarioCfg):
        super().__init__(cfg)

    def load_robot(self):
        # 机器人在原点 (0, 0, 0)
        translation = (0.0, 0.0, 0.0)

        self.robot = self.robot_cls(self.cfg, prim_path="/World/Franka")
        self.robot.xform_prim.set_world_pose(
            position=translation,
        )
        self.world.scene.add(self.robot.articulation)  # type: ignore[arg-type]

    def load_objects(self):
        
        self.obstacle_list = []
        self.table = FixedCuboid(
                name="table",
                prim_path="/World/objects/obstacle_1",
                scale=np.array([0.9, 0.5, 0.05]),
                position=np.array([0.,-0.5, 0.05]),
                color=np.array([0.08, 0.08, 0.08]),
            )
        self.world.scene.add(self.table)
        self.obstacle_list.append(self.table)
        
        # 摆放小方块
        self.cube1 = DynamicCuboid(
            name="cube1",  # 棕色，横竖
            prim_path="/World/targets/cube1",
            position=np.array([-0.2,-0.5,0.1]),
            size=0.05,
            color=np.array([1, 0, 0], dtype=float), #np.array([0.5, 0.2, 0.2]
        )
        self.world.scene.add(self.cube1)
        self.obstacle_list.append(self.cube1)

        self.cube2 = DynamicCuboid(
            name="cube2",  # 绿色
            prim_path="/World/targets/cube2",
            position=np.array([0.35,-0.45,0.1]),
            size=0.05,
            color=np.array([0, 1, 0], dtype=float), # np.array([0.4, 0.6, 0.3]
        )
        self.world.scene.add(self.cube2)
        self.obstacle_list.append(self.cube2)

        self.cube3 = DynamicCuboid(
            name="cube3",  # 紫色
            prim_path="/World/targets/cube3",
            position=np.array([0,-0.6,0.1]),
            size=0.05,
            color=np.array([0, 0, 1], dtype=float), # [0.2, 0.1, 0.7]
        )
        self.world.scene.add(self.cube3)
        self.obstacle_list.append(self.cube3)

        self.cube4 = DynamicCuboid(
            name="cube4",  # 蓝色
            prim_path="/World/targets/cube4",
            position=np.array([0.25,-0.65,0.1]),
            size=0.05,
            color=np.array([1, 1, 0], dtype=float), # [0.2, 0.6, 0.9]
        )
        self.world.scene.add(self.cube4)
        self.obstacle_list.append(self.cube4)

        self.obj_map: Dict[str, DynamicCuboid] = {
            "cube1": self.cube1,
            "cube2": self.cube2,
            "cube3": self.cube3,
            "cube4": self.cube4,
        }

    def setup(self):
        self.setup_viewport_camera()
        self.setup_camera()
        self.robot.setup()
        self.reset()




####################
# pour
#####################
# class CoverScenarioCfg(FrankaScenarioCfg):
#     # myx input

#     obj_cls5 = Milk   # Cup
#     obj_init_pos5 = [-0.2,-0.55,0.1] 
#     obj_init_euler5 = [0, 0, 0] 
#     obj_mass5 = 3  

#     obj_cls6 = Bowl   # Cup
#     obj_init_pos6 = [0.2,-0.55,0.1] 
#     obj_init_euler6 = [0, 0, 0] 
#     obj_mass6 = 3  

# class CoverScenario(FrankaScenario):
#     """单臂 Franka 场景，单个机器人在原点。"""

#     cfg_cls: Type[CoverScenarioCfg] = CoverScenarioCfg
#     cfg: CoverScenarioCfg
#     robot_cls: Type[Robot] = Robot
#     world: World
#     stage: object

#     def __init__(self, cfg: ScenarioCfg):
#         super().__init__(cfg)

#     def load_robot(self):
#         # 机器人在原点 (0, 0, 0)
#         translation = (0.0, 0.0, 0.0)

#         self.robot = self.robot_cls(self.cfg, prim_path="/World/Franka")
#         self.robot.xform_prim.set_world_pose(
#             position=translation,
#         )
#         self.world.scene.add(self.robot.articulation)  # type: ignore[arg-type]

#     def load_objects(self):
        
#         self.obstacle_list = []
#         self.table = FixedCuboid(
#                 name="table",
#                 prim_path="/World/objects/obstacle_1",
#                 scale=np.array([1.2, 0.6, 0.05]),
#                 position=np.array([0.,-0.5, 0.05]),
#                 color=np.array([0.05, 0.05, 0.05]),
#             )
#         self.world.scene.add(self.table)
#         self.obstacle_list.append(self.table)

#         self.plate1 = self.cfg.obj_cls5(
#             position=self.cfg.obj_init_pos5,
#             orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler5),
#             mass=self.cfg.obj_mass5,
#             name="obj5",
#         )
#         self.plate1_prim = self.plate1.create_prim()
#         self.world.scene.add(self.plate1_prim)
#         self.task_obj_list.append(self.plate1)

#         self.plate2 = self.cfg.obj_cls6(
#             position=self.cfg.obj_init_pos6,
#             orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler6),
#             mass=self.cfg.obj_mass6,
#             name="obj6",
#         )
#         self.plate2_prim = self.plate2.create_prim()
#         self.world.scene.add(self.plate2_prim)
#         self.task_obj_list.append(self.plate2)

#         # self.milk = self.cfg.obj_milk(
#         #     position=self.cfg.obj_init_pos_milk,
#         #     orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler_milk),
#         #     mass=self.cfg.obj_mass_milk,
#         #     name="milk_myx",
#         # )
#         # self.milk_prim = self.milk.create_prim()
#         # self.world.scene.add(self.milk_prim)
#         # self.task_obj_list.append(self.milk)

#         # self.cup = self.cfg.obj_cup(
#         #     position=self.cfg.obj_init_pos_cup,
#         #     orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler_cup),
#         #     mass=self.cfg.obj_mass_cup,
#         #     name="cup_myx",
#         # )
#         # self.cup_prim = self.cup.create_prim()
#         # self.world.scene.add(self.cup_prim)
#         # self.task_obj_list.append(self.cup)

#         self.obj_map: Dict[str, DynamicCuboid] = {
#             "plate1": self.plate1,
#             "plate2": self.plate2,
#         }

#     def setup(self):
#         self.setup_viewport_camera()
#         self.setup_camera()
#         self.robot.setup()
#         self.reset()



# class CoverScenarioCfg(FrankaScenarioCfg):
#     # myx input
#     obj_cls5 = Milk_myx # Boiler  E13
#     obj_init_pos5 = [-0.2,-0.55,0.1] #[0.2, -0.45, 0.96]
#     obj_init_euler5 = [0, 0, 0] # ?
#     obj_mass5 = 3  # ?

#     obj_cls7 = Bowl # Bowl E31
#     obj_init_pos7 = [0.2,-0.55,0.1]  #[0.2, -0.45, 0.96]
#     obj_init_euler7 = [0, 0, 0] # ?
#     obj_mass7 = 3  # ?

# class CoverScenario(FrankaScenario):
#     """单臂 Franka 场景，单个机器人在原点。"""

#     cfg_cls: Type[CoverScenarioCfg] = CoverScenarioCfg
#     cfg: CoverScenarioCfg
#     robot_cls: Type[Robot] = Robot
#     world: World
#     stage: object

#     def __init__(self, cfg: ScenarioCfg):
#         super().__init__(cfg)

#     def load_robot(self):
#         # 机器人在原点 (0, 0, 0)
#         translation = (0.0, 0.0, 0.0)

#         self.robot = self.robot_cls(self.cfg, prim_path="/World/Franka")
#         self.robot.xform_prim.set_world_pose(
#             position=translation,
#         )
#         self.world.scene.add(self.robot.articulation)  # type: ignore[arg-type]

#     def load_objects(self):
        
#         self.obstacle_list = []
#         self.table = FixedCuboid(
#                 name="table",
#                 prim_path="/World/objects/obstacle_1",
#                 scale=np.array([1.2, 0.6, 0.05]),
#                 position=np.array([0.,-0.5, 0.05]),
#                 color=np.array([0.05, 0.05, 0.05]),
#             )
#         self.world.scene.add(self.table)
#         self.obstacle_list.append(self.table)
        
#         self.plate1 = self.cfg.obj_cls5(
#             position=self.cfg.obj_init_pos5,
#             orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler5),
#             mass=self.cfg.obj_mass5,
#             name="obj5",
#         )
#         self.plate1_prim = self.plate1.create_prim()
#         self.world.scene.add(self.plate1_prim)
#         self.task_obj_list.append(self.plate1)

#         self.plate3 = self.cfg.obj_cls7(
#             position=self.cfg.obj_init_pos7,
#             orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler7),
#             mass=self.cfg.obj_mass7,
#             name="obj7",
#         )
#         self.plate3_prim = self.plate3.create_prim()
#         self.world.scene.add(self.plate3_prim)
#         self.task_obj_list.append(self.plate3)


#         self.obj_map: Dict[str, DynamicCuboid] = {
#             "plate1": self.plate1,
#             "plate3": self.plate3,
#         }

#     def setup(self):
#         self.setup_viewport_camera()
#         self.setup_camera()
#         self.robot.setup()
#         self.reset()