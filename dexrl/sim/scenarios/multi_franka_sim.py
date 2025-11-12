from typing import List, Optional, Type

import numpy as np
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.world import World
from dexrl.sim.objects import *
from dexrl.sim.scenarios._cfg import ScenarioCfg
from dexrl.sim.scenarios.franka import FrankaScenario, FrankaScenarioCfg, Robot
from dexrl.utils.configclass import configclass
from dexrl.sim.objects import BigLego, Carton
from dexrl.sim import utils


class MultiFrankaScenario(FrankaScenario):
    """双臂 Franka 场景，左右对称分布。"""

    cfg_cls: Type[FrankaScenarioCfg] = FrankaScenarioCfg
    cfg: FrankaScenarioCfg
    robot_cls: Type[Robot] = Robot
    world: World
    stage: object

    def __init__(self, cfg: ScenarioCfg):
        super().__init__(cfg)
        self.robots: List[Robot] = []
        self.left_robot: Optional[Robot] = None
        self.right_robot: Optional[Robot] = None

    def load_robot(self):
        horizontal_offset = 0.45
        depth_offset = 0 #-0.5
        height_offset = 0.0

        left_translation = (-horizontal_offset, depth_offset, height_offset)
        right_translation = (horizontal_offset, depth_offset, height_offset)

        self.left_robot = self.robot_cls(self.cfg, prim_path="/World/Franka_left")
        self.left_robot.xform_prim.set_world_pose(
            position=left_translation,
        )
        self.world.scene.add(self.left_robot.articulation)  # type: ignore[arg-type]

        self.right_robot = self.robot_cls(self.cfg, prim_path="/World/Franka_right")
        self.right_robot.xform_prim.set_world_pose(
            position=right_translation,
        )
        self.world.scene.add(self.right_robot.articulation)  # type: ignore[arg-type]

        self.robot = self.left_robot
        self.robots = [self.left_robot, self.right_robot]

    def setup(self):
        self.setup_viewport_camera()
        self.setup_camera()
        for robot in self.robots:
            robot.setup()
        self.reset()





####################
# 整理桌面
#####################

@configclass
class MultiFrankaCleanScenarioCfg(FrankaScenarioCfg):
    block_size = 0.05
    block_positions = [
        [0.0, -0.4, 0.1],
        [0.1, -0.6, 0.1],
        [0.4, -0.4, 0.1],
    ]
    block_colors = [
        [1, 0, 0],
    ]
    
    # Table configuration
    table_pos = [0, -0.5, 0.01]
    table_scale = [1.6, 0.7, 0.04]
    
    # Target pad configuration
    target_pad_pos = [0, -0.6280072354597021, 0.06]
    target_pad_scale = [0.6, 0.15, 0.08]
    
    # Box (Carton) configuration
    box_cls = Carton
    box_init_pos = [0.0, -0.35, 0.1]
    box_init_euler = [0, 0, 1.57]
    box_mass = 3.0
    
    # Lego configuration
    lego_count = 3
    lego_x_range = [-0.5, 0.5]
    lego_y_range = [-0.4, -0.3]
    lego_z_range = [0.2, 0.1]
    lego_mass = 0.1

class MultiFrankaCleanScenario(MultiFrankaScenario):
    cfg_cls: Type[MultiFrankaCleanScenarioCfg] = MultiFrankaCleanScenarioCfg
    cfg: MultiFrankaCleanScenarioCfg
    robot_cls: Type[Robot] = Robot
    world: World
    stage: object

    def __init__(self, cfg: MultiFrankaCleanScenarioCfg):
        super().__init__(cfg)
        # 将配置设置到 self.cfg 中，以便 load_objects() 可以访问
        self.cfg = cfg

    def load_objects(self):
        # 初始化场景对象列表
        print(f"[MultiFrankaCleanScenario] load_objects() called")
        print(f"[MultiFrankaCleanScenario] cfg type: {type(self.cfg)}")
        print(f"[MultiFrankaCleanScenario] storage_bins: {getattr(self.cfg, 'storage_bins', None)}")
        print(f"[MultiFrankaCleanScenario] block_positions: {getattr(self.cfg, 'block_positions', None)}")
        print(f"[MultiFrankaCleanScenario] block_colors: {getattr(self.cfg, 'block_colors', None)}")
        print(f"[MultiFrankaCleanScenario] world: {self.world}")
        
        self.obstacle_list = []
        self.task_obj_list = []

        # 创建 table
        table_pos = getattr(self.cfg, "table_pos", [0, -0.4, 0.1])
        table_scale = getattr(self.cfg, "table_scale", [1.6, 0.7, 0.04])
        self.table = FixedCuboid(
            name="clean_table",
            position=np.array(table_pos, dtype=float),
            size=1,
            scale=np.array(table_scale, dtype=float),
            color=np.array([0.1, 0.1, 0.1], dtype=float),
            prim_path="/World/clean_scene/table",
        )
        self.world.scene.add(self.table)
        
        # 创建 target_pad
        target_pad_pos = getattr(self.cfg, "target_pad_pos", [0, -0.6280072354597021, 0.96])
        target_pad_scale = getattr(self.cfg, "target_pad_scale", [0.6, 0.15, 0.08])
        self.target_pad = FixedCuboid(
            name="target_pad",
            position=np.array(target_pad_pos, dtype=float),
            size=1,
            scale=np.array(target_pad_scale, dtype=float),
            color=np.array([0.04, 0.02, 0.01], dtype=float),  # 深色棕色
            prim_path="/World/clean_scene/target_pad",
        )
        self.world.scene.add(self.target_pad)
        
        # 创建 box (Carton)
        self.box: BaseObject = self.cfg.box_cls(
            position=self.cfg.box_init_pos,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.box_init_euler),
            mass=self.cfg.box_mass,
            # name="carton",
            # prim_path="/World/carton",
            )
        self.box_prim = self.box.create_prim()
        self.world.scene.add(self.box_prim)
        self.task_obj_list.append(self.box)
        self.obstacle_list.append(self.box)
        
        # 生成随机位置和颜色的 lego
        lego_count = 5 #getattr(self.cfg, "lego_count", 5)
        lego_x_range = getattr(self.cfg, "lego_x_range", [-0.2, 0.2])
        lego_y_range = getattr(self.cfg, "lego_y_range", [-0.4, -0.3])
        lego_z_range = getattr(self.cfg, "lego_z_range", [1.0, 1.2])
        lego_mass = getattr(self.cfg, "lego_mass", 0.1)


        # 生成一个
        lego_init_euler = [np.pi/2, 0, 0]
        lego_obj = Lego(
            position=[-0.5, -0.5, 0.05],
            orientation=utils.rot.euler_angles_to_quat(lego_init_euler),
            mass=lego_mass,
            name=f"clean_lego_new",
            prim_path=f"/World/clean_scene/clean_lego_new",
        )
        lego_prim = lego_obj.create_prim()
        
        self.world.scene.add(lego_prim)
        self.task_obj_list.append(lego_obj)
        self.obstacle_list.append(lego_obj)
       
        
        self.lego_list = []
        for i in range(lego_count):
            # 随机位置
            x = np.random.uniform(lego_x_range[0], lego_x_range[1])
            y = np.random.uniform(lego_y_range[0], lego_y_range[1])
            z = np.random.uniform(lego_z_range[0], lego_z_range[1])
            
            # 随机颜色
            r = np.random.uniform(0.1, 0.9)
            g = np.random.uniform(0.1, 0.9)
            b = np.random.uniform(0.1, 0.9)
            
            # 随机欧拉角
            euler_x = np.random.uniform(0, 2*np.pi)
            euler_y = np.random.uniform(0, 2*np.pi)
            euler_z = np.random.uniform(0, 2*np.pi)
            
            lego_obj = BigLego(
                position=[x, y, z],
                orientation=utils.rot.euler_angles_to_quat([euler_x, euler_y, euler_z]),
                mass=lego_mass,
                name=f"clean_lego_{i}",
                prim_path=f"/World/clean_scene/lego_{i}",
                rgb=np.array([r, g, b]),
            )
            lego_prim = lego_obj.create_prim()
            
            self.world.scene.add(lego_prim)
            self.lego_list.append(lego_obj)
            self.task_obj_list.append(lego_obj)
            self.obstacle_list.append(lego_obj)

        try:
            import omni.replicator.core as rep  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            rep = None




@configclass
class MultiFrankaHandOverScenarioCfg(FrankaScenarioCfg):
    table_pos = [0,-0.4,0.01]
    table_scale = [1.6,0.7,0.04]

    obj_cls = Biscuit
    # table_pos[2] + table_scale[2]/2 + 176.328*Biscuit.scale_ratio_z/2
    obj_init_pos = [-0.15, -0.38, 0.1]
    # obj_init_pos = [-0.15, -0.4, table_pos[2] + table_scale[2]/2 + 176.328*Biscuit.scale_ratio_z/2]
    obj_init_euler = [0, 1.57, 0] # z改为 -1.57/6 1.57/6
    # obj_delta_euler = 1.57/6
    obj_mass = 0.2 #0.2


    # obj_cls = Carton #ThickCoconutMilk #Biscuit
    # obj_init_pos = [-0.4288952946662903, -0.44621750712394714, 1.0045260190963745]
    # obj_init_euler = [1.57, 0, 0.7]  #[0, 1.57, 0] 
    # obj_mass = 0.2
    
    obj_cls2 = Biscuit #Bowl # Biscuit
    obj_init_pos2 = [-0.32910566627979279, -0.5669710040092468, 0.3]
    obj_init_euler2 = [0, 3.14, 0]  #[0, 1.57, 0] 
    obj_mass2 = 0.2
    
    obj_cls3 = Biscuit #Microwave
    obj_init_pos3 = [0.35, -0.55, 0.3] #[0.2, -0.45, 0.96]
    obj_init_euler3 = [0, 3.14, 1.57] 
    obj_mass3 = 3
    
    obj_cls4 = Biscuit #Microwave
    obj_init_pos4 = [0.45, -0.55, 0.3] #[0.2, -0.45, 0.96]
    obj_init_euler4 = [0, 3.14, 1.57] 
    obj_mass4 = 3
    
    target_pad_pos = [0, -0.6280072354597021, 0.96]
    target_pad_scale = [0.6, 0.15, 0.08]


class MultiFrankaHandOverScenario(MultiFrankaScenario):
    cfg_cls: Type[MultiFrankaHandOverScenarioCfg] = MultiFrankaHandOverScenarioCfg
    cfg: MultiFrankaHandOverScenarioCfg
    robot_cls: Type[Robot] = Robot
    world: World
    stage: object

    def __init__(self, cfg: MultiFrankaHandOverScenarioCfg):
        super().__init__(cfg)
        # 将配置设置到 self.cfg 中，以便 load_objects() 可以访问
        self.cfg = cfg

    def load_objects(self):
        self.table = FixedCuboid(
            name="table",
            position=np.array([self.cfg.table_pos]),
            size=1,
            scale=self.cfg.table_scale,
            color=np.array([0.1, 0.1, 0.1]),
            prim_path="/World/table",
        )
        self.world.scene.add(self.table)
        
        self.obj: BaseObject = self.cfg.obj_cls(
            position=self.cfg.obj_init_pos,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler),
            mass=self.cfg.obj_mass,
            name="obj1",
            )
        self.obj_prim = self.obj.create_prim()
        
        self.obj2: BaseObject = self.cfg.obj_cls2(
            position=self.cfg.obj_init_pos2,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler2),
            mass=self.cfg.obj_mass2,
            name="obj2",
            )
        self.obj_prim2 = self.obj2.create_prim()
        self.world.scene.add(self.obj_prim2)
        self.task_obj_list.append(self.obj2)
        
        self.obj3: BaseObject = self.cfg.obj_cls3(
            position=self.cfg.obj_init_pos3,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler3),
            mass=self.cfg.obj_mass3,
            name="obj3",
            )
        self.obj_prim3 = self.obj3.create_prim()
        self.world.scene.add(self.obj_prim3)
        self.task_obj_list.append(self.obj3)
        
        self.obj4: BaseObject = self.cfg.obj_cls4(
            position=self.cfg.obj_init_pos4,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler4),
            mass=self.cfg.obj_mass4,
            name="obj4",
            )
        self.obj_prim4 = self.obj4.create_prim()
        self.world.scene.add(self.obj_prim4)
        self.task_obj_list.append(self.obj4)
        
        self.world.scene.add(self.obj_prim)
        self.task_obj_list.append(self.obj)