import numpy as np
from typing import Optional

from omni.isaac.core.objects import VisualCuboid

from dexrl.sim import utils
from dexrl.sim.scenarios.franka import FrankaScenarioCfg, Robot
from dexrl.sim.scenarios.one_franka_sim import OneFrankaScenario,CoverScenario,CoverScenarioCfg,BlocksScenario
from dexrl.sim_extension.ext_scenarios.franka_ext_scenario import FrankaExtScenario

class OneFrankaExtScenario(FrankaExtScenario):
    """单臂 Franka 扩展场景：机器人跟随目标立方体。"""
    
    scenario_cls = OneFrankaScenario
    scenario_cfg_cls = FrankaScenarioCfg
    viewport_camera_pos_lookat = np.array([1.36,-1.93,1.38,0.84,-1.24,0.88])

    def __init__(self, cfg):
        self.cfg = cfg
        self.set_scenario_cfg()
        self.scenario = self.scenario_cls(self.scenario_cfg)
        self.script_generator = self.script()

        self.robot: Optional[Robot] = None
        self.target_cube: Optional[VisualCuboid] = None

        self.cube_center = np.array([0.0, -0.4, 0.3], dtype=float)

        self.ee_offset = np.array([0.0, 0.0, 0.05])
        self.tool_quat = utils.rot.euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))

        self.animate_targets = True
        self.animation_radius = 0.15
        self.animation_height = 0.04
        self.animation_speed = 0.8
        self._time = 0.0

    def load_assets(self):
        super().load_assets()
        self.robot = self.scenario.robot

    def load_ext_objects(self):
        self.target_cube = VisualCuboid(
            name="target_cube",
            prim_path="/World/targets/target_cube",
            position=self.cube_center.tolist(),
            size=0.04,
            color=np.array([0.9, 0.2, 0.2], dtype=float),
        )

        # 记录初始 cube 的欧拉角，用于旋转控制
        _, cube_quat = self.target_cube.get_world_pose()
        self.cube_init_euler = utils.rot.quat_to_euler_angles(cube_quat)

    def script(self):
        assert self.robot is not None
        assert self.target_cube is not None

        while True:
            self._time += self.animation_speed * (1 / 60.0)
            if self.animate_targets:
                self._update_target_animation(self.target_cube, self.cube_center)

            self._follow_cube(self.robot, self.target_cube)
            yield

    def _update_target_animation(self, cube: VisualCuboid, center: np.ndarray):
        """更新目标立方体的动画位置"""
        angle = self._time

        x = center[0] + self.animation_radius * np.cos(angle)
        y = center[1] + 0.05 * np.sin(self._time * 0.7)
        z = center[2] + self.animation_height * np.sin(self._time * 1.2)

        cube.set_world_pose(
            position=[x, y, z],
            orientation=utils.rot.euler_angles_to_quat(np.array([0.0, 0.0, 0.0])).tolist(),
        )

    def _follow_cube(self, robot: Robot, cube: VisualCuboid):
        """让机器人末端跟随立方体"""
        target_pos, _ = cube.get_world_pose()
        world_target = np.array(target_pos) + self.ee_offset

        base_pos, base_quat = robot.xform_prim.get_world_pose()
        local_target = self._world_to_local(world_target, base_pos, base_quat)

        articulation_action, success = robot.inverse_kinematics(local_target, self.tool_quat)
        if success and articulation_action is not None:
            robot.articulation.apply_action(articulation_action)

    def _follow_cube_with_euler(self, robot: Robot, cube: VisualCuboid):
        """让机器人末端跟随立方体，包括旋转"""
        target_pos, target_quat = cube.get_world_pose()

        delta_euler = utils.rot.quat_to_euler_angles(target_quat) - self.cube_init_euler
        current_quat = utils.rot.euler_angles_to_quat(
            np.array([0 + delta_euler[0], np.pi + delta_euler[1], 0 + delta_euler[2]])
        )

        world_target = np.array(target_pos) + self.ee_offset
        base_pos, base_quat = robot.xform_prim.get_world_pose()
        local_target = self._world_to_local(world_target, base_pos, base_quat)

        articulation_action, success = robot.inverse_kinematics(local_target, current_quat)
        if success and articulation_action is not None:
            robot.articulation.apply_action(articulation_action)

    @staticmethod
    def _world_to_local(world_point, base_pos, base_quat):
        """将世界坐标系中的点转换为机器人基座坐标系"""
        offset = np.array(world_point) - np.array(base_pos)
        rot_matrix = utils.rot.quat_to_rot_matrix(base_quat)
        return rot_matrix.T @ offset






####################
# 扁的cover
#####################
class CoverExtScenario(OneFrankaExtScenario):
    scenario: CoverScenario
    scenario_cls = CoverScenario
    scenario_cfg_cls = CoverScenarioCfg
    
    def script(self):
        self.animate_targets = False
        assert self.robot is not None
        assert self.target_cube is not None

        # 叠放第一个立方体，粉色
        # pos = np.array([-0.3,-0.7,0.2])
        # quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0])) 
        # yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        # pos = np.array([-0.3,-0.7,0.08]) # 向下抓取
        # yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        # close_gripper = self.scenario.robot.close_gripper()
        # yield from close_gripper

        # pos = np.array([-0.3,-0.4,0.15]) # 抬起来
        # yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        # open_gripper = self.scenario.robot.open_gripper()
        # yield from open_gripper 

        # # 叠放第二个立方体，绿色
        # pos = np.array([0,-0.7,0.2])
        # quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0])) 
        # yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        # pos = np.array([0,-0.7,0.1]) # 向下抓取
        # yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        # close_gripper = self.scenario.robot.close_gripper()
        # yield from close_gripper

        # pos = np.array([0,-0.4,0.15]) # 抬起来
        # yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        # open_gripper = self.scenario.robot.open_gripper()
        # yield from open_gripper     

        # 叠放第三个立方体，紫色
        pos = np.array([0.3,-0.7,0.2])
        quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0])) 
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        pos = np.array([0.3,-0.7,0.1]) # 向下抓取
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        close_gripper = self.scenario.robot.close_gripper()
        yield from close_gripper

        pos = np.array([0.3,-0.4,0.3]) # 抬起来
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        open_gripper = self.scenario.robot.open_gripper()
        yield from open_gripper 

       # 稳态，跟随目标立方体
        while True:
            self._time += self.animation_speed * (1 / 60.0)
            if self.animate_targets:
                self._update_target_animation(self.target_cube, self.cube_center)

            self._follow_cube(self.robot, self.target_cube)
            yield


####################
# BlocksExtScenario
#####################
class BlocksExtScenario(OneFrankaExtScenario):
    scenario: BlocksScenario
    scenario_cls = BlocksScenario
    scenario_cfg_cls = FrankaScenarioCfg

    def script(self):
        assert self.robot is not None
        assert self.target_cube is not None

        # 去抓取第一个立方体，蓝色
        pos = np.array([0.25,-0.65,0.2])
        quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0])) 
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        pos = np.array([0.25,-0.65,0.1]) # 向下抓取
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        close_gripper = self.scenario.robot.close_gripper()
        yield from close_gripper

        pos = np.array([0.35,-0.45,0.2]) # 抬起来
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        open_gripper = self.scenario.robot.open_gripper()
        yield from open_gripper 

        # 去抓取第二个立方体，棕色
        pos = np.array([-0.2,-0.5,0.2])
        quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0])) 
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        pos = np.array([-0.2,-0.5,0.1]) # 向下抓取
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        close_gripper = self.scenario.robot.close_gripper()
        yield from close_gripper

        pos = np.array([0.35,-0.45,0.25]) # 抬起来
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        open_gripper = self.scenario.robot.open_gripper()
        yield from open_gripper 

        # 去抓取第三个立方体，紫色
        pos = np.array([0,-0.6,0.2])
        quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0])) 
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        pos = np.array([0,-0.6,0.1]) # 向下抓取
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        
        close_gripper = self.scenario.robot.close_gripper()
        yield from close_gripper

        pos = np.array([0.35,-0.45,0.3]) # 抬起来
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)

        open_gripper = self.scenario.robot.open_gripper()
        yield from open_gripper     

       # 稳态，跟随目标立方体
        while True:
            self._time += self.animation_speed * (1 / 60.0)
            if self.animate_targets:
                self._update_target_animation(self.target_cube, self.cube_center)

            self._follow_cube(self.robot, self.target_cube)
            yield
