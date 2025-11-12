import numpy as np
from typing import Optional, cast
import threading

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    keyboard = None

from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.types import ArticulationAction

from dexrl.sim import utils
from dexrl.sim.scenarios.franka import FrankaScenarioCfg, Robot
from dexrl.sim.scenarios.multi_franka_sim import MultiFrankaScenario, MultiFrankaCleanScenario, MultiFrankaCleanScenarioCfg, MultiFrankaHandOverScenario, MultiFrankaHandOverScenarioCfg
from dexrl.sim_extension.ext_scenarios.franka_ext_scenario import FrankaExtScenario


class MultiFrankaExtScenario(FrankaExtScenario):
    """双臂 Franka 扩展场景：左右臂分别跟随不同的目标立方体。"""
    
    scenario_cls = MultiFrankaScenario
    scenario_cfg_cls = FrankaScenarioCfg

    def __init__(self, cfg):
        self.cfg = cfg
        self.set_scenario_cfg()
        self.scenario = self.scenario_cls(self.scenario_cfg)
        self.script_generator = self.script()

        self.left_robot: Optional[Robot] = None
        self.right_robot: Optional[Robot] = None

        self.left_cube: Optional[VisualCuboid] = None
        self.right_cube: Optional[VisualCuboid] = None

        self.left_center = np.array([-0.45, -0.45, 0.25], dtype=float)
        self.right_center = np.array([0.45, -0.45, 0.25], dtype=float)

        self.ee_offset = np.array([0.0, 0.0, 0.05])
        self.tool_quat = utils.rot.euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))

        self.animate_targets = True
        self.animation_radius = 0.15
        self.animation_height = 0.04
        self.animation_speed = 0.8
        self._time = 0.0

    def load_assets(self):
        super().load_assets()

        multi_scenario = cast(MultiFrankaScenario, self.scenario)
        self.left_robot = multi_scenario.left_robot
        self.right_robot = multi_scenario.right_robot

    def load_ext_objects(self):
        self.left_cube = VisualCuboid(
            name="left_target_cube",
            prim_path="/World/targets/left_cube",
            position=self.left_center.tolist(),
            size=0.04,
            color=np.array([0.9, 0.2, 0.2], dtype=float),
        )
        self.right_cube = VisualCuboid(
            name="right_target_cube",
            prim_path="/World/targets/right_cube",
            position=self.right_center.tolist(),
            size=0.04,
            color=np.array([0.2, 0.4, 0.9], dtype=float),
        )

        # 旋转两个 cube 能旋转 机器人末端
        _,left_cube_quat = self.left_cube.get_world_pose()
        _,right_cube_quat = self.right_cube.get_world_pose()
        init_left_euler = utils.rot.quat_to_euler_angles(left_cube_quat)
        init_right_euler = utils.rot.quat_to_euler_angles(right_cube_quat)

        self.cube_init_euler = {
            "left_cube": init_left_euler,
            "right_cube": init_right_euler,
        }

    def script(self):
        assert self.left_robot is not None and self.right_robot is not None
        assert self.left_cube is not None and self.right_cube is not None

        while True:
            self._time += self.animation_speed * (1 / 60.0)
            if self.animate_targets:
                self._update_target_animation(self.left_cube, self.left_center, True)
                self._update_target_animation(self.right_cube, self.right_center, False)

            self._follow_cube(self.left_robot, self.left_cube)
            self._follow_cube(self.right_robot, self.right_cube)
            yield

    def _update_target_animation(self, cube: VisualCuboid, center: np.ndarray, mirror: bool):
        direction = -1.0 if mirror else 1.0
        angle = self._time * direction

        x = center[0] + self.animation_radius * np.cos(angle)
        y = center[1] + 0.05 * np.sin(self._time * 0.7 * direction)
        z = center[2] + self.animation_height * np.sin(self._time * 1.2)

        cube.set_world_pose(
            position=[x, y, z],
            orientation=utils.rot.euler_angles_to_quat(np.array([0.0, 0.0, 0.0])).tolist(),
        )

    def _follow_cube(self, robot: Robot, cube: VisualCuboid):
        target_pos, _ = cube.get_world_pose()
        world_target = np.array(target_pos) + self.ee_offset

        base_pos, base_quat = robot.xform_prim.get_world_pose()
        local_target = self._world_to_local(world_target, base_pos, base_quat)

        articulation_action, success = robot.inverse_kinematics(local_target, self.tool_quat)
        if success and articulation_action is not None:
            robot.articulation.apply_action(articulation_action)

    def _follow_cube_with_euler(self, robot: Robot, cube: VisualCuboid,cube_name: str):
        assert cube_name in ["left_cube", "right_cube"]
        target_pos, target_quat = cube.get_world_pose()

        delta_euler = utils.rot.quat_to_euler_angles(target_quat) - self.cube_init_euler[cube_name]
        current_quat = utils.rot.euler_angles_to_quat(np.array([0+delta_euler[0],np.pi+delta_euler[1],0+delta_euler[2]]))


        world_target = np.array(target_pos) + self.ee_offset
        base_pos, base_quat = robot.xform_prim.get_world_pose()
        local_target = self._world_to_local(world_target, base_pos, base_quat)

        articulation_action, success = robot.inverse_kinematics(local_target, current_quat)
        if success and articulation_action is not None:
            robot.articulation.apply_action(articulation_action)

    def _follow_cube_without_euler(self, robot: Robot, cube: VisualCuboid,cube_name: str,current_quat):
        assert cube_name in ["left_cube", "right_cube"]
        target_pos, target_quat = cube.get_world_pose()

        # delta_euler = utils.rot.quat_to_euler_angles(target_quat) - self.cube_init_euler[cube_name]
        # current_quat = utils.rot.euler_angles_to_quat(np.array([0+delta_euler[0],np.pi+delta_euler[1],0+delta_euler[2]]))

        world_target = np.array(target_pos) + self.ee_offset
        base_pos, base_quat = robot.xform_prim.get_world_pose()
        local_target = self._world_to_local(world_target, base_pos, base_quat)

        articulation_action, success = robot.inverse_kinematics(local_target, current_quat)
        if success and articulation_action is not None:
            robot.articulation.apply_action(articulation_action)

    @staticmethod
    def _world_to_local(world_point, base_pos, base_quat):
        offset = np.array(world_point) - np.array(base_pos)
        rot_matrix = utils.rot.quat_to_rot_matrix(base_quat)
        return rot_matrix.T @ offset

    def _open_gripper(self, robot: Robot):
        """打开指定机器人的夹爪"""
        gripper_pos = 0.04
        articulation_action = ArticulationAction(
            joint_positions=[gripper_pos] * 2,
            joint_indices=[7, 8]
        )
        robot.articulation.apply_action(articulation_action)

    def _close_gripper(self, robot: Robot):
        """关闭指定机器人的夹爪"""
        gripper_pos = 0.0
        articulation_action = ArticulationAction(
            joint_positions=[gripper_pos] * 2,
            joint_indices=[7, 8]
        )
        robot.articulation.apply_action(articulation_action)












# 整理桌面
class MultiFrankaCleanExtScenario(MultiFrankaExtScenario):
    scenario: MultiFrankaCleanScenario
    scenario_cls = MultiFrankaCleanScenario
    scenario_cfg_cls = MultiFrankaCleanScenarioCfg
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # 初始化按键状态字典，用于存储按键按下事件
        self.key_pressed = {
            'g': False,
            'h': False,
            'j': False,
            'k': False
        }
        # 用于线程同步的锁
        self.key_lock = threading.Lock()
        # 键盘监听器
        self.keyboard_listener = None
        # 记录上一次处理的按键状态，用于检测按键按下事件
        self.prev_key_states = {
            'g': False,
            'h': False,
            'j': False,
            'k': False
        }

    def _on_key_press(self, key):
        """键盘按下事件处理"""
        try:
            # 获取按键字符（如果是普通字符键）
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                if key_char in self.key_pressed:
                    with self.key_lock:
                        self.key_pressed[key_char] = True
        except AttributeError:
            pass

    def _on_key_release(self, key):
        """键盘释放事件处理"""
        try:
            # 获取按键字符（如果是普通字符键）
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                if key_char in self.key_pressed:
                    with self.key_lock:
                        self.key_pressed[key_char] = False
        except AttributeError:
            pass

    def _start_keyboard_listener(self):
        """启动键盘监听器"""
        if not PYNPUT_AVAILABLE:
            print("警告: pynput 模块未安装，键盘监听功能不可用。请在 Isaac Sim 的 Python 环境中安装: pip install pynput")
            return
        if self.keyboard_listener is None or not self.keyboard_listener.running:
            assert keyboard is not None  # 类型检查：此时 keyboard 一定不为 None
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.keyboard_listener.start()

    def _stop_keyboard_listener(self):
        """停止键盘监听器"""
        if not PYNPUT_AVAILABLE:
            return
        if self.keyboard_listener is not None and self.keyboard_listener.running:
            self.keyboard_listener.stop()

    def script(self):
        assert self.left_robot is not None and self.right_robot is not None
        assert self.left_cube is not None and self.right_cube is not None



        pos = np.array([0,-0.5,0.2])
        quat = utils.rot.euler_angles_to_quat(np.array([np.pi/2,np.pi,0]))
        yield from self.scenario.goto_position(pos,quat,self.right_robot.articulation,self.right_robot.rmpflow)

        pos = np.array([-0.045,-0.5,0.05])
        quat = utils.rot.euler_angles_to_quat(np.array([0,np.pi,0]))
        yield from self.scenario.goto_position(pos,quat,self.left_robot.articulation,self.left_robot.rmpflow)
        print(f"pos: {pos}, quat: {quat}")
        close_gripper = self.scenario.left_robot.close_gripper()
        yield from close_gripper
        pos = np.array([0.3,-0.5,0.5])
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        pos = np.array([0.35,-0.4,0.3])
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)


        # 启动键盘监听器
        self._start_keyboard_listener()

        try: 
            while True:
                # self._follow_cube_with_euler(self.left_robot, self.left_cube, "left_cube")
                current_quat = utils.rot.euler_angles_to_quat(np.array([np.pi/2,np.pi,0]))
                self._follow_cube_without_euler(self.right_robot, self.right_cube,"left_cube",current_quat,)

                # 按 g 控制左边的夹爪关闭，按 h 控制左边夹爪打开
                # 按 j 控制右边的夹爪关闭，按 k 控制右边夹爪打开
                
                # 检查键盘输入（检测按键按下事件：从未按下到按下）
                with self.key_lock:
                    current_key_states = self.key_pressed.copy()
                
                # 检测按键按下事件（从未按下到按下）
                if current_key_states['g'] and not self.prev_key_states['g']:
                    # 左边夹爪关闭
                    self._close_gripper(self.left_robot)
                
                if current_key_states['h'] and not self.prev_key_states['h']:
                    # 左边夹爪打开
                    self._open_gripper(self.left_robot)
                
                if current_key_states['j'] and not self.prev_key_states['j']:
                    # 右边夹爪关闭
                    self._close_gripper(self.right_robot)
                
                if current_key_states['k'] and not self.prev_key_states['k']:
                    # 右边夹爪打开
                    self._open_gripper(self.right_robot)
                
                # 更新上一次按键状态
                self.prev_key_states = current_key_states.copy()
                
                yield
        finally:
            # 确保在退出时停止监听器
            self._stop_keyboard_listener()


    # def load_ext_objects(self):
    #     super().load_ext_objects()
        # self.scenario.load_objects()

        # 箱子，积木，木板





class MultiFrankaHandOverExtScenario(MultiFrankaCleanExtScenario):
    scenario: MultiFrankaHandOverScenario
    scenario_cls = MultiFrankaHandOverScenario
    scenario_cfg_cls = MultiFrankaHandOverScenarioCfg


    def script(self):
        assert self.left_robot is not None and self.right_robot is not None
        assert self.left_cube is not None and self.right_cube is not None



        # pos = np.array([0,-0.5,0.2])
        # quat = utils.rot.euler_angles_to_quat(np.array([np.pi/2,np.pi,0]))
        # yield from self.scenario.goto_position(pos,quat,self.right_robot.articulation,self.right_robot.rmpflow)

        pos = np.array([0.03,-0.2,0.05])
        quat = utils.rot.euler_angles_to_quat(np.array([np.pi/2,0,0]))
        yield from self.scenario.goto_position(pos,quat,self.left_robot.articulation,self.left_robot.rmpflow)
        print(f"pos: {pos}, quat: {quat}")
        close_gripper = self.scenario.left_robot.close_gripper()
        yield from close_gripper
        # pos = np.array([0.3,-0.5,0.5])
        # yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        pos = np.array([0.35,-0.4,0.3])
        yield from self.scenario.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)


        # 启动键盘监听器
        self._start_keyboard_listener()

        try: 
            while True:
                # self._follow_cube_with_euler(self.left_robot, self.left_cube, "left_cube")
                current_quat = utils.rot.euler_angles_to_quat(np.array([np.pi/2,np.pi,0]))
                # self._follow_cube_without_euler(self.right_robot, self.right_cube,"left_cube",current_quat,)
                self._follow_cube(self.right_robot, self.right_cube,"left_cube",current_quat,)

                # 按 g 控制左边的夹爪关闭，按 h 控制左边夹爪打开
                # 按 j 控制右边的夹爪关闭，按 k 控制右边夹爪打开
                
                # 检查键盘输入（检测按键按下事件：从未按下到按下）
                with self.key_lock:
                    current_key_states = self.key_pressed.copy()
                
                # 检测按键按下事件（从未按下到按下）
                if current_key_states['g'] and not self.prev_key_states['g']:
                    # 左边夹爪关闭
                    self._close_gripper(self.left_robot)
                
                if current_key_states['h'] and not self.prev_key_states['h']:
                    # 左边夹爪打开
                    self._open_gripper(self.left_robot)
                
                if current_key_states['j'] and not self.prev_key_states['j']:
                    # 右边夹爪关闭
                    self._close_gripper(self.right_robot)
                
                if current_key_states['k'] and not self.prev_key_states['k']:
                    # 右边夹爪打开
                    self._open_gripper(self.right_robot)
                
                # 更新上一次按键状态
                self.prev_key_states = current_key_states.copy()
                
                yield
        finally:
            # 确保在退出时停止监听器
            self._stop_keyboard_listener()