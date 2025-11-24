import numpy as np
from typing import Dict, Type

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.world import World

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import GroundPlane, FixedCuboid, DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.motion_generation.interface_config_loader import load_supported_motion_policy_config
from omni.isaac.core.utils import distance_metrics
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.numpy.rotations import quats_to_rot_matrices
from omni.isaac.core.utils.types import ArticulationAction

import os
import pxr.Usd

import torch

from dexrl import global_config
try:
    from omni.isaac.nucleus import get_assets_root_path
except ImportError:
    # 如果导入失败，使用备用方法
    import carb
    def get_assets_root_path():
        return carb.settings.get_settings().get("/persistent/isaac/asset_root/default")

from dexrl.sim.scenarios._cfg import ScenarioCfg
from dexrl.sim import utils
from dexrl.utils.configclass import configclass
from omni.kit.viewport.utility import get_active_viewport_and_window, capture_viewport_to_file
from omni.isaac.sensor import Camera
import omni.replicator.core as rep

class Robot:
    '''
    Franka机器人
    Args:
        cfg: 配置
        prim_path: 机器人 prim 路径
    Returns:
        None
    Attributes:
        cfg: 配置
        prim_path: 机器人 prim 路径
        stage: 场景
        xform_prim: 机器人 xform prim
        articulation: 机器人 articulation
        articulation_controller: 机器人 articulation controller
        right_pose_forward_direction: 机器人右臂 forward 方向
        xyz_lower_limit: 机器人 xyz 下限
        xyz_upper_limit: 机器人 xyz 上限
        rmpflow: 机器人 rmpflow
        articulation_rmpflow: 机器人 articulation rmpflow
        inverse_kinematics: 机器人 inverse kinematics
        forward_kinematics: 机器人 forward kinematics
        get_mixed_pose_joint: 机器人 get mixed pose joint
        get_eef_pose_rmpflow: 机器人 get eef pose rmpflow
        get_eef_mixed_pose: 机器人 get eef mixed pose
    '''
    def __init__(self,cfg:ScenarioCfg, prim_path="/Franka"):
        self.cfg = cfg
        # 优先使用本地路径，如果不存在则使用 Nucleus 路径
        local_path = os.path.join(global_config.isaacsim_data_path, "Robots/Franka/franka.usd")
        # local_path = "/home/cys/RL_sim2real_WS/Sim2Real/assets/franka/franka.usd" 
        if os.path.exists(local_path):
            path_to_robot_usd = local_path
        else:
            # 使用 Isaac Sim 的 Nucleus 路径
            assets_root = get_assets_root_path()
            path_to_robot_usd = os.path.join(assets_root, "Robots/Franka/franka.usd")
        self.stage = get_current_stage()
        add_reference_to_stage(path_to_robot_usd, prim_path)
        self.prim_path = prim_path
        # self.xform_prim = XFormPrim(prim_path,orientation=utils.rot.euler_angles_to_quat(np.array([0,0,0])))
        self.xform_prim = XFormPrim(
            prim_path,
            orientation=tuple(utils.rot.euler_angles_to_quat(np.array([0.0, 0.0, -np.pi / 2])))
        )
        
        articulation_name = prim_path.rstrip("/").split("/")[-1] or "franka"
        self.articulation:Articulation = Articulation(prim_path=prim_path, name=articulation_name)
        self.articulation_controller = self.articulation.get_articulation_controller()


        self.right_pose_forward_direction = np.array([0,0,1])
        
        self.xyz_lower_limit = np.array([-0.5,       -0.5,    1]) 
        self.xyz_upper_limit = np.array([0.5,        -0.1,    1.5])


    def setup(self):
        # self.articulation_controller.set_gains(kps=self.cfg.controller_kps,kds=self.cfg.controller_kds)
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
        # from omni.isaac.motion_generation import ArticulationKinematicsSolver,LulaKinematicsSolver
        # self.kinematics_solver = LulaKinematicsSolver(rmp_config["robot_description_path"],rmp_config["urdf_path"])
        # self.end_effector_frame = "panda_link7"
        # self.articulation_kinematics_solver = ArticulationKinematicsSolver(self.articulation, self.kinematics_solver, self.end_effector_frame)
        # self.articulation_kinematics_solver.set_end_effector_frame(self.end_effector_frame)
        # self.robot_articulation.get_articulation_controller().set_gains(kps=1e7,kds=1e6)

        self.rmpflow = RmpFlow(**rmp_config)
        self.articulation_rmpflow = ArticulationMotionPolicy(self.articulation, self.rmpflow)



    def inverse_kinematics(
        self,
        pos,
        quat,
        translation_thresh: float = 1e-3,
        orientation_thresh: float = 1e-2,
        step_dt: float = 1 / 60,
    ):
        target_pos = np.asarray(pos, dtype=np.float64)
        target_quat = np.asarray(quat, dtype=np.float64)

        arm_joint_positions = np.asarray(self.articulation.get_joint_positions()[:7], dtype=np.float64)
        current_pos, current_rot = self.rmpflow.get_end_effector_pose(arm_joint_positions)
        target_rot = quats_to_rot_matrices(target_quat)

        trans_dist = distance_metrics.weighted_translational_distance(current_pos, target_pos)
        rot_dist = distance_metrics.rotational_distance_angle(current_rot, target_rot)

        if trans_dist < translation_thresh and rot_dist < orientation_thresh:
            return ArticulationAction(joint_positions=self.articulation.get_joint_positions()), True

        self.rmpflow.set_end_effector_target(target_pos, target_quat)
        self.rmpflow.update_world()

        action = self.articulation_rmpflow.get_next_articulation_action(step_dt)

        if action is None:
            return None, False

        return action, True

    def forward_kinematics(self,joint):
        ee_translation, ee_rotation = self.rmpflow.get_end_effector_pose(np.asarray(joint, dtype=np.float64))
        ee_quaternion = utils.rot.rot_matrix_to_quat(ee_rotation)
        return np.concatenate([ee_translation, ee_quaternion])


    def get_mixed_pose_joint(self,current_joint):
        current_pose = self.forward_kinematics(current_joint)
        # print(f"current_pose: {current_pose}","")
        pos = current_pose[:3] # 位置
        quat = current_pose[3:7] # 旋转
        
        # left  4元数
        
        
        rot_matrix = utils.rot.quat_to_rot_matrix(quat) # 旋转矩阵
        lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=self.right_pose_forward_direction) # 朝向

        y_euler = np.arcsin(-lookat_direction[2]) + np.pi/2
        z_euler = np.arctan2(lookat_direction[1], lookat_direction[0])

        origin_rot_matrix = utils.rot.euler_to_rot_matrix(np.array([0,y_euler,z_euler]))
        R = rot_matrix @ origin_rot_matrix.T

        x_lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([1,0,0])) # 朝向
        x_lookat_direction_after = utils.rot.rot_matrix_to_lookat_direction(R,forward_direction=x_lookat_direction)

        cos_theta = np.dot(x_lookat_direction,x_lookat_direction_after)
        x_euler = np.arccos(cos_theta)
        cross_product = np.cross(x_lookat_direction,x_lookat_direction_after)
        sin_theta = np.dot(cross_product, lookat_direction)

        # if left_arm:
        #     x_euler = -x_euler
            
        if sin_theta <0:
            
            x_euler = -x_euler

        # pos[2] += self.right_arm_hand.arm.arm_base_height
        return np.array([*pos,x_euler,y_euler,z_euler]) # 读的是绝对的


    # 方法1：通过 RMPflow（最推荐）
    def get_eef_pose_rmpflow(self):
        joint_positions = self.articulation.get_joint_positions()
        ee_trans, ee_rot = self.rmpflow.get_end_effector_pose(joint_positions[:7])
        return ee_trans, ee_rot
    
    
    # 方法3：获取混合位姿
    # def get_eef_mixed_pose(self):
    #     joint_positions = self.articulation.get_joint_positions()
    #     arm_joint = joint_positions[0:7]
    #     mixed_pose = self.get_mixed_pose_joint(arm_joint)
    #     return mixed_pose  # [x, y, z, x_euler, y_euler, z_euler]

    def open_gripper(self):
        self.current_gripper_pos = 0.04

        articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
        self.articulation.apply_action(articulation_action)
        while not np.allclose(self.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.002):
            yield
    def close_gripper(self):
        self.current_gripper_pos = 0 #0.02
        articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
        self.articulation.apply_action(articulation_action)
        stuck_count = 0
        while not np.allclose(self.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.004):
            last_gripper_pos = self.articulation.get_joint_positions()[7:]
            yield
            current_gripper_pos = self.articulation.get_joint_positions()[7:]
            # print(current_gripper_pos,last_gripper_pos)
            if np.allclose(last_gripper_pos, current_gripper_pos, atol=0.001):
                # print("gripper stuck")
                stuck_count += 1
                if stuck_count > 20:
                    break
            else:
                stuck_count = 0
            last_gripper_pos = current_gripper_pos


    def close_gripper(self):
        self.current_gripper_pos = 0 #0.02
        articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
        self.articulation.apply_action(articulation_action)
        stuck_count = 0
        while not np.allclose(self.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.004):
            last_gripper_pos = self.articulation.get_joint_positions()[7:]
            yield
            current_gripper_pos = self.articulation.get_joint_positions()[7:]
            # print(current_gripper_pos,last_gripper_pos)
            if np.allclose(last_gripper_pos, current_gripper_pos, atol=0.001):
                # print("gripper stuck")
                stuck_count += 1
                if stuck_count > 20:
                    break
            else:
                stuck_count = 0
            last_gripper_pos = current_gripper_pos

@configclass
class FrankaScenarioCfg(ScenarioCfg):
    viewport_camera_pos_lookat = [0, -2, 1,0,0,0.1]


class FrankaScenario:
    cfg_cls: Type[FrankaScenarioCfg] = FrankaScenarioCfg
    cfg: Type[FrankaScenarioCfg]
    robot_cls:Type[Robot] = Robot
    world:World
    stage: pxr.Usd.Stage

    def __init__(self,cfg:Type[FrankaScenarioCfg]):
        self.cfg = cfg
        self.task_obj_list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._viewport_api = None
        self._viewport_prim= None
        self._viewport_camera = None

    @property
    def viewport_api(self):
        if self._viewport_api is None:
            self._viewport_api,self._window = get_active_viewport_and_window()
        return self._viewport_api

    # @property
    # def viewport_prim(self):
    #     if self._viewport_prim is None:
    #         viewport_prim_path = str(self.viewport_api.get_active_camera().GetPrimPath())
    #         self._viewport_prim = XFormPrim(viewport_prim_path)
    #     return self._viewport_prim

    @property
    def viewport_camera(self):
        if self._viewport_camera is None:
            viewport_prim_path = str(self.viewport_api.get_active_camera().GetPrimPath())
            # rep.utils.viewport_manager.get_render_product(viewport_prim_path, resolution=(1024,1024))
            self._render_product = rep.create.render_product(viewport_prim_path, resolution=(1024,1024))
            self._viewport_camera = Camera(viewport_prim_path,render_product_path=self._render_product.path)

            # self._viewport_camera.add_bounding_box_2d_loose_to_frame()
            # self._viewport_camera.add_bounding_box_2d_tight_to_frame()
            # self._viewport_camera.add_semantic_segmentation_to_frame()
            if self._viewport_camera._custom_annotators["semantic_segmentation"] is None:
                self._viewport_camera._custom_annotators["semantic_segmentation"] = rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation",init_params = {"colorize":True}
                )
            self._viewport_camera._custom_annotators["semantic_segmentation"].attach([self._render_product.path])
            self._viewport_camera._current_frame["semantic_segmentation"] = None

            self._viewport_camera.initialize()
            # self._viewport_camera.add_normals_to_frame()
            self.world.scene.add(self._viewport_camera)
        return self._viewport_camera

    def setup_camera(self):
        print("supported_annotators",self.viewport_camera.supported_annotators)

    def save_viewport_image(self,output_path):
        capture_viewport_to_file(self.viewport_api, output_path)

    def get_viewport_image(self):
        # self.viewport_camera.add_bounding_box_2d_tight_to_frame()
        frame = self.viewport_camera.get_current_frame()
        # rgba = frame
        print(frame)
        return frame
        # mask = frame['semantic_segmentation']['data']
        # mask = (mask.astype(bool) * 255).astype(np.uint8)
        # return mask

    def set_world(self, world:World):
        self.world = world

    def load_assets(self):
        self.stage = get_current_stage()

        self.load_ground_plane()
        self.load_objects()
        self.load_robot()



    # def load_ground_plane(self):
    #     self.ground_plane = GroundPlane("/World/Ground")
    #     self.world.scene.add(self.ground_plane)

    #     # self.ground_plane = self.world.scene.add_default_ground_plane()

    def load_ground_plane(self, floor_type="gridroom_curved"):
        """
        加载地板平面
        Args:
            floor_type: 地板类型，可选值：
                       - "default": 默认平面地板
                       - "gridroom_curved": Isaac Sim内置的网格房间地板（带弯曲效果）
                       - "gridroom_curved_white": 白色网格房间地板
                       - "gridroom_curved_black": 黑色网格房间地板
        """
        if floor_type == "default":
            self.ground_plane = self.world.scene.add_default_ground_plane()
        else:
            # 使用网格地板
            self._load_grid_floor(floor_type)
    
    def _set_grid_floor_color(self, prim, color_rgb):
        """
        为网格地板设置颜色（淡蓝色），使用displayColor属性保留网格纹理
        Args:
            prim: 地板 prim
            color_rgb: RGB颜色元组，例如 (0.7, 0.9, 1.0) 表示淡蓝色
        """
        try:
            from pxr import UsdGeom, Gf
            
            # 递归查找所有几何体 prim 并设置displayColor
            def set_display_color_recursive(prim):
                """递归设置所有几何体的displayColor"""
                # 检查是否是几何体
                if prim.IsA(UsdGeom.Gprim):
                    gprim = UsdGeom.Gprim(prim)
                    # 设置displayColor，这会作为颜色叠加，不会覆盖材质纹理
                    gprim.CreateDisplayColorAttr([Gf.Vec3f(color_rgb[0], color_rgb[1], color_rgb[2])])
                
                # 递归处理子prim
                for child in prim.GetChildren():
                    set_display_color_recursive(child)
            
            set_display_color_recursive(prim)
            print(f"✓ 成功设置网格地板显示颜色为淡蓝色 (RGB: {color_rgb})，网格纹理已保留")
            
        except Exception as e:
            print(f"✗ 设置网格地板颜色失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_grid_floor(self, floor_type="gridroom_curved"):
        """
        加载Isaac Sim内置的网格地板
        """
        try:
            from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim
            from pxr import Sdf
            import carb
            import os
            
            # 获取data_path
            from dexrl.global_config import data_path
            
            # 本地地板资源路径映射
            local_floor_assets = {
                "gridroom_curved": f"{data_path}/Assets/IsaacSim/Assets/Isaac/4.2/Isaac/Environments/Grid/gridroom_curved.usd",
                "gridroom_curved_white": f"{data_path}/Assets/IsaacSim/Assets/Isaac/4.2/Isaac/Environments/Grid/gridroom_curved.usd",
                "gridroom_curved_black": f"{data_path}/Assets/IsaacSim/Assets/Isaac/4.2/Isaac/Environments/Grid/gridroom_black.usd",
            }
            if floor_type not in local_floor_assets:
                print(f"未知的地板类型: {floor_type}，使用默认gridroom_curved_white")
                floor_type = "gridroom_curved_white"
            
            # 检查是否已经存在地板
            existing_ground = get_prim_at_path("/World/Ground")
            if existing_ground:
                print("删除现有的地板...")
                delete_prim("/World/Ground")
            
            # 尝试加载本地地板资源
            floor_asset_path = local_floor_assets[floor_type]
            
            print(f"正在加载Isaac Sim内置网格地板: {floor_type}")
            print(f"尝试路径: {floor_asset_path}")
            
            # 检查文件是否存在
            if os.path.exists(floor_asset_path):
                print(f"✓ 找到本地资源文件: {floor_asset_path}")
                try:
                    add_reference_to_stage(floor_asset_path, "/World/Ground")
                    
                    # 检查是否成功加载
                    ground_prim = get_prim_at_path("/World/Ground")
                    if ground_prim:
                        print(f"✓ 成功加载Isaac Sim内置网格地板: {floor_type}")
                        self.ground_plane = ground_prim
                        # 设置淡蓝色，保留网格纹理
                        self._set_grid_floor_color(ground_prim, (0.7, 0.9, 1.0))
                        return
                    else:
                        print(f"✗ 加载失败: 无法找到Ground prim")
                        
                except Exception as load_error:
                    print(f"✗ 加载本地资源失败: {load_error}")
            else:
                print(f"✗ 文件不存在: {floor_asset_path}")
            
            # 如果加载失败，回退到默认地板
            print("回退到默认地板...")
            self.ground_plane = self.world.scene.add_default_ground_plane()
                
        except Exception as e:
            print(f"✗ 加载Isaac Sim内置网格地板时出错: {e}")
            print("回退到默认地板...")
            self.ground_plane = self.world.scene.add_default_ground_plane()

    def load_robot(self):
        self.robot:Robot = self.robot_cls(self.cfg)
        self.world.scene.add(self.robot.articulation)

    def load_objects(self):
        self.obstacle_list = []
        self.table = FixedCuboid(
                name="table",
                prim_path="/World/objects/obstacle_1",
                scale=np.array([1.2, 0.6, 0.05]),
                position=np.array([0.,-0.5, 0.05]),
                color=np.array([0.05, 0.05, 0.05]),
            )
        self.world.scene.add(self.table)
        self.obstacle_list.append(self.table)

        self.red_cube = DynamicCuboid(
            name="RedCube",
            position=np.array([ 0,-0.4, 0.1]),
            prim_path="/World/objects/red_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )
        self.world.scene.add(self.red_cube)
        self.obstacle_list.append(self.red_cube)

        self.green_cube = DynamicCuboid(
            name="GreenCube",
            position=np.array(( 0.1,-0.6, 0.1)),
            prim_path="/World/objects/green_cube",
            size=0.05,
            color=np.array([1, 1, 0]),
        )
        self.world.scene.add(self.green_cube)
        self.obstacle_list.append(self.green_cube)

        self.blue_cube = DynamicCuboid(
            name="BlueCube",
            position=np.array(( 0.4,-0.4, 0.1)),
            prim_path="/World/objects/blue_cube",
            size=0.05,
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
            with rep.get.prims(obj.prim_path) as prims:
                rep.modify.semantics([("class", obj.name)])


    def setup(self):
        self.setup_viewport_camera()
        self.setup_camera()
        self.robot.setup()
        self.reset()

    def reset(self):
        pass
        # self.robot.articulation.set_joint_positions(self.robot.default_joint_pos)

    def setup_viewport_camera(self):
        print(f"self.cfg.viewport_camera_pos_lookat: {self.cfg.viewport_camera_pos_lookat}")
        set_camera_view(eye=self.cfg.viewport_camera_pos_lookat[:3], target=self.cfg.viewport_camera_pos_lookat[3:])

    def close(self):
        pass



    def goto_position_no_yield(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """
        pos = np.array([-translation_target[1],translation_target[0],translation_target[2]])

        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 20)
        rmpflow.set_end_effector_target(pos, orientation_target)

        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, pos)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 20)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            # yield ()

        return False

    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """
        pos = np.array([-translation_target[1],translation_target[0],translation_target[2]])



        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(pos, orientation_target)

        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, pos)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False

    # API
    def open_gripper(self):
        self.current_gripper_pos = 0.04

        articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
        self.robot.articulation.apply_action(articulation_action)
        while not np.allclose(self.robot.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.002):
            yield
        
    # def close_gripper(self):
    #     self.current_gripper_pos = 0 #0.02
    #     articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
    #     self.robot.articulation.apply_action(articulation_action)
    #     stuck_count = 0
    #     while not np.allclose(self.robot.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.004):
    #         last_gripper_pos = self.robot.articulation.get_joint_positions()[7:]
    #         yield
    #         current_gripper_pos = self.robot.articulation.get_joint_positions()[7:]
    #         # print(current_gripper_pos,last_gripper_pos)
    #         if np.allclose(last_gripper_pos, current_gripper_pos, atol=0.001):
    #             # print("gripper stuck")
    #             stuck_count += 1
    #             if stuck_count > 20:
    #                 break
    #         else:
    #             stuck_count = 0
    #         last_gripper_pos = current_gripper_pos

    # def close_gripper(self):
    #     self.current_gripper_pos = 0.02  # 使用0.02而不是0，确保能夹住物体
    #     articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2,joint_indices=(7,8))
    #     stuck_count = 0
    #     while not np.allclose(self.articulation.get_joint_positions()[7:], np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.004):
    #         # 持续应用动作，确保夹爪持续施加力来夹住物体
    #         self.articulation.apply_action(articulation_action)
    #         last_gripper_pos = self.articulation.get_joint_positions()[7:]
    #         yield
    #         current_gripper_pos = self.articulation.get_joint_positions()[7:]
    #         # print(current_gripper_pos,last_gripper_pos)
    #         if np.allclose(last_gripper_pos, current_gripper_pos, atol=0.001):
    #             # print("gripper stuck")
    #             stuck_count += 1
    #             if stuck_count > 20:
    #                 break
    #         else:
    #             stuck_count = 0
    #         last_gripper_pos = current_gripper_pos


    def close_gripper(self):
        # 渐进式关闭夹爪，避免关闭过快导致物体被挤出
        target_gripper_pos = 0.02  # 目标位置，使用 0.02 而不是 0，允许夹爪在遇到物体时继续施加力
        
        # 获取当前夹爪位置
        current_gripper_pos = self.robot.articulation.get_joint_positions()[7:]
        start_gripper_pos = np.mean(current_gripper_pos)  # 起始位置（取两个手指的平均值）
        
        # 计算关闭步长，使用较小的步长使关闭更平滑
        step_size = 0.002  # 每次关闭的步长（2mm）
        num_steps = max(1, int(abs(start_gripper_pos - target_gripper_pos) / step_size))
        
        stuck_count = 0
        max_iterations = 200  # 增加最大迭代次数，确保夹爪有足够时间关闭
        
        for _ in range(max_iterations):
            # 持续应用关闭动作，而不是只应用一次
            articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2, joint_indices=(7,8))
            self.robot.articulation.apply_action(articulation_action)
            
            yield
            
            current_gripper_pos = self.robot.articulation.get_joint_positions()[7:]
            
            # 检查是否已经关闭到目标位置（允许一定的容差）
            if np.allclose(current_gripper_pos, np.array([self.current_gripper_pos, self.current_gripper_pos]), atol=0.004):
                # 已经到达目标位置，继续施加力一段时间以确保夹紧
                for _ in range(10):
                    articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2, joint_indices=(7,8))
                    self.robot.articulation.apply_action(articulation_action)
                    yield
                break
            
            # 检查是否卡住（位置不再变化）
            if not hasattr(self, '_last_gripper_pos_check'):
                self._last_gripper_pos_check = current_gripper_pos.copy()
            else:
                if np.allclose(self._last_gripper_pos_check, current_gripper_pos, atol=0.001):
                    stuck_count += 1
                    if stuck_count > 20:
                        # 即使卡住，也继续施加力一段时间
                        for _ in range(10):
                            articulation_action = ArticulationAction(joint_positions=[self.current_gripper_pos]*2, joint_indices=(7,8))
                            self.robot.articulation.apply_action(articulation_action)
                            yield
                        break
                else:
                    stuck_count = 0
                self._last_gripper_pos_check = current_gripper_pos.copy()
        
        # 清理临时属性
        if hasattr(self, '_last_gripper_pos_check'):
            delattr(self, '_last_gripper_pos_check')

    def move_to_object(self,obj_name):
        obj = self.obj_map[obj_name]
        pos,quat = obj.get_world_pose()
        target_pos = np.array([pos[0],pos[1],pos[2]+0.04])
        yield from self.move_to_pos(target_pos,50)
        target_pos = np.array([pos[0],pos[1],pos[2]+0.0])
        yield from self.move_to_pos(target_pos,50)

    def move_on_object(self,obj_name):
        obj = self.obj_map[obj_name]
        pos,quat = obj.get_world_pose()
        target_pos = np.array([pos[0],pos[1],pos[2]+0.1])
        yield from self.move_to_pos(target_pos,200)


    def move_to_table(self):
        target_pos = np.array([0,-0.5,0.1])
        yield from self.move_to_pos(target_pos,50)

    def move_to_pos(self,pos,time_range=120):
        quat = utils.rot.euler_angles_to_quat([0,np.pi,0])
        yield from self.goto_position(pos,quat,self.robot.articulation,self.robot.rmpflow)
        