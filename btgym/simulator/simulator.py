import btgym
import os
import queue
import multiprocessing

from btgym.utils.logger import set_logger_entry
import numpy as np
import yaml

import omnigibson as og
from omnigibson.simulator import _launch_app
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitiveSet,
)

from btgym.core.action_starter import action_starter_map as action_map
from btgym.core.action_starter import ActionPrimitives, get_grasp_poses_for_object_sticky


from omnigibson.macros import gm
from btgym.utils.logger import log
from omnigibson.robots.tiago import Tiago
from btgym.utils.path import ROOT_PATH
import json
import torch as th
import omnigibson.utils.transform_utils as T

# from btgym.core.curobo import CuRoboMotionGenerator
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator   
import math
from btgym.dataclass.cfg import cfg
from btgym.dataclass.state import state
import os
import btgym.utils.og_utils as og_utils
from bddl.object_taxonomy import ObjectTaxonomy
import time


from pathlib import Path
from PIL import Image
from btgym.molmo.molmo_client import MolmoClient
DIR = Path(__file__).parent
from btgym.dataclass import cfg, state

OBJECT_TAXONOMY = ObjectTaxonomy()

task_list_path = os.path.join(cfg.ASSETS_PATH, 'tasks.txt')
VALID_TASK_LIST = open(task_list_path, 'r').read().splitlines()
VALID_SCENE_LIST = open(os.path.join(cfg.ASSETS_PATH, 'scene_list.txt'), 'r').read().splitlines()
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = False


gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    try:
        for action in ctrl_gen:
            if action!=None:
                env.step(action)
        else:
            og.sim.step()
    except Exception as e:
        log(f"执行控制器时发生错误: {str(e)}")

task_scene_map = json.load(open(f'{ROOT_PATH}/assets/task_to_scenes.json', 'r'))

class Simulator:
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.
    """

    def __init__(self,headless=False):
        gm.HEADLESS = headless
        
        # gm.USE_GPU_DYNAMICS = True
        # gm.ENABLE_FLATCACHE = False
        
        
        self.og_sim = None
        self.current_task_name = None
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        self.robot = None
        self.scene = None
        self.action_primitives = None
        self.curobo_mg = None
        self.camera = None
        self.target_visual = None
        self.reload_count = 0
        self.reload_count_max = 1
        self.batch_size = 4 #32

        # self.action = []
        self.default_move_action = th.tensor([0,0])
        self.default_camera_action = th.tensor([0,0])
        self.default_trunk_action = th.tensor([1])
        self.default_arm_action = th.tensor([-1.5,-0.8,1.7,2.0,-1.0,1.37,1.91])
        self.default_gripper_action = th.tensor([0,0])
        self.move_action = self.default_move_action.clone()
        self.camera_action = self.default_camera_action.clone()
        self.trunk_action = self.default_trunk_action.clone()
        self.arm_action = self.default_arm_action.clone()
        self.gripper_action = self.default_gripper_action.clone()
        if not og.app:
            og.app = _launch_app()
        # self.load_task(task_name)


    def idle(self):
        while True:
            og.sim.step()
    
    def idle_step(self,step_num=1):
        for _ in range(step_num):
            if og.sim:
                og.sim.step()

    # def step(self):
    #     if self.og_sim is not None:
    #         action = th.cat([self.move_action,self.camera_action,self.trunk_action,self.arm_action,self.gripper_action])
    #         self.og_sim.step(action)
            # self.og_sim.step(self.action)

    # def load_empty_scene(self):
    #     config = {
    #         "env": {
    #         },
    #         "scene": {
    #             "type": "Scene",
    #             "trav_map_with_objects": False,  # 不生成导航地图
    #         },
    #         "robots": [],
    #         "objects": [],
    #         "task": {
    #             "type": "DummyTask"
    #         }
    #     }
    #     self.og_sim = og.Environment(configs=config)
        # self.action_primitive = StarterSemanticActionPrimitives(self.og_sim,enable_head_tracking=False)




    def load_task(self, task_name=None):
        if task_name in VALID_TASK_LIST:
            self.load_behavior_task(task_name)


    def load_behavior_task(self, task_name):
        self.current_task_name = task_name
        log(f"load_behavior_task: {task_name}")


        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        # config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
        # config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Update it to run a grocery shopping task
        scene_name = task_scene_map[task_name][0]
        config["scene"]["scene_model"] = scene_name
        log(f'scene: {scene_name}')
        config["scene"]["load_task_relevant_only"] = True
        # config["scene"]["not_load_object_categories"] = ["ceilings"]
        config["task"] = {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": False,
        }
        # config["robot"]["grasping_mode"] = "sticky"
        # gm.USE_GPU_DYNAMICS = True
        # gm.ENABLE_FLATCACHE = False

        self.load_from_config(config)


    def load_custom_task(self, task_name, scene_name=None, scene_file_name=None,is_sample=False,folder_path=None,load_task_relevant_only=True):
        from bddl import config
        # 如果 folder_path没提供,就活动配置路径 默认路径  my_tasks
        if folder_path is None:
            config.ACTIVITY_CONFIGS_PATH = f'{cfg.ASSETS_PATH}/my_tasks'
        else:
            config.ACTIVITY_CONFIGS_PATH = f'{folder_path}'
            cfg.task_folder = folder_path
            
        from omnigibson.utils import bddl_utils
        # 如果任务名不在bddl_utils.BEHAVIOR_ACTIVITIES中，则添加
        if task_name not in bddl_utils.BEHAVIOR_ACTIVITIES:
            bddl_utils.BEHAVIOR_ACTIVITIES.append(task_name)

        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        cfgs = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        if is_sample:
            cfgs["scene"]["scene_model"] = scene_name
        else:   
            cfgs["scene"]["scene_file"] = f'{cfg.task_folder}/{task_name}/{scene_file_name}.json'
        
        if not is_sample:
            cfgs["scene"]["load_task_relevant_only"] = load_task_relevant_only

        cfgs['task'] = {
                "type": "BehaviorTask",
                "activity_name": task_name,
                "activity_definition_id": 0,
                "activity_instance_id": 0,
                "online_object_sampling": is_sample,
            }
        self.load_from_config(cfgs)
        
        # 保存
        output_json_path = f'{cfg.task_folder}/{task_name}/{scene_file_name}.json'
        if is_sample:
            self.og_sim.task.write_task_metadata()
            og.sim.save(json_paths=[output_json_path])
        return output_json_path
        
        

    # def sample_custom_task(self,task_name, scene_name=None):
    #     try:
    #         self.load_custom_task(task_name, scene_name=scene_name, is_sample=True)
    #     except Exception as e:
    #         print(f"Error loading task {task_name}: {str(e)}")
    #         return ''
    #     os.makedirs(f'{cfg.OUTPUTS_PATH}/sampled_tasks',exist_ok=True)
    #     json_path = f'{cfg.OUTPUTS_PATH}/sampled_tasks/{task_name}_{int(time.time())}.json'

    #     self.og_sim.task.write_task_metadata()
    #     # og.sim.write_metadata('scene_file_name',scene_name)
    #     og.sim.save(json_paths=[json_path])
    #     return json_path


    # cys: 采样任务
    # 输入:
    # task_name: 任务的名称。
    # scene_name: 场景的名称（可选）。
    # folder_path: 任务配置文件夹的路径（可选）。
    # output_json_path: 输出 JSON 文件的路径（可选）。
    def sample_custom_task(self,task_name, scene_name=None,folder_path=None,output_json_path=None):
        try:
            self.load_custom_task(task_name, scene_name=scene_name, folder_path=folder_path,is_sample=True)
        except Exception as e:
            print(f"Error loading task {task_name}: {str(e)}")
            return ''
        if output_json_path is None:
            os.makedirs(f'{cfg.OUTPUTS_PATH}/sampled_tasks',exist_ok=True)
            output_json_path = f'{cfg.OUTPUTS_PATH}/sampled_tasks/{task_name}_{int(time.time())}.json'
        else:
            os.makedirs(os.path.dirname(output_json_path),exist_ok=True)
            output_json_path = os.path.join(cfg.task_folder,task_name,output_json_path)
        

        self.og_sim.task.write_task_metadata()
        # og.sim.write_metadata('scene_file_name',scene_name)
        og.sim.save(json_paths=[output_json_path])
        return output_json_path


    def load_scene(self, scene_name):
        config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        config["scene"]["scene_model"] = scene_name
        # config["scene"]["load_task_relevant_only"] = True
        # config["scene"]["not_load_object_categories"] = ["ceilings"]
        self.load_from_config(config)

    def load_from_config(self, config):
        if self.og_sim is not None: 
            og.clear()
            del self.og_sim
            if self.target_visual: del self.target_visual
            del self.action_primitives
            del self.curobo_mg
            # self.reload_count += 1
            # if self.reload_count >= self.reload_count_max:
            #     try:
                    
            #     except:
            #         pass

        self.og_sim = og.Environment(configs=config)

        # 等待几帧确保相机初始化完成
        for _ in range(10):
            self.idle_step()
        
        # 然后再设置相机
        og.sim.enable_viewer_camera_teleoperation()

        self.scene = self.og_sim.scene
        self.robot = self.og_sim.robots[0]
        self.camera = list(self.robot.sensors.values())[0]

        # Allow user to move camera more easily
        # og.sim.enable_viewer_camera_teleoperation()

        self.action_primitives = ActionPrimitives(self)
        self.reset_hand()
        self.set_camera_lookat_robot()

        self.curobo_mg = CuRoboMotionGenerator(self.robot,
            robot_cfg_path=f"{ROOT_PATH}/assets/fetch_description_curobo.yaml",
            debug=False,
            batch_size=self.batch_size)
        self.kinematics_config = self.curobo_mg.mg.robot_cfg.kinematics.kinematics_config

        log("load task: success!")
        

        # from omni.isaac.core.objects import cuboid

        # # Make a target to follow
        # self.target_visual = cuboid.VisualCuboid(
        #     "/World/visual",
        #     position=np.array([0.3, 0, 0.67]),
        #     orientation=np.array([0, 1, 0, 0]),
        #     color=np.array([1.0, 0, 0]),
        #     size=0.05,
        # )

        for i in range(10):
            self.idle_step()


    def load_from_json(self, task_name, json_path):
        config_filename = os.path.join(ROOT_PATH, "assets/fetch_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        config["scene"]["scene_file"] = json_path
        
        config["scene"]["load_task_relevant_only"] = True
        
        self.load_from_config(config)

    # def load_from_json_task(self, json_path, task_name):
    #     pass
        # config_filename = os.path.join(cfg.ASSETS_PATH, "fetch_primitives.yaml")
        # config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        # config["task"] = {
        #     "type": "BehaviorTask",
        #     "activity_name": task_name,
        #     "activity_definition_id": 0,
        #     "activity_instance_id": 0,
        #     "online_object_sampling": True,
        # }


    def reset(self):
        self.og_sim.reset()
        self.target_visual = None

    # def step(self):
    #     if self.control_queue.empty():
    #         pass
    #         # self.og_sim.step(self.action_primitives._empty_action())
    #         # log('robot idle !!!')
    #     else:
    #         action = self.control_queue.get()
    #         if action is not None:
    #             self.og_sim.step(action)
    #         else:
    #             self.og_sim.step(self.action_primitives._empty_action())
    #             # log('robot step !!!')

    def set_viewer_camera_pose(self, position, orientation):
        og.sim.viewer_camera.set_position_orientation(position=position, orientation=orientation)

    def set_camera_lookat_robot(self):
        #设置
        robot_pos, robot_quat = self.robot.get_position_orientation()
        # 计算机器人前方向量
        forward_dir = T.quat2mat(robot_quat)[:3, 0]  # 取旋转矩阵的第一列作为前方向量
        # 计算前方1米的位置
        camera_pos = robot_pos + forward_dir * 1.0
        camera_pos[2] = 2

        # 计算与y轴的夹角
        y_axis = th.tensor([0.0, 1.0, 0.0])
        angle = th.acos(th.dot(-forward_dir, y_axis) / (th.norm(-forward_dir) * th.norm(y_axis)))
        angle = th.sign(forward_dir[0])*angle

        camera_quat = T.euler2quat(th.tensor([0.45,0,angle]))
        og.sim.viewer_camera.set_position_orientation(camera_pos, camera_quat)


    def get_scene_name(self):
        return self.scene.scene_model
    
    def get_robot_pos(self):
        return self.robot.get_position_orientation()[0]

    def get_trav_map(self):
        return self.scene._trav_map

    def get_joint_states(self):
        return self.robot.get_joint_positions()

    def set_joint_states(self,joint_states):
        joint_states = th.tensor(joint_states,device=self.device)
        self.robot.set_joint_positions(joint_states)

    def get_end_effector_pose(self):
        return self.robot.get_eef_position_orientation()

    def get_relative_eef_pose(self):
        return self.robot.get_relative_eef_pose()

    def get_task_objects(self):
        return list(self.og_sim.task.object_instance_to_category.keys())

    def navigate_to_object(self, object_name):
        # object = self.scene.object_registry("name", object_name)        self.reset_hand()
        obj = self.og_sim.task.object_scope[object_name]
        self.action_primitives._navigate_to_obj(obj)
        self.idle_step(10)

        # primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, object)
        # execute_controller(primitive_action, self.og_sim)

    def navigate_to_pos(self, object_name,pos,offset=(1.2,-0.2)):
        # offset = th.tensor(offset,dtype=th.double)
        self.reset_hand()
        obj = self.og_sim.task.object_scope[object_name]
        self.action_primitives._navigate_to_pos(obj,pos,offset=offset)

        self.idle_step(10)


    def get_object_face_tensor(self,object_name,pos,horizontal=True):
        obj = self.og_sim.task.object_scope[object_name]
        face_tensor = self.action_primitives.get_object_face_tensor(obj,pos,horizontal=horizontal)
        return face_tensor






    def grasp_object(self, object_name):
        object = self.og_sim.task.object_scope[object_name]
        primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, object)
        execute_controller(primitive_action, self.og_sim)
        
    def place_ontop_object(self, object_name):
        object = self.og_sim.task.object_scope[object_name]
        primitive_action = self.action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, object)
        execute_controller(primitive_action, self.og_sim)


    def add_control(self,control):
        self.control_queue.put(control)


    def pose_to_local(self,pose):
        target_pos = th.tensor(pose[:3],device=self.device)
        target_euler = th.tensor(pose[3:],device=self.device)
        target_quat = T.euler2quat(target_euler)

        robot_pos, robot_quat = self.robot.get_position_orientation()
        target_pose = th.zeros((4, 4))
        target_pose[3, 3] = 1.0
        target_pose[:3, :3] = T.quat2mat(target_quat)
        target_pose[:3, 3] = target_pos
        inv_robot_pose = th.eye(4)
        inv_robot_ori = T.quat2mat(robot_quat).T
        inv_robot_pose[:3, :3] = inv_robot_ori
        inv_robot_pose[:3, 3] = -inv_robot_ori @ robot_pos
        target_pose = inv_robot_pose @ target_pose
        target_pos = target_pose[:3, 3]
        target_quat = T.mat2quat(target_pose[:3, :3])
        target_euler = T.quat2euler(target_quat)
        target_pose = th.cat([target_pos, target_euler], dim=0).cpu().numpy()
        return target_pose

    def reach_pose(self, pose, is_local=True):
        robot = self.robot
        target_pos = th.tensor(pose[:3],device=self.device)
        offset = th.tensor([0.0,0.0,0.0],device=self.device)
        target_pos = target_pos + offset
        target_euler = th.tensor(pose[3:],device=self.device)
        target_quat = T.euler2quat(target_euler)

        # 在pos周围生成更多的目标点
        pos_list = []

        scales = th.tensor([[0.000, 0.000, 0.000]], device=self.device).expand(self.batch_size-1, -1)  # [batch_size, 3]
        offsets = th.randn(self.batch_size-1, 3, device=self.device) * scales  # [batch_size, 3]

        pos_tensor = target_pos.unsqueeze(0).expand(self.batch_size-1, -1) + offsets  # [batch_size, 3]
        pos_tensor = th.cat([pos_tensor, target_pos.unsqueeze(0)], dim=0)
        quat_tensor = target_quat.unsqueeze(0).expand(self.batch_size, -1)

        # 确保pos_tensor和quat_tensor的形状正确
        # pos_tensor = target_pos.unsqueeze(0).expand(2, -1)  # 使用两个目标位置
        # quat_tensor = target_quat.unsqueeze(0).expand(2, -1)  # 使用两个目标方向

        self.open_gripper()

        # 如果机器人接近关节限制，则调整关节位置
        jp = robot.get_joint_positions(normalized=True)
        if not th.all(th.abs(jp)[:-2] < 0.97):
            new_jp = jp.clone()
            new_jp[:-2] = th.clamp(new_jp[:-2], min=-0.95, max=0.95)
            robot.set_joint_positions(new_jp, normalized=True)
        self.idle_step(10)

        try:  
            successes, paths = self.curobo_mg.compute_trajectories(pos_tensor, quat_tensor, is_local=is_local)
        except Exception as e:
            log(f'error: {str(e)}')
            log(f"IK求解失败!")
            return False

        # 找出所有成功的位置和路径
        success_indices = th.where(successes)[0]
        valid_pos_tensor = pos_tensor[success_indices]

        # 得到每个pos的cost
        dist_cost = th.norm(valid_pos_tensor - target_pos, dim=1)
        min_cost_index = th.argmin(dist_cost)
        success_idx = success_indices[min_cost_index]
        success_path = paths[success_idx]
        target_pos = valid_pos_tensor[min_cost_index]

        # 执行轨迹
        joint_trajectory = self.curobo_mg.path_to_joint_trajectory(success_path)
        joint_positions = self.robot.get_joint_positions()

        action = th.zeros(self.robot.n_joints)
        for time_i,jp in enumerate(joint_trajectory):
            action[2:4] = self.camera_action
            # action[2] = joint_positions[3]
            # action[3] = joint_positions[5]
            action[4:] = jp
            self.og_sim.step(action.to('cpu'))
        self.idle_step(20)

        log(f"尝试位置: {target_pos.tolist()}")  # 打印当前尝试的位置
        return True

    def grasp_object_by_pose(self, pose, object_name, is_local=True):
        robot = self.robot
        

        target_pos = th.tensor(pose[:3],device=self.device)
        # offset = th.tensor([0.0,0.0,0.02],device=self.device)
        # target_pos = target_pos + offset
        target_euler = th.tensor(pose[3:],device=self.device)
        target_quat = T.euler2quat(target_euler)

        # 根据欧拉角计算单位向量
        # 欧拉角为 [roll, pitch, yaw]
        # roll: 绕x轴旋转的角度
        # pitch: 绕y轴旋转的角度
        # yaw: 绕z轴旋转的角度
        x = th.cos(target_euler[1]) * th.cos(target_euler[2])
        y = th.cos(target_euler[1]) * th.sin(target_euler[2]) 
        z = th.sin(target_euler[1])
        unit_vector = th.tensor([x, y, z], device=self.device)
        unit_vector = unit_vector / th.norm(unit_vector)  # 归一化

        # 从0.02到-0.02
        offset_tensor = unit_vector.unsqueeze(0).expand(self.batch_size, -1) * th.linspace(0.02, -0.02, self.batch_size, device=self.device).unsqueeze(1)
        pos_tensor = target_pos.unsqueeze(0).expand(self.batch_size, -1) + offset_tensor
        quat_tensor = target_quat.unsqueeze(0).expand(self.batch_size, -1)

        self.open_gripper()

        # 如果机器人接近关节限制，则调整关节位置
        jp = robot.get_joint_positions(normalized=True)
        if not th.all(th.abs(jp)[:-2] < 0.97):
            new_jp = jp.clone()
            new_jp[:-2] = th.clamp(new_jp[:-2], min=-0.95, max=0.95)
            robot.set_joint_positions(new_jp, normalized=True)
        self.idle_step(20)

        try:  
            successes, paths = self.curobo_mg.compute_trajectories(pos_tensor, quat_tensor, is_local=is_local)
        except Exception as e:
            log(f'error: {str(e)}')
            log(f"IK求解失败!")
            return False

        # 找出第一个成功的位置和路径
        if not th.any(successes):
            log(f"IK求解失败!")
            return False
        
        success_idx = th.where(successes)[0][0]  # 获取第一个成功的索引
        target_pos = pos_tensor[success_idx].unsqueeze(0)  # 添加维度以保持张量形状
        # 得到每个pos的cost
        # dist_cost = th.norm(valid_pos_tensor - target_pos, dim=1)
        # min_cost_index = th.argmin(dist_cost)
        # success_idx = success_indices[min_cost_index]
        success_path = paths[success_idx]

        # 执行轨迹
        joint_trajectory = self.curobo_mg.path_to_joint_trajectory(success_path)
        joint_positions = self.robot.get_joint_positions()
        # target_jp = joint_trajectory[-1]

        # joint_positions[2] = target_jp[0]
        # joint_positions[4] = target_jp[1]
        # joint_positions[6:] = target_jp[2:]
        # self.set_joint_states(joint_positions)

        action = th.zeros(self.robot.n_joints)
        for time_i,jp in enumerate(joint_trajectory):
            action[2:4] = self.camera_action
            # action[2] = joint_positions[3]
            # action[3] = joint_positions[5]
            action[4:] = jp
            self.og_sim.step(action.to('cpu'))
        self.idle_step(40)

        log(f"尝试位置: {target_pos.tolist()}")  # 打印当前尝试的位置
                
        self.close_gripper()

        object_name = self.og_sim.task.load_task_metadata()['inst_to_name'][object_name]
        # 检测是否抓起了物体
        # 目前存在问题: curobo碰撞总是发生
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        # log(f"obj_in_hand: {obj_in_hand}")
        if obj_in_hand is not None and obj_in_hand.name == object_name:  # 如果有接触
            log(f"检测到物体接触 {obj_in_hand.name}")
            return True  # 成功找到并执行了轨迹
        else:
            log("未检测到物体接触")
            return False


    def move_hand_linearly(self,dir,is_local=True,distance=0.5,ignore_obj_in_hand=True):
        action = self.action_primitives._move_hand_linearly(dir,distance=distance, ignore_obj_in_hand=ignore_obj_in_hand)
        execute_controller(action, self.og_sim)
        self.idle_step(20)


    def open_object_by_pose_by_sticky(self, pose, object_name, is_local=True):
        robot = self.robot
        

        target_pos = th.tensor(pose[:3],device=self.device)
        # offset = th.tensor([0.0,0.0,0.02],device=self.device)
        # target_pos = target_pos + offset
        # target_euler = th.tensor(pose[3:],device=self.device)
        # 从前到后
        target_euler = th.tensor([0,0,math.pi/2],device=self.device)
        target_quat = T.euler2quat(target_euler)

        # 根据欧拉角计算单位向量
        # 欧拉角为 [roll, pitch, yaw]
        # roll: 绕x轴旋转的角度
        # pitch: 绕y轴旋转的角度
        # yaw: 绕z轴旋转的角度
        x = th.cos(target_euler[1]) * th.cos(target_euler[2])
        y = th.cos(target_euler[1]) * th.sin(target_euler[2]) 
        z = th.sin(target_euler[1])
        unit_vector = th.tensor([x, y, z], device=self.device)
        unit_vector = unit_vector / th.norm(unit_vector)  # 归一化

        # 从0.02到-0.02
        offset_tensor = unit_vector.unsqueeze(0).expand(self.batch_size, -1) * th.linspace(0.02, -0.02, self.batch_size, device=self.device).unsqueeze(1)
        pos_tensor = target_pos.unsqueeze(0).expand(self.batch_size, -1) + offset_tensor
        quat_tensor = target_quat.unsqueeze(0).expand(self.batch_size, -1)

        self.open_gripper()

        # 如果机器人接近关节限制，则调整关节位置
        jp = robot.get_joint_positions(normalized=True)
        if not th.all(th.abs(jp)[:-2] < 0.97):
            new_jp = jp.clone()
            new_jp[:-2] = th.clamp(new_jp[:-2], min=-0.95, max=0.95)
            robot.set_joint_positions(new_jp, normalized=True)
        self.idle_step(20)
        

        try:  
            successes, paths = self.curobo_mg.compute_trajectories(pos_tensor, quat_tensor, is_local=is_local)
        except Exception as e:
            log(f'error: {str(e)}')
            log(f"IK求解失败!")
            return False

        # 找出第一个成功的位置和路径
        if not th.any(successes):
            log(f"IK求解失败!")
            return False
        
        success_idx = th.where(successes)[0][0]  # 获取第一个成功的索引
        target_pos = pos_tensor[success_idx].unsqueeze(0)  # 添加维度以保持张量形状
        # 得到每个pos的cost
        # dist_cost = th.norm(valid_pos_tensor - target_pos, dim=1)
        # min_cost_index = th.argmin(dist_cost)
        # success_idx = success_indices[min_cost_index]
        success_path = paths[success_idx]

        # 执行轨迹
        joint_trajectory = self.curobo_mg.path_to_joint_trajectory(success_path)
        joint_positions = self.robot.get_joint_positions()
        # target_jp = joint_trajectory[-1]

        # joint_positions[2] = target_jp[0]
        # joint_positions[4] = target_jp[1]
        # joint_positions[6:] = target_jp[2:]
        # self.set_joint_states(joint_positions)

        action = th.zeros(self.robot.n_joints)
        for time_i,jp in enumerate(joint_trajectory):
            action[2:4] = self.camera_action
            # action[2] = joint_positions[3]
            # action[3] = joint_positions[5]
            action[4:] = jp
            self.og_sim.step(action.to('cpu'))
        self.idle_step(40)

        log(f"尝试位置: {target_pos.tolist()}")  # 打印当前尝试的位置
                
        self.close_gripper()

        object_name = self.og_sim.task.load_task_metadata()['inst_to_name'][object_name]
        # 检测是否抓起了物体
        # 目前存在问题: curobo碰撞总是发生
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        # log(f"obj_in_hand: {obj_in_hand}")
        if obj_in_hand is not None and obj_in_hand.name == object_name:  # 如果有接触
            log(f"检测到物体接触 {obj_in_hand.name}")
            return True  # 成功找到并执行了轨迹
        else:
            log("未检测到物体接触")
            return False


    def place_object_by_pose(self, pose, object_name, is_local=True):
        robot = self.robot
        

        target_pos = th.tensor(pose[:3],device=self.device)
        # offset = th.tensor([0.0,0.0,0.02],device=self.device)
        # target_pos = target_pos + offset
        target_euler = th.tensor(pose[3:],device=self.device)
        target_quat = T.euler2quat(target_euler)

        # 根据欧拉角计算单位向量
        # 欧拉角为 [roll, pitch, yaw]
        # roll: 绕x轴旋转的角度
        # pitch: 绕y轴旋转的角度
        # yaw: 绕z轴旋转的角度
        x = th.cos(target_euler[1]) * th.cos(target_euler[2])
        y = th.cos(target_euler[1]) * th.sin(target_euler[2]) 
        z = th.sin(target_euler[1])
        unit_vector = th.tensor([x, y, z], device=self.device)
        unit_vector = unit_vector / th.norm(unit_vector)  # 归一化

        # 从0.02到-0.02
        offset_tensor = unit_vector.unsqueeze(0).expand(self.batch_size, -1) * th.linspace(0.02, -0.02, self.batch_size, device=self.device).unsqueeze(1)
        pos_tensor = target_pos.unsqueeze(0).expand(self.batch_size, -1) + offset_tensor
        quat_tensor = target_quat.unsqueeze(0).expand(self.batch_size, -1)

        self.close_gripper()

        # 如果机器人接近关节限制，则调整关节位置
        jp = robot.get_joint_positions(normalized=True)
        if not th.all(th.abs(jp)[:-2] < 0.97):
            new_jp = jp.clone()
            new_jp[:-2] = th.clamp(new_jp[:-2], min=-0.95, max=0.95)
            robot.set_joint_positions(new_jp, normalized=True)
        self.idle_step(20)
        
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        try:  
            successes, paths = self.curobo_mg.compute_trajectories(pos_tensor, quat_tensor, is_local=is_local,attached_obj=obj_in_hand)
        except Exception as e:
            log(f'error: {str(e)}')
            log(f"IK求解失败!")
            return False

        # 找出第一个成功的位置和路径
        if not th.any(successes):
            log(f"IK求解失败!")
            return False
        
        success_idx = th.where(successes)[0][0]  # 获取第一个成功的索引
        target_pos = pos_tensor[success_idx].unsqueeze(0)  # 添加维度以保持张量形状
        # 得到每个pos的cost
        # dist_cost = th.norm(valid_pos_tensor - target_pos, dim=1)
        # min_cost_index = th.argmin(dist_cost)
        # success_idx = success_indices[min_cost_index]
        success_path = paths[success_idx]

        # 执行轨迹
        joint_trajectory = self.curobo_mg.path_to_joint_trajectory(success_path)
        joint_positions = self.robot.get_joint_positions()
        # target_jp = joint_trajectory[-1]

        # joint_positions[2] = target_jp[0]
        # joint_positions[4] = target_jp[1]
        # joint_positions[6:] = target_jp[2:]
        # self.set_joint_states(joint_positions)

        action = th.zeros(self.robot.n_joints)
        for time_i,jp in enumerate(joint_trajectory):
            action[2:4] = self.camera_action
            # action[2] = joint_positions[3]
            # action[3] = joint_positions[5]
            action[4:] = jp
            self.og_sim.step(action.to('cpu'))
        self.idle_step(40)

        log(f"尝试位置: {target_pos.tolist()}")  # 打印当前尝试的位置
                
        self.open_gripper()
        
        obj_in_hand = self.action_primitives._get_obj_in_hand()
        if obj_in_hand is None:
            return True
        else:
            return False
    

        # object_name = self.og_sim.task.load_task_metadata()['inst_to_name'][object_name]
        # # 检测是否抓起了物体
        # # 目前存在问题: curobo碰撞总是发生
        # obj_in_hand = self.action_primitives._get_obj_in_hand()
        # # log(f"obj_in_hand: {obj_in_hand}")
        # if obj_in_hand is not None and obj_in_hand.name == object_name:  # 如果有接触
        #     log(f"检测到物体接触 {obj_in_hand.name}")
        #     return True  # 成功找到并执行了轨迹
        # else:
        #     log("未检测到物体接触")
        #     return False

    def reset_hand(self):
        jp = self.get_joint_states()
        self.set_joint_states([
                0.0,
                0.0,  # wheels
                1.0,  # trunk
                jp[3],
                -1.5,
                jp[5],  # head
                -0.8,
                1.7,
                2.0,
                -1.0,
                1.36904,
                0,  # arm
                jp[-2],
                jp[-1],  # gripper
            ])
        self.robot.set_joint_velocities(th.zeros(self.robot.n_joints))
        self.idle_step(1)
        # self.idle_step(20)
        
    def save_camera_image(self, output_path):
        """
        保存机器人视角的RGB图像
        Args:
            output_path: 输出图像的路径，例如 "robot_view.png"
        """
        rgb_obs = list(self.robot.get_obs()[0].values())[0]['rgb'].cpu().numpy()
        # 将numpy数组转换为PIL图像并保存
        from PIL import Image
        img = Image.fromarray(rgb_obs)
        img = img.convert('RGB')  # 将RGBA转换为RGB
        img.save(output_path, format='PNG')
    
    def set_target_visual_pose(self, pose,size = 0.1):

        pos = th.tensor(pose[:3],device=self.device)
        euler = th.tensor(pose[3:],device=self.device)
        if euler is None or len(euler) == 0:
            quat = T.euler2quat(th.tensor([0,math.pi/2,0], dtype=th.float32))
        else:
            quat = T.euler2quat(euler)
        
        if self.target_visual is None:
            from omni.isaac.core.objects import cuboid

            # Make a target to follow
            self.target_visual = cuboid.VisualCuboid(
                "/World/visual",
                position=np.array([0.3, 0, 0.67]),
                orientation=np.array([0, 1, 0, 0]),
                color=np.array([1.0, 0, 0]),
                size=size,
            )
        self.target_visual.set_world_pose(pos, quat)

    def start_follow_visual(self,euler=(0,math.pi/2,math.pi/2)):
        n = 0
        while True:
            cube_position, cube_orientation = self.target_visual.get_world_pose()
            success = self.reach_pose(self.pose_to_local(th.cat([th.tensor(cube_position), th.tensor(euler)])))
            if not success:
                n+=1
                if n > 5:
                    self.reset_hand()
                    n = 0

    def set_camera_lookat_pos(self,pos):
        z_limits = [-1.57, 1.57]
        y_limits = [-0.76, 1.45]
        
        target_position = th.tensor(pos)
        current_camera_position = self.camera.get_position_orientation()[0]
        current_robot_quat = self.robot.get_position_orientation()[1]
        robot_euler = T.quat2euler(current_robot_quat)
        robot_euler[0] = 0.0
        robot_euler[1] = 0.0
        direction = target_position - current_camera_position
        
        # 考虑机器人朝向x轴正方向的情况
        cos_yaw = th.cos(robot_euler[2])
        sin_yaw = th.sin(robot_euler[2])
        
        # 修改：交换x和y的角色，因为机器人朝向x轴
        rel_x = direction[1] * cos_yaw - direction[0] * sin_yaw
        rel_y = direction[0] * cos_yaw + direction[1] * sin_yaw
        rel_z = direction[2]
        
        target_z_angle = th.atan2(rel_x, rel_y)
        distance_horizontal = th.sqrt(rel_x**2 + rel_y**2)
        target_y_angle = -th.atan2(rel_z, distance_horizontal)
        
        target_z_angle = th.clamp(target_z_angle, min=z_limits[0], max=z_limits[1])
        target_y_angle = th.clamp(target_y_angle, min=y_limits[0], max=y_limits[1])
        
        joint_state = self.get_joint_states()
        joint_state[3] = target_z_angle
        joint_state[5] = target_y_angle
        self.robot.set_joint_positions(joint_state)
        self.robot.set_joint_velocities(th.zeros(self.robot.n_joints))
        print(f"target_z_angle: {target_z_angle}, target_y_angle: {target_y_angle}")
        self.camera_action = th.tensor([target_z_angle, target_y_angle])

        # action = th.zeros(self.robot.n_joints)
        # joint_positions = self.robot.get_joint_positions()
        # self.camera_action = th.tensor([target_z_angle, target_y_angle])
        # action[2:4] = self.camera_action
        # action[4] = joint_positions[2]
        # action[5] = joint_positions[4]
        # action[6:] = joint_positions[6:]

        # for i in range(20):
        #     self.og_sim.step(action.to('cpu'))
        # self.set_camear_lookat_pos_ori(pos)

    #TODO 目前无法使物体在相机的正中心，有待修正
    def set_camear_lookat_pos_ori(self,pos):
        pos = [0,0,1]
        v1 = [0,0,-1]
        target_pos = th.tensor(pos)
        current_pos = self.camera.get_position_orientation()[0]
        v2 = target_pos - current_pos


        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # 计算旋转轴
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        
        # 计算旋转角度
        angle = np.arccos(np.dot(v1, v2))
        
        # 计算四元数的各个分量
        w = math.cos(angle / 2.0)
        x = axis[0] * math.sin(angle / 2.0)
        y = axis[1] * math.sin(angle / 2.0)
        z = axis[2] * math.sin(angle / 2.0)
        # 将旋转矩阵转换为四元数
        quat = th.tensor([w,x,y,z])
        
        self.camera.set_position_orientation(current_pos, quat)



    def get_camera_info_server(self):
        sensor = list(self.robot.sensors.values())[0]
        intrinsics = og_utils.get_cam_intrinsics(sensor).reshape(-1)
        # intrinsics = sensor.intrinsic_matrix.reshape(-1).cpu().numpy()
        pose = sensor.get_position_orientation()
        extrinsics = T.pose_inv(T.pose2mat(pose)).reshape(-1).cpu().numpy()

        return {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }

    def get_camera_info(self):
        sensor = list(self.robot.sensors.values())[0]
        intrinsics = og_utils.get_cam_intrinsics(sensor)
        # intrinsics = sensor.intrinsic_matrix.reshape(-1).cpu().numpy()
        pose = sensor.get_position_orientation()
        extrinsics = T.pose_inv(T.pose2mat(pose))

        return {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }


    def get_obs_server(self):
        sensor_obs = list(self.robot.sensors.values())[0].get_obs()
        rgb_obs = sensor_obs[0]['rgb'][:,:,:3].cpu().numpy().tobytes()
        depth_obs = sensor_obs[0]['depth_linear'].cpu().numpy().tobytes()
        seg_obs = sensor_obs[0]['seg_semantic'].cpu().numpy().tobytes()
        seg_info = json.dumps(sensor_obs[1]['seg_semantic'])
        proprio_obs = self.robot.get_proprioception()[0].cpu().numpy()

        return {
            'rgb': rgb_obs,
            'depth': depth_obs,
            'seg_semantic': seg_obs,
            'seg_info': seg_info,
            'proprio': proprio_obs
        }

    def get_obs(self):
        sensor_obs = list(self.robot.sensors.values())[0].get_obs()
        rgb_obs = sensor_obs[0]['rgb'][:,:,:3].cpu().numpy()
        depth_obs = sensor_obs[0]['depth_linear'].cpu().numpy()
        seg_obs = sensor_obs[0]['seg_semantic'].cpu().numpy()
        seg_info = json.dumps(sensor_obs[1]['seg_semantic'])
        proprio_obs = self.robot.get_proprioception()[0].cpu().numpy()

        return {
            'rgb': rgb_obs,
            'depth': depth_obs,
            'seg_semantic': seg_obs,
            'seg_info': seg_info,
            'proprio': proprio_obs
        }


    def get_available_objects(self):
        exclude_prefix = ['wall', 'floor', 'ceilings']
        fixed_object_names = set(self.scene.fixed_objects)

        exclude_objects = []
        fixed_objects = []
        moveable_objects = []
        for obj in self.scene.objects:
            category = obj.category
            if any(category.startswith(prefix) for prefix in exclude_prefix):
                exclude_objects.append(obj.name)
            elif obj.name in fixed_object_names:
                fixed_objects.append(obj.name)
                synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
                print(f'fixed: {synset}')
            else:
                moveable_objects.append(obj.name)
                synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
                print(f'moveable: {synset}')

        return {'moveable_objects': moveable_objects,
                'fixed_objects': fixed_objects}

    def get_object_pos(self,object_name):
        obj = self.og_sim.task.object_scope[object_name]
        grasp_poses = get_grasp_poses_for_object_sticky(obj)
        object_pos = grasp_poses[0][0][0]
        return {'pos': object_pos}

    def get_object_pos_by_pose(self,object_name):
        obj = self.og_sim.task.object_scope[object_name]
        pose = obj.get_position_orientation()
        object_pos = pose[0]
        return {'pos': object_pos}


    def close_gripper(self):
        self.gripper_control(open=False)
        self.kinematics_config.lock_jointstate.position = th.tensor([0., 0.],device=self.kinematics_config.lock_jointstate.position.device)

    def open_gripper(self):
        self.gripper_control(open=True)
        self.kinematics_config.lock_jointstate.position = th.tensor([0.05, 0.05],device=self.kinematics_config.lock_jointstate.position.device)

    def gripper_control(self, open=True):
        joint_positions = self.robot.get_joint_positions()
        action = th.zeros(self.robot.n_joints)
        action[2:4] = self.camera_action
        # action[2] = joint_positions[3]
        # action[3] = joint_positions[5]
        action[4] = joint_positions[2]
        action[5] = joint_positions[4]
        action[6:-2] = joint_positions[6:-2]
        gripper_target = 0.05 if open else 0.0
        action[-2:] = gripper_target
        for _ in range(20):
            self.og_sim.step(action.to('cpu'))
        self.gripper_open = open

        self.idle_step(20)

    def close(self):
        og.shutdown()
        
    ####################################################################################
    
    # 传入图片和相机信息，返回全局坐标系下的目标点
    def keypoint_proposal(self,query,img_dir=None,file_name='camera_0_rgb.png'):
        img_dir = DIR.parent / '../exps_bt_learning/outputs' if img_dir is None else img_dir
        
        obs = self.get_obs()
        rgb_img = Image.fromarray(obs['rgb'])
        camera_info = self.get_camera_info()

        
        rgb_img = Image.fromarray(obs['rgb'])
        
        rgb_img.save(f'{img_dir}/{file_name}')
        molmo_client = MolmoClient()
        point = molmo_client.get_grasp_pose_by_molmo(query,dir=img_dir,point_img_path=f'{img_dir}/{file_name}')
        if point:
            target_pos = og_utils.pixel_to_world(obs, camera_info, point[0], point[1])
            print(f"target_pos: {target_pos}")
            # self.navigate_to_pos(object_name=cfg.target_object_name,pos=target_pos,offset=(0.9,-0.2))
        else:
            target_pos = None
        return target_pos


    # 用于水平抓取获取 抓取点
    def eef_reach_pos(self,pos,horizontal=True,grounding=False,obj_face_tensor = None):
        target_pos = pos
        
        if obj_face_tensor==None:obj_face_tensor = th.tensor([0,1,0.])
        
        if horizontal:
            # obj_face_tensor = th.tensor([0,1,0.])
            
            yaw = math.atan2(-obj_face_tensor[1],obj_face_tensor[0])

            target_pos = target_pos+obj_face_tensor*0.1
            target_euler = [math.pi/2, 0, yaw]

            # 可视化目标点
            self.set_target_visual_pose([*target_pos,0,0,0],size=0.05)
            self.idle_step(10)

            target_local_pose = self.pose_to_local([*target_pos, *target_euler])
            success = self.reach_pose(target_local_pose,is_local=True)
        else:
            
            if not grounding:
                # 确定抓取的方向为 相机位置到 目标位置的方向
                camera_pos = self.camera.get_position_orientation()[0] # 0为位置,1为四元数
                grasp_direction = target_pos - camera_pos
                grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)

                target_pos = target_pos + grasp_direction * 0.02 # target_pos 为朝着 grasp_direction 方向的 1厘米 处
            
            # if grounding， just reach the target_pos
            state.target_local_pose = self.pose_to_local(target_pos.tolist()+state.target_euler)
            success = self.grasp_object_by_pose(state.target_local_pose,object_name=cfg.target_object_name)
        

    def move_hand_forward(self,distance=0.5,obj_face_tensor=None):
        if obj_face_tensor==None:
            obj_face_tensor=th.tensor([0,1,0.])
        self.move_hand_linearly(dir=-obj_face_tensor,distance=distance)


    def move_hand_backward(self,distance=0.3,obj_face_tensor=None):
        if obj_face_tensor==None:
            obj_face_tensor=th.tensor([0,1,0.])
        self.move_hand_linearly(dir=obj_face_tensor,distance=distance,ignore_obj_in_hand=True)
    
    ####################################################################################



if __name__ == "__main__":
    set_logger_entry(__file__)
    
    simulator = Simulator()
    # simulator.load_custom_task('test_task',scene_file_name='scene_file_0')
    from btgym.dataclass.cfg import cfg
    cfg.task_name = "task1"
    cfg.scene_file_name='scene_file_0'

    simulator.load_custom_task(task_name=cfg.task_name, scene_file_name=cfg.scene_file_name)

    object_name = 'apple.n.01_1'
    
    for i in range(10):
        simulator.navigate_to_object(object_name=object_name)
        grasp_pos = simulator.get_object_pos(object_name)['pos'].tolist()
        simulator.set_target_visual_pose([*grasp_pos,0,0,0])
        # simulator.set_camera_lookat_pos(grasp_pos)
        state.target_local_pose = simulator.pose_to_local(grasp_pos+state.target_euler)
        print(f"local pose: {state.target_local_pose}")
        success = simulator.grasp_object_by_pose(state.target_local_pose,object_name=object_name)
        if success:
            print(f"第{i}次尝试成功")
            break

        print(f"第{i}次尝试失败")

    # grasp_pos = [ 0.7103, -3.6875,  0.8163, 0,0,0]
    # 获取物体中心点
    # grasp_pos转为list

    
    # simulator.grasp_object(object_name=object_name)
    # object_name = 'coffee_table.n.01_1'
    # simulator.navigate_to_object(object_name=object_name)
    # simulator.place_ontop_object(object_name=object_name)
    
    # simulator.load_behavior_task('putting_shoes_on_rack')

    # print(simulator.get_object_pos('shoe_1'))

    # simulator.get_obs()
    # simulator.get_camera_info()
    # pos = [0,1,1]
    # simulator.set_target_visual_pose([*pos,0,0,0])
    # simulator.set_camera_lookat_pos(pos)


    # simulator.load_behavior_task('putting_shoes_on_rack')
    # simulator.get_obs()
    # simulator.get_camera_info()
    # pos = [0,1,1]
    # simulator.set_target_visual_pose([*pos,0,0,0])
    # simulator.set_camera_lookat_pos(pos)

    # simulator = Simulator('Rs_int')
    # simulator = Simulator('putting_shoes_on_rack')
    # simulator.load_behavior_task_by_name('putting_shoes_on_rack')
    # simulator.init_action_primitives()
    # gm.USE_GPU_DYNAMICS = True
    # gm.ENABLE_FLATCACHE = False
    #adding_chemicals_to_hot_tub
    # simulator.load_behavior_task_by_name('folding_clothes')

    # scene_name = simulator.get_scene_name()
    # print(f"当前场景名称: {scene_name}")

    # robot_pos = simulator.get_robot_pos()
    # print(f"机器人位置: {robot_pos}")
    # simulator.get_obs()
    # camera_info = simulator.get_camera_info()
    # print(f"相机信息: {camera_info}")

    # available_objects = simulator.get_available_objects()
    # print(f"可用的物体: {available_objects}")
    simulator.idle()
    # simulator.do_task()


# error
# tidy_your_garden

# correct
# buy_a_keg
# lighting_fireplace
# bringing_in_mail
# setting_up_room_for_games 把东西放抽屉里