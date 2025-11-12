import os
import queue
import multiprocessing

from btgym.utils.logger import set_logger_entry
import numpy as np
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from btgym.utils.logger import log
from omnigibson.robots.tiago import Tiago
from btgym.utils.path import ROOT_PATH
import json
import torch as th
import omnigibson.utils.transform_utils as T


gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

task_scene_map = json.load(open(f'{ROOT_PATH}/assets/task_to_scenes.json', 'r'))

class Simulator:
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.

    It loads Rs_int with a robot, and the robot picks and places a bottle of cologne.

    valid task_list:
store_firewood
re_shelving_library_books
putting_in_a_hot_tub
place_houseplants_around_your_home
buy_boxes_for_packing
set_up_a_home_office_in_your_garage
getting_package_from_post_office
buying_office_supplies
bringing_in_wood
shopping_at_warehouse_stores
setting_up_room_for_games
putting_shoes_on_rack
buy_alcohol
paying_for_purchases
buy_candle_making_supplies
carrying_out_garden_furniture
buy_a_keg
bringing_newspaper_in
buy_a_good_avocado
buy_dog_food
moving_boxes_to_storage
lighting_fireplace
picking_up_take_out_food
laying_tile_floors
collecting_childrens_toys
rearrange_your_room
taking_trash_outside
collecting_mail_from_the_letterbox
fold_a_tortilla
buy_basic_garden_tools
collecting_wood

    """

    def __init__(self):
        self.og_sim = None
        self.current_task_name = None
        # Load the config
        self.load_behavior_task_by_name('collecting_childrens_toys')

        
        self.null_control = np.zeros(self.robot.action_space.shape)
        self.control_queue = queue.Queue()
        # # Allow user to move camera more easily
        self.idle_control = np.array([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.1700,  0.8585, -0.1485,
         1.8101,  1.6337,  0.1376, -1.3249, -0.6841,  0.0450,  0.0450,  0.8585,
        -0.1485,  1.8101,  1.6337,  0.1376, -1.3249, -0.6841,  0.0450,  0.0450])

        self.get_task_list()


    def get_task_list(self):
        plan_folder = f'{ROOT_PATH}/../outputs/bddl_planning/success'
        self.task_list = os.listdir(plan_folder)
        # for plan_file in os.listdir(plan_folder):
        #     log("执行task: " + plan_file)
        #     execute_task_single(plan_file, f'{plan_folder}/{plan_file}')

        # task_name = 'putting_shoes_on_rack'
        # plan_file = f'{plan_folder}/{task_name}'
        # execute_task_single(task_name, plan_file)

    def load_simple_scene(self):
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Update it to create a custom environment and run some actions
        config["scene"]["scene_model"] = "Rs_int"
        config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
        config["objects"] = [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "rjgmmy",
                "scale": [0.3, 0.3, 0.3],
                "position": [-0.7, 0.5, 0.2],
                "orientation": [0, 0, 0, 1],
            },
        ]
        # Load the environment
        if self.og_sim is None:
            self.og_sim = og.Environment(configs=config)
        else:
            self.og_sim.reload(configs=config)
        self.scene = self.og_sim.scene
        self.robot:Tiago = self.og_sim.robots[0]
        og.sim.enable_viewer_camera_teleoperation()

    def load_behavior_task_by_index(self, task_index):
        task_name = self.task_list[task_index]
        self.load_behavior_task_by_name(task_name)

    def load_behavior_task_by_name(self, task_name):
        self.current_task_name = task_name
        log(f"load_behavior_task: {task_name}")

        config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
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


        # Load the environment
        if self.og_sim is None: 
            self.og_sim = og.Environment(configs=config)
        else:
            og.clear()
            og.sim.stop()
            self.og_sim.reload(configs=config)
            og.sim.play()
            self.og_sim.post_play_load()

        # self.reset()

        self.scene = self.og_sim.scene
        self.robot = self.og_sim.robots[0]

        # Allow user to move camera more easily
        # og.sim.enable_viewer_camera_teleoperation()

        # self.action_primitives = StarterSemanticActionPrimitives(self.og_sim, enable_head_tracking=False)
        og.sim.enable_viewer_camera_teleoperation()
        self.set_camera_lookat_robot()

    def reset(self):
        self.og_sim.reset()

    def step(self):
        if self.control_queue.empty():
            pass
            # self.og_sim.step(self.action_primitives._empty_action())
            # log('robot idle !!!')
        else:
            action = self.control_queue.get()
            if action is not None:
                self.og_sim.step(action)
            else:
                pass
                # self.og_sim.step(self.action_primitives._empty_action())
                # log('robot step !!!')

    def set_viewer_camera_pose(self, position, orientation):
        og.sim.viewer_camera.set_position_orientation(position=position, orientation=orientation)

    def set_camera_lookat_robot(self):
        #set camera position and orientation
        robot_pos, robot_quat = self.robot.get_position_orientation()
        # calculate robot forward direction vector
        forward_dir = T.quat2mat(robot_quat)[:3, 0]  # take first column of rotation matrix as forward direction vector
        # calculate forward 1 meter position
        camera_pos = robot_pos + forward_dir * 1.0
        camera_pos[2] = 2

        # calculate angle between robot forward direction vector and y-axis
        y_axis = th.tensor([0.0, 1.0, 0.0])
        angle = th.acos(th.dot(-forward_dir, y_axis) / (th.norm(-forward_dir) * th.norm(y_axis)))
        angle = th.sign(forward_dir[0])*angle

        camera_quat = T.euler2quat(th.tensor([0.45,0,angle]))
        og.sim.viewer_camera.set_position_orientation(camera_pos, camera_quat)
        self.camera_pos = camera_pos
        self.camera_quat = camera_quat


            
    def add_control(self,control):
        self.control_queue.put(control)

    def get_scene_name(self):
        return self.scene.scene_model
    
    def get_robot_pos(self):
        return self.robot.get_position_orientation()[0]

    def get_trav_map(self):
        return self.scene._trav_map

    def do_task(self):
        log("start do_task")
        controller = StarterSemanticActionPrimitives(self.og_sim, enable_head_tracking=False)

        # Grasp of cologne
        grasp_obj = self.scene.object_registry("name", "cologne")
        print("Executing controller")

        primitive_action = controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj,attempts=10)


        self.add_control(primitive_action)
        # execute_controller(primitive_action, self.og_sim)
        # print("Finished executing grasp")

        # Place cologne on another table
        print("Executing controller")
        table = self.scene.object_registry("name", "table")
        primitive_action = controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table,attempts=10)
        self.add_control(primitive_action)
        # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table), self.og_sim)
        # print("Finished executing place")

