# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import numpy as np

import omni.timeline
import omni.ui as ui
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import create_new_stage, get_current_stage,close_stage,clear_stage
from omni.isaac.core.world import World

from omni.isaac.ui.element_wrappers import CollapsableFrame, StateButton, StringField, Button
from omni.isaac.ui.element_wrappers.core_connectors import LoadButton, ResetButton
from omni.isaac.ui.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux
from omni.isaac.core.utils.rotations import quat_to_euler_angles,\
quat_to_rot_matrix, euler_angles_to_quat

import omni.isaac.core.utils.rotations as rot_utils
from dexrl.sim.utils import calculate_lookat_position, format_array, euler_yz_angles_to_quats
from dexrl.sim.utils.pxr_utils.gf import quat_gf_to_numpy
from dexrl.sim import utils
from pxr import Gf
from typing import Dict
from .ext_scenarios import *
# from .ext_scenarios_dual_arm import *
from ._cfg import ExtSimCfg

class UIBuilder:
    def __init__(self,cfg:ExtSimCfg):
        self.cfg = cfg
        self._is_menu_open = False

        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()


    # @property
    # def scenario(self):
    #     return self._scenario.scenario
    
    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        self._is_menu_open = True

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()
        
        # self._is_menu_open = False

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=True)
        self.arm_ui_dict:Dict[str, Dict[str, ui.StringField]] = {"Left":{},"Right":{}}
        self.hand_ui_dict:Dict[str, Dict[str, ui.StringField]] = {"Left":{},"Right":{}}
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                dt = 1/60
                # self._load_btn.set_world_settings(physics_dt=dt, rendering_dt=dt)
                self._load_btn.set_world_settings(physics_dt=dt, rendering_dt=dt,backend="numpy",device=self.cfg.device)
                # self._load_btn.set_world_settings(physics_dt=dt, rendering_dt=dt,backend="torch",device="cuda:0")
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        scenario_info_frame = CollapsableFrame("Scenario Info", collapsed=True)

        with scenario_info_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._viewport_pos_lookat_label = StringField("Viewport Position", default_value="")
                self.wrapped_ui_elements.append(self._viewport_pos_lookat_label)


        self.hand_btn_func_dict = {"Left":self._on_set_left_hand_control_btn,"Right":self._on_set_right_hand_control_btn}

        for arm_name, arm_ui_dict_current in self.arm_ui_dict.items():
            arm_ik_info_frame = CollapsableFrame(f"{arm_name} Arm IK Info", collapsed=False if arm_name == "Right" else True)
            arm_ui_dict_current["arm_ik_info_frame"] = arm_ik_info_frame

            with arm_ik_info_frame:
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    arm_ui_dict_current['eef_target_position_label'] = StringField(f"{arm_name} EEF Target Position", default_value="")
                    self.wrapped_ui_elements.append(arm_ui_dict_current['eef_target_position_label'])

                    arm_ui_dict_current['eef_target_mixed_euler_label'] = StringField(f"{arm_name} EEF Target Mixed Euler", default_value="")
                    self.wrapped_ui_elements.append(arm_ui_dict_current['eef_target_mixed_euler_label'])


                    arm_ui_dict_current['eef_position_label'] = StringField("EEF Position", default_value="")
                    self.wrapped_ui_elements.append(arm_ui_dict_current['eef_position_label'])

                    arm_ui_dict_current['eef_mixed_euler_label'] = StringField(f"{arm_name} EEF Mixed Euler", default_value="")
                    self.wrapped_ui_elements.append(arm_ui_dict_current['eef_mixed_euler_label'])

                    arm_ui_dict_current['eef_quat_label'] = StringField(f"{arm_name} EEF Quat", default_value="")
                    self.wrapped_ui_elements.append(arm_ui_dict_current['eef_quat_label'])

                    arm_ui_dict_current['eef_lookat_direction_label'] = StringField(f"{arm_name} EEF Lookat Direction", default_value="")
                    self.wrapped_ui_elements.append(arm_ui_dict_current['eef_lookat_direction_label'])


            hand_ui_dict_current = self.hand_ui_dict[arm_name]

            hand_ik_info_frame = CollapsableFrame(f"{arm_name} Hand IK Info", collapsed=False if arm_name == "Right" else True)
            hand_ui_dict_current["hand_ik_info_frame"] = hand_ik_info_frame

            with hand_ui_dict_current["hand_ik_info_frame"]:
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    hand_ui_dict_current['hand_joint_list'] = []
                    for i in range(6):
                        hand_ui_dict_current['hand_joint_list'].append(StringField(f"Hand Joint {i}", default_value=""))
                        self.wrapped_ui_elements.append(hand_ui_dict_current['hand_joint_list'][-1])

                    hand_ui_dict_current['hand_control_list'] = []
                    for i in range(3):
                        hand_ui_dict_current['hand_control_list'].append(StringField(f"Hand Control {i}", default_value="1"))
                        self.wrapped_ui_elements.append(hand_ui_dict_current['hand_control_list'][-1])

                    hand_ui_dict_current['hand_control_btn'] = Button(
                        "Set Hand Control",
                        "SET",
                        on_click_fn=self.hand_btn_func_dict[arm_name],
                    )
                    self.wrapped_ui_elements.append(hand_ui_dict_current['hand_control_btn'])


        robot_info_frame = CollapsableFrame("Robot Info", collapsed=True)

        with robot_info_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._robot_joints_position_label = StringField("Robot Joints Position", default_value="")
                self.wrapped_ui_elements.append(self._robot_joints_position_label)

                self._set_ik_pos_label = StringField("Set Left Arm IK Position", default_value="")
                self.wrapped_ui_elements.append(self._set_ik_pos_label)

                self._set_ik_pos_btn = Button(
                    "Set Left Arm IK Position",
                    "SET",
                    on_click_fn=self._on_set_ik_pos_btn,
                )



    def _on_set_ik_pos_btn(self):
        try:
            ik_pos_str = self._set_ik_pos_label.get_value()
            if not ik_pos_str.strip():  # Check if string is empty or only whitespace
                print("Error: Position input is empty")
                return
                
            # Split by comma and filter out empty strings
            ik_pos_values = [x.strip() for x in ik_pos_str.split(",") if x.strip()]
            
            # Convert to float and handle invalid inputs
            ik_pos = []
            for val in ik_pos_values:
                try:
                    ik_pos.append(float(val))
                except ValueError:
                    print(f"Error: Invalid number format in position: {val}")
                    return
            
            # Pad with zeros if less than 3 values
            if len(ik_pos) < 3:
                ik_pos.extend([0.0] * (3 - len(ik_pos)))
            elif len(ik_pos) > 3:
                print("Warning: More than 3 values provided, using first 3 values")
                ik_pos = ik_pos[:3]
                
            print(f"Setting position to: {ik_pos}")
            self.ext_scenario.red_cube_right.set_world_pose(position=np.array(ik_pos))
            
        except Exception as e:
            print(f"Error setting IK position: {str(e)}")

    def _on_set_right_hand_control_btn(self):
        side = "Right"
        joint_target_str = [self.hand_ui_dict[side]['hand_control_list'][i].get_value() for i in range(3)]
        joint_target_values = [float(x) for x in joint_target_str]
        self.ext_scenario.set_hand_control_target(side,joint_target_values)

    def _on_set_left_hand_control_btn(self):
        side = "Left"
        joint_target_str = [self.hand_ui_dict['Left']['hand_control_list'][i].get_value() for i in range(3)]
        joint_target_values = [float(x) for x in joint_target_str]
        self.ext_scenario.set_hand_control_target(side,joint_target_values)

    # def _on_set_ik_eef_quat_btn(self):
    #     try:
    #         ik_eef_quat_str = self._set_ik_eef_quat_label.get_value()
    #         if not ik_eef_quat_str.strip():  # Check if string is empty or only whitespace
    #             print("Error: Position input is empty")
    #             return
                
    #         # Split by comma and filter out empty strings
    #         ik_eef_quat_values = [x.strip() for x in ik_eef_quat_str.split(",") if x.strip()]
            
    #         # Convert to float and handle invalid inputs
    #         ik_eef_quat = []
    #         for val in ik_eef_quat_values:
    #             try:
    #                 ik_eef_quat.append(float(val))
    #             except ValueError:
    #                 print(f"Error: Invalid number format in position: {val}")
    #                 return
            
    #         # Pad with zeros if less than 3 values
    #         if len(ik_eef_quat) < 3:
    #             ik_eef_quat.extend([0.0] * (3 - len(ik_eef_quat)))
    #         elif len(ik_eef_quat) > 3:
    #             print("Warning: More than 4 values provided, using first 4 values")
    #             ik_eef_quat = ik_eef_quat[:3]
                
    #         print(f"Setting position to: {ik_eef_quat}")
    #         self._scenario.blue_cube_right.set_world_pose(position=np.array(ik_eef_quat))
            
    #     except Exception as e:
    #         print(f"Error setting IK position: {str(e)}")
    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    def _on_init(self):
        self._articulation = None
        self._cuboid = None


        ext_scenario_cfg = ExtSimCfg(
            device=self.cfg.device,
        )
        self.ext_scenario:ExtDualArmScenario = globals()[self.cfg.ext_scenario_name](ext_scenario_cfg)


    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight:UsdLux.SphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(60000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([0, -6.5, 12])

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        self._add_light_to_stage()

        world:World = World.instance()
        self.ext_scenario.set_world(world)

        self.ext_scenario.load_assets()


    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self.ext_scenario.setup()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._scenario_state_btn.trigger_click_if_a_state()
        self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self.ext_scenario.reset()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True



    def _update_scenario(self,action):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        This function will repeatedly advance the script in scenario.py until it is finished.

        Args:
            step (float): The dt of the current physics step
        """
        done = self.ext_scenario.ext_update()

        if done:
            self._scenario_state_btn.enabled = False

        if not self._is_menu_open:
            return

        position = utils.usd.get_prim_position(self.ext_scenario.viewport_camera_prim)
        euler = utils.usd.get_prim_euler(self.ext_scenario.viewport_camera_prim)
        rot_matrix = utils.rot.euler_to_rot_matrix(euler)
        lookat = utils.rot.rot_matrix_to_lookat(position, rot_matrix)

        pos_lookat = np.concatenate([position, lookat])

        self._viewport_pos_lookat_label.set_value(format_array(pos_lookat))


        # left arm target
        for arm_name, arm_ui_dict_current in self.arm_ui_dict.items():
            arm_hand = self.ext_scenario.robot.arm_hand_name_dict[arm_name]
            arm = arm_hand.arm
            target_pos = arm.target_eef_pos
            arm_ui_dict_current["eef_target_position_label"].set_value(format_array(target_pos))
            target_euler_yz = arm.target_eef_euler_yz
            arm_ui_dict_current["eef_target_mixed_euler_label"].set_value(format_array([arm.target_eef_rotation,*target_euler_yz]))

            current_joint = self.ext_scenario.robot.arm_name_dict[arm_name].get_joint_positions()
            current_pose = self.ext_scenario.robot.arm_name_dict[arm_name].forward_kinematics(current_joint)
            pos = current_pose[:3] # 位置
            quat = current_pose[3:7] # 旋转
            mixed_pose = self.ext_scenario.robot.arm_name_dict[arm_name].get_mixed_pose()
            rot_matrix = utils.rot.euler_to_rot_matrix(mixed_pose[3:6])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))

            arm_ui_dict_current["eef_position_label"].set_value(format_array(mixed_pose[:3]))
            arm_ui_dict_current["eef_mixed_euler_label"].set_value(format_array(mixed_pose[3:6]))
            arm_ui_dict_current["eef_quat_label"].set_value(format_array(quat))
            # self.eef_target_rotation_label.set_value(format_array(current_joint[6].item()))
            arm_ui_dict_current["eef_lookat_direction_label"].set_value(format_array(lookat_direction))

            hand = arm_hand.hand
            hand_ui_dict_current = self.hand_ui_dict[arm_name]
            real_hand_joint_6 = hand.get_hand_joint_real_6()
            for i in range(6):
                hand_ui_dict_current["hand_joint_list"][i].set_value(format_array(real_hand_joint_6[i]))

        self._robot_joints_position_label.set_value(format_array(self.ext_scenario.robot.articulation.get_joint_positions()))


    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the aa is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False

    def close(self):
        self.ext_scenario.close()
