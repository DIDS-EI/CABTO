from dexrl.sim.scenarios.base_scenario import *
from omni.isaac.core.objects import  VisualCuboid
from omni.kit.viewport.utility import get_active_viewport_and_window
import numpy as np
from dexrl.sim import utils
from dexrl import global_config
import carb
from omni.isaac.sensor import Camera
from omni.ui import DockPosition, Workspace
from omni.kit.viewport.utility import create_viewport_window
from omni.isaac.core.simulation_context import SimulationContext
from .._cfg import ExtSimCfg
from dexrl.utils import format_array
from dexrl.sim_extension.ext_scenarios.ext_scenario import ExtScenario



class RightArmHandIKExtScenario(ExtScenario):
    """右臂手跟随立方体
    """
    scenario_cls = Scenario
    viewport_camera_pos_lookat = -0.14,-2.78,1.69,-0.17,-1.80,1.50


    def load_ext_objects(self):

        self.hand_link_base_cube = VisualCuboid(
            name="hand_link_base_cube",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((-0.306,-0.266,1.06)),
            # orientation = utils.rot.euler_angles_to_quat(np.deg2rad([0,0,-30])),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/hand_link_base_cube",
            size=0.05,
            color=np.array([1, 0, 1]), 
        )

        self.pos_cube = VisualCuboid(
            name="pos_cube",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((-0.306,-0.266,1.06)),
            # orientation = utils.rot.euler_angles_to_quat(np.deg2rad([0,0,-30])),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/pos_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )

        self.rot_cube = VisualCuboid(
            name="rot_cube",
            position=np.array((-0.1,-0.40,1.05)),
            # orientation=euler_angles_to_quats(np.array([-140, 8, 14])),
            prim_path="/World/rot_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.euler_y_cube = VisualCuboid(
            name="euler_y_cube",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((0.0,-0.40,1.05)),
            # orientation = utils.rot.euler_angles_to_quat([0,0.6,0]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/euler_y_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.euler_z_cube = VisualCuboid(
            name="euler_z_cube",
            position=np.array((0.1, -0.4, 1.05)),
            orientation=utils.rot.euler_angles_to_quat([0,0.,-0.6]),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/euler_z_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.lookat_cube = VisualCuboid(
            name="lookat_cube",
            position=np.array((0.1, -0.4, 1.05)),
            # orientation=utils.rot.euler_angles_to_quat([0,0.,-0.6]),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/lookat_cube",
            size=0.05,
            color=np.array([0, 1, 0]),
        )

        # self.cube_list:List[VisualCuboid] = [self.cube_left, self.pos_cube]
        # self.cube_use_prim_list:List[pxr.Usd.Prim] = [
        #     get_prim_at_path(self.cube_left.prim_path),
        #     get_prim_at_path(self.pos_cube.prim_path)]
       
    def script(self):

        while True:
            current_pose = self.robot.forward_kinematics(self.robot.right_arm_hand.arm.get_joint_positions())
            current_pos = current_pose[:3] # 位置
            current_quat = current_pose[3:7] # 旋转
            mixed_pose = self.robot.get_mixed_pose()
            rot_matrix = utils.rot.euler_to_rot_matrix(mixed_pose[3:6])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))
            # lookat = lookat_direction*0.2+mixed_pose[:3]
            pos_target,_ = self.pos_cube.get_world_pose()
            _,quat_target = self.rot_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            rot = euler[0]
            _,quat_target = self.euler_y_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            euler_y = euler[1] + np.pi/2
            _,quat_target = self.euler_z_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            euler_z = euler[2]

            yz_rot_matrix = utils.rot.euler_to_rot_matrix([0,euler_y,euler_z])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(yz_rot_matrix,forward_direction=np.array([0,0,1]))

            x_rot_matrix = utils.rot.axis_theta_to_rot_matrix(lookat_direction,rot)
            final_rot_matrix = x_rot_matrix @ yz_rot_matrix
            final_quat = utils.rot.rot_matrix_to_quat(final_rot_matrix)

            print(final_quat)

            # print(utils.rot.rotate_quat(current_quat,lookat_direction,rot))
            # final_rot_matrix = rot_matrix @  rot
            # print(utils.rot.rot_matrix_to_quat(final_rot_matrix))
            # x_lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))



            # quat = utils.rot.euler_angles_to_quat([0, euler_y, euler_z])
            # rotation = R.from_quat(quat)
            # axis = rotation.as_rotvec() / np.linalg.norm(rotation.as_rotvec())
            # new_quat = rotate_quat_by_angle(quat, axis, -rot)


            # print(new_quat)
            # self.lookat_cube.set_world_pose(position=x_lookat_direction*0.2+mixed_pose[:3])

            # print(self.cube_use_prim_list[i].GetAttribute("xformOp:rotateXZY").Get())
            # pos_target[2] -= 
            # euler_target = utils.rot.quat_to_euler_angles(quat_target)
            # euler_target[1] += np.pi/2
            # euler_target[0] -= 1

            # # 通过指向物体计算欧拉角
            self.scenario.robot.right_arm_hand.arm.set_ik_target_real(np.concatenate([pos_target, [rot,euler_y,euler_z]]))
            current_pos, status = self.scenario.robot.right_arm_hand.arm.reach_joint_target()
            # if status != JointControlStatus.REACHED_TARGET:
            #     print(f"current_pos: {current_pos}, status: {status}")
            
            
            hand_link_base_cube_pos = self.scenario.robot.right_arm_hand.arm.xform_view.get_world_poses()[0]
            self.hand_link_base_cube.set_world_pose(position=hand_link_base_cube_pos)
            yield ()



class LeftArmHandIKExtScenario(ExtScenario):
    """左臂手跟随立方体
    """
    scenario_cls = Scenario
    viewport_camera_pos_lookat = -0.14,-2.78,1.69,-0.17,-1.80,1.50


    def load_ext_objects(self):

        self.hand_link_left_base_cube = VisualCuboid(
            name="hand_link_left_base_cube",
            position=np.array((0.306,-0.30586,1.05517)),
            prim_path="/World/hand_link_left_base_cube",
            size=0.05,
            color=np.array([0, 1, 1]), 
        )
        self.hand_link_right_base_cube = VisualCuboid(
            name="hand_link_right_base_cube",
            position=np.array((-0.1, -0.4, 1.05)),
            prim_path="/World/hand_link_right_base_cube",
            size=0.05,
            color=np.array([1, 1, 1]), 
        )


        self.pos_cube = VisualCuboid(
            name="pos_cube",
            position=np.array((0.306,-0.30586,1.05517)),
            prim_path="/World/pos_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )

        self.rot_cube = VisualCuboid(
            name="rot_cube",
            position=np.array((-0.1,-0.40,1.05)),
            # orientation=euler_angles_to_quats(np.array([-140, 8, 14])),
            prim_path="/World/rot_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.euler_y_cube = VisualCuboid(
            name="euler_y_cube",
            # position=np.array((-0.1, -0.4, 1.05)),
            position=np.array((0.0,-0.40,1.05)),
            # orientation = utils.rot.euler_angles_to_quat([0,0.6,0]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/euler_y_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.euler_z_cube = VisualCuboid(
            name="euler_z_cube",
            position=np.array((0.1, -0.4, 1.05)),
            orientation=utils.rot.euler_angles_to_quat([0,0.,-0.6]),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/euler_z_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

        self.lookat_cube = VisualCuboid(
            name="lookat_cube",
            position=np.array((0.1, -0.4, 1.05)),
            # orientation=utils.rot.euler_angles_to_quat([0,0.,-0.6]),
            # orientation = np.array([-0.14581,-0.14581,-0.03285,-0.42832]),#np.array([-16.777,3.805,3.805]), #np.array([-0.14581,-0.14581,-0.03285,-0.42832]),
            # 1.39,-0.87 -0.24
            prim_path="/World/lookat_cube",
            size=0.05,
            color=np.array([0, 1, 0]),
        )


    def script(self):

        while True:
            current_pose = self.robot.forward_kinematics(self.robot.left_arm_hand.arm.get_joint_positions())
            current_pos = current_pose[:3] # 位置
            current_quat = current_pose[3:7] # 旋转
            mixed_pose = self.robot.get_mixed_pose()
            rot_matrix = utils.rot.euler_to_rot_matrix(mixed_pose[3:6])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))
            # lookat = lookat_direction*0.2+mixed_pose[:3]
            pos_target,_ = self.pos_cube.get_world_pose()
            _,quat_target = self.rot_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            rot = euler[0]
            _,quat_target = self.euler_y_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            euler_y = euler[1] + np.pi/2
            _,quat_target = self.euler_z_cube.get_world_pose()
            euler = utils.rot.quat_to_euler_angles(quat_target)
            euler_z = euler[2]

            yz_rot_matrix = utils.rot.euler_to_rot_matrix([0,euler_y,euler_z])
            lookat_direction = utils.rot.rot_matrix_to_lookat_direction(yz_rot_matrix,forward_direction=np.array([0,0,1]))

            x_rot_matrix = utils.rot.axis_theta_to_rot_matrix(lookat_direction,rot)
            final_rot_matrix = x_rot_matrix @ yz_rot_matrix
            final_quat = utils.rot.rot_matrix_to_quat(final_rot_matrix)

            print(final_quat)


            # # 通过指向物体计算欧拉角
            self.scenario.robot.left_arm_hand.arm.set_ik_target_real(np.concatenate([pos_target, [rot,euler_y,euler_z]]))
            current_pos, status = self.scenario.robot.left_arm_hand.arm.reach_joint_target()
            # if status != JointControlStatus.REACHED_TARGET:
            #     print(f"current_pos: {current_pos}, status: {status}")
            
            
            hand_link_left_base_cube_pos = self.scenario.robot.left_arm_hand.arm.xform_view.get_world_poses()[0]
            self.hand_link_left_base_cube.set_world_pose(position=hand_link_left_base_cube_pos)
            
            hand_link_right_base_cube_pos = self.scenario.robot.right_arm_hand.arm.xform_view.get_world_poses()[0]
            self.hand_link_right_base_cube.set_world_pose(position=hand_link_right_base_cube_pos)
            
            yield ()


