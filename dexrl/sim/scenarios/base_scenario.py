import numpy as np

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.world import World

from typing import List
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation import ArticulationKinematicsSolver,LulaKinematicsSolver
from omni.isaac.core.prims import XFormPrimView,XFormPrim
import os
from pxr import Usd, Sdf
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats

from dexrl.sim.utils import Normalizer_N1_1, Normalizer_0_1
import torch

from dexrl import global_config
from typing import Dict
from dexrl.sim.scenarios._cfg import ScenarioCfg
from omni.isaac.sensor import Camera
from external.Robotic_Arm.rm_robot_interface import rm_robot_arm_model_e,rm_force_type_e,\
    Algo,rm_frame_t,rm_inverse_kinematics_params_t
import carb
from dexrl.sim import utils
import time
from dexrl.utils.logger import print_red
from omni.kit.viewport.utility import create_viewport_window


"""
机器人的动作空间分为三类：
1. abs，即仿真器中的绝对关节角，受到机器人本身的硬件限制
2. real，即真机中的输入，也是人容易理解和设置的空间。受到任务场景设定的限制。一般 hand 是 [0,1]，arm_ik 视具体而定。
3. policy, 即强化学习网络的输出，一般为 [-1,1]。
ps: hand 的 policy 和 real 还分为 6 和 11 个关节，但hand的 abs 固定为11个关节。
"""

def print_red(text):
    print(f"\033[91m{text}\033[0m")

class JointControlStatus:
    RUNNING = 0
    REACHED_TARGET = 1
    STUCK = 2
    TIMEOUT = 3
    OUT_OF_LIMIT = 4
class JointController:
    def __init__(self, owner):
        self.cfg:ScenarioCfg = owner.cfg
        self.articulation:Articulation = owner.articulation
        self.joint_indices:np.ndarray = owner.joint_indices
        self.joint_lower_limits:np.ndarray = owner.joint_lower_limits
        self.joint_upper_limits:np.ndarray = owner.joint_upper_limits
        self.joint_normalizer = Normalizer_0_1(self.joint_lower_limits, self.joint_upper_limits)


        self.joint_target = None
        self.current_joint_pos = None
        self.current_joint_pos_normalized = None
        self.articulation_action = None

        self.last_joint_pos_normalized = None
        self.stuck_steps = 0
        self.action_steps = 0
        


    def reach_joint_target(self):
        self.current_joint_pos = self.articulation.get_joint_positions(joint_indices=self.joint_indices)
        self.current_joint_pos_normalized = self.joint_normalizer.normalize(self.current_joint_pos)

        if self.joint_target is None: return self.current_joint_pos, JointControlStatus.REACHED_TARGET
        
        # 是否达到目标
        if self.is_reached_target():
            self.clear_target()
            # print_red("action reached target")
            return self.current_joint_pos, JointControlStatus.REACHED_TARGET

        # 是否长时间不动
        if self.last_joint_pos_normalized is not None:
            joint_pos_error = np.abs(self.current_joint_pos_normalized - self.last_joint_pos_normalized)

            if np.all(joint_pos_error < 1e-2):
                self.stuck_steps += 1
            
            if self.stuck_steps > 5:
                self.clear_target()
                # print_red("action stuck")
                return self.current_joint_pos, JointControlStatus.STUCK


        self.last_joint_pos_normalized = self.current_joint_pos_normalized

        if self.articulation_action is None:  
            self.articulation_action = ArticulationAction(
                joint_positions=self.joint_target,
                joint_indices=self.joint_indices)

        if self.action_steps > 20:
            self.clear_target()
            # print_red("action timeout")
            return self.current_joint_pos, JointControlStatus.TIMEOUT

        self.articulation.apply_action(self.articulation_action)
        self.action_steps += 1

        return self.current_joint_pos, JointControlStatus.RUNNING

    def set_joint_target(self,joint_target):
        self.clear_target()
        self.joint_target = joint_target
        self.joint_target_normalized = self.joint_normalizer.normalize(self.joint_target)


    def is_reached_target(self):
        joint_pos_error = np.abs(self.current_joint_pos_normalized - self.joint_target_normalized)
        # return np.all(joint_pos_error < 1e-3)
        # print(f"joint_pos_error: {joint_pos_error}")
        return np.all(joint_pos_error < 1e-2)

    def clear_target(self):
        self.joint_target = None
        self.articulation_action = None
        self.last_joint_pos_normalized = None
        self.stuck_steps = 0
        self.action_steps = 0




class Arm:
    name = None
    config_folder = os.path.join(global_config.assets_path, "lula_config/")
    end_effector_frame_name = None
    urdf_path = None
    robot_description_path = None
    joint_lower_limits = np.array([-3.1067, -2.2690, -3.1067, -2.3562, -3.1067, -2.2340, -6.2800])
    joint_upper_limits = np.array([3.1067, 2.2690, 3.1067, 2.3562, 3.1067, 2.2340, 6.2800])

    ik_lower_limit = np.array([0.,0,0,0,0,0])
    ik_upper_limit = np.array([0.,0,0,0,0,0])
    eef_euler_yz_scale = np.array([1.,1.])
    eef_euler_yz_bias = np.array([0.,0])
    eef_rotation_scale = 1.0
    eef_rotation_bias = 0 #1.0903504
    # joint_indices = np.array([1,3,5,7,9,11,13])

    arm_base_height = 1.05
    
    prim_paths_expr = None
    
    eef_x_offset = 0.14353-0.08277-0.00062
    eef_y_offset = -0.02855-0.0057
    eef_z_offset = 0.00408+0.0021+0.00047
    
    right_pose_forward_direction = np.array([0,0,1])
    
    def __init__(self,cfg:ScenarioCfg,robot_prim_path: str,articulation: Articulation):
        self.cfg = cfg
        self.robot_prim_path = robot_prim_path
        self.articulation = articulation
        # self.joint_target = cfg.default_joint_pos[self.joint_indices]
        self.joint_target = None
        
        self.target_eef_pos = np.array([0.,0,0])
        self.target_eef_euler_yz = np.array([0.,0])
        self.target_eef_rotation = 0.

        self.articulation_motion_policy: ArticulationMotionPolicy = None

        self.ik_normalizer = Normalizer_N1_1(self.ik_lower_limit, self.ik_upper_limit)
        self.articulation_action = None

        self.joint_controller = JointController(self)

        self.defalut_last_joint = np.array([0.72972894, -1.2601283, -3.1058693, 1.1898057, -2.5298553, -1.5799452, -1.1200296])
        self.defalut_last_joint_deg = np.deg2rad(self.defalut_last_joint)

        if self.prim_paths_expr is not None:
            rim_paths_expr = list(map(lambda x: robot_prim_path + x, self.prim_paths_expr))
            self.xform_view = XFormPrimView(prim_paths_expr=rim_paths_expr,name="xform_prim_view")
        else:
            self.xform_view = None


    def setup(self):
        arm_model = rm_robot_arm_model_e.RM_MODEL_RM_75_E # RM_75机械臂
        # arm_model = 1
        force_type = rm_force_type_e.RM_MODEL_RM_B_E + 2   # 标准版
        self.algo_handle = Algo(arm_model, force_type)
        self.algo_handle.rm_algo_set_angle(0.0,-30.0,0.0)
        frame = rm_frame_t("", [0.0, 0.0, 0, 0, 0, 0.], 0, 0.0, 0.0, 0.0)
        self.algo_handle.rm_algo_set_toolframe(frame)


        # self.lula_kinematics_solver = LulaKinematicsSolver(self.robot_description_path, self.urdf_path)
        # self.articulation_kinematics_solver = ArticulationKinematicsSolver(self.articulation, self.lula_kinematics_solver, self.end_effector_frame_name)


    # def inverse_kinematics(self,):


    def get_ik_target_real(self, ik_target,abs_rot=True):
        '''
        求出 相对于臂 的 目标位置
        ik 的 target 的 real
        传入的是世界坐标
        返回 out_joint
        '''
        self.target_eef_pos = np.array([ik_target[0],ik_target[1], ik_target[2]])
        self.target_eef_euler_yz = np.array([ik_target[4],ik_target[5]])
        self.target_eef_rotation = ik_target[3]
        
        target_pose = np.concatenate([self.target_eef_pos, [0.],self.target_eef_euler_yz])
        target_pose = self.convert_ik_target(target_pose) # set 转到臂 的坐标

        target_pose[2] -= self.arm_base_height
        last_joint = self.articulation.get_joint_positions(joint_indices=self.joint_indices)
        ret,out_joint = self.inverse_kinematics(last_joint,target_pose,flag=1)

        if ret == 0:
            out_joint[6] += self.eef_rotation_scale*ik_target[3] + self.eef_rotation_bias # 改过以后
            self.joint_target = out_joint
            return True,out_joint
        else:
            print("IK did not converge to a solution.  No action is being taken")
            carb.log_error(f"IK ERROR: {ret}, output: {out_joint}")
            return False,None
        
        return False,None


    # 设置 ik 目标
    def set_ik_target_real(self, ik_target):
        success,joint_target = self.get_ik_target_real(ik_target)
        if not success:
            return False,None
        
        self.joint_controller.set_joint_target(joint_target)
        return True,joint_target


    # 转换相对于 arm-base 的 ik 目标
    def convert_ik_target(self,ik_target):
        return ik_target

    # 正向运动学 get_mixed_pose_joint
    def forward_kinematics(self,joint,flag=0):
        pose = self.algo_handle.rm_algo_forward_kinematics(np.rad2deg(joint),flag)
        
        pose = self.convert_forward_kinematics(pose)
        return pose

    # 转换相对于 arm-base 的正向运动学
    def convert_forward_kinematics(self,pose):
        return pose

    # 逆向运动学
    def inverse_kinematics(self,last_joint,pose,flag=0):
        # last_joint = self.defalut_last_joint
        # params = rm_inverse_kinematics_params_t(self.defalut_last_joint_deg, pose, flag)
        params = rm_inverse_kinematics_params_t(np.rad2deg(last_joint), pose, flag)
        ret,out_joint = self.algo_handle.rm_algo_inverse_kinematics(params)
        out_joint = np.deg2rad(out_joint)
        return ret, out_joint

    # 到达关节目标
    def reach_joint_target(self):
        return self.joint_controller.reach_joint_target()

    # 转换 eef 欧拉角为四元数
    def eef_euler_yz_to_quats(self,euler_yz):
        euler_yz = euler_yz * self.eef_euler_yz_scale + self.eef_euler_yz_bias
        quat = euler_angles_to_quats((0,euler_yz[0], euler_yz[1]))
        return quat

    # 获取关节位置
    def get_joint_positions(self):
        # print(f"=======self.joint_indices: {self.joint_indices}==========")
        return self.articulation.get_joint_positions(joint_indices=self.joint_indices)


    def get_mixed_pose_joint(self,current_joint):
        current_pose = self.forward_kinematics(current_joint)
        # print(f"current_pose: {current_pose}","")
        pos = current_pose[:3] # 位置
        quat = current_pose[3:7] # 旋转
        
        rot_matrix = utils.rot.quat_to_rot_matrix(quat) # 旋转矩阵
        lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=self.right_pose_forward_direction) # 朝向
        # 方向向量
        y_euler = np.arcsin(-lookat_direction[2]) + np.pi/2
        z_euler = np.arctan2(lookat_direction[1], lookat_direction[0])

        origin_rot_matrix = utils.rot.euler_to_rot_matrix([0,y_euler,z_euler])
        R = rot_matrix @ origin_rot_matrix.T

        x_lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([1,0,0])) # 朝向
        x_lookat_direction_after = utils.rot.rot_matrix_to_lookat_direction(R,forward_direction=x_lookat_direction)#超右的向量转一遍

        cos_theta = np.dot(x_lookat_direction,x_lookat_direction_after) # 求夹角
        x_euler = np.arccos(cos_theta)  #角度值
        cross_product = np.cross(x_lookat_direction,x_lookat_direction_after) #叉乘  法向量
        sin_theta = np.dot(cross_product, lookat_direction)     # 确定符号叉乘和朝向的点乘

            
        if sin_theta <0: #角度始终为正
            
            x_euler = -x_euler

        pos[2] += self.arm_base_height # pos
        return np.array([*pos,x_euler,y_euler,z_euler]) # 读的是绝对的


    def get_mixed_pose(self):
        current_joint = self.get_joint_positions()
        mixed_pose = self.get_mixed_pose_joint(current_joint)
        return mixed_pose



class LeftArm(Arm):
    name = "left_arm"
    end_effector_frame_name = "arm1_link7"
    urdf_path = os.path.join(Arm.config_folder + "PsiRobot_DC_01.urdf")
    robot_description_path = os.path.join(Arm.config_folder + "PsiRobot_DC_01_left_arm_descriptor.yaml")
    eef_euler_yz_scale = np.array([1.,1.])
    eef_euler_yz_bias = np.array([np.pi/2,0])
    # eef_rotation_scale = -1.0
    eef_rotation_bias = 0 #2 #2.2
    
    joint_indices = np.array([0, 2, 4, 6, 8, 10, 12])

    ik_lower_limit = np.array([0.45,-0.45, 0.9,  -999,  1/3 * np.pi,    -1/3 * np.pi])
    ik_upper_limit = np.array([0.25,-0.25, 1.1,   999,     2/3 * np.pi,    1/6 * np.pi])
    
    eef_x_offset = 0.09680658 #0.14353-0.08277-0.00062+0.03666658
    eef_y_offset = 0.00344287 #-0.02855-0.0057+0.03769287
    eef_z_offset = 0.02804688 #0.00408+0.0021+0.00047+0.02139688

    # cys
    # eef_x_offset = 0.09680658 - 0.015  #0.14353-0.08277-0.00062+0.03666658
    # eef_y_offset = 0.00344287 + 0.02 #-0.02855-0.0057+0.03769287
    # eef_z_offset = 0.00408+0.0021+0.00047+0.02139688
    
    prim_paths_expr = [
        "/hand1_link_base",
    ]


    def convert_ik_target(self,ik_target):
        ik_target[0] -= self.eef_x_offset 
        ik_target[1] += self.eef_y_offset
        ik_target[2] += self.eef_z_offset
        # ik_target[3] += np.pi/4*3 
        # ik_target[4] = -ik_target[4] 
        # ik_target[5] = -ik_target[5] # Z

        ik_target[0],ik_target[1] = -ik_target[0],-ik_target[1]
        return ik_target

    def convert_forward_kinematics(self,pose):
        # print("pose:",pose)
        pose[0],pose[1] = -pose[0],-pose[1]
        pose[0] += self.eef_x_offset
        pose[1] -= self.eef_y_offset
        pose[2] -= self.eef_z_offset
        # print("next pose:",pose)
        return pose

class RightArm(Arm):
    name = "right_arm"
    end_effector_frame_name = "arm2_link7"
    urdf_path = os.path.join(Arm.config_folder + "PsiRobot_DC_01.urdf")
    robot_description_path = os.path.join(Arm.config_folder + "PsiRobot_DC_01_right_arm_descriptor.yaml")
    eef_euler_yz_scale = np.array([-1.,1.])
    eef_euler_yz_bias = np.array([-np.pi/2,-np.pi])
    eef_rotation_bias = 0 #-1 #-1 #-1 #1.0903504
    
    joint_indices = np.array([1,3,5,7,9,11,13])

    ik_lower_limit = np.array([0.35,  -0.45,   0.9,   -0.5,   -0.2,  -1.6]) # 标准范围-更小范围
    ik_upper_limit = np.array([0.45,  -0.2,   1.2,  0.68,    0.14 ,  0.22])
    
    # eef_x_offset = 0.06014
    # eef_y_offset = 0.03425
    # eef_z_offset = 0.00665
    
    # eef_x_offset = 0.09680658 #0.14353-0.08277-0.00062+0.03666658
    # eef_y_offset = 0.00344287 #-0.02855-0.0057+0.03769287
    # eef_z_offset = 0.02804688 #0.00408+0.0021+0.00047+0.02139688
    
    
    eef_x_offset = 0.050171 #0.06014-0.01416911+0.00138318+0.00044656+0.00517277-0.0028024
    eef_y_offset = 0.02684164 #0.03425+0.01091721+0.00293165-0.0002203 -0.01976585-0.00127107
    eef_z_offset = 0.00032368 #0.00665-0.00884026+0.00079352+0.0002601+0.00311332 -0.001653
    
    prim_paths_expr = [
        "/hand2_link_base",
    ]

    # def __init__(self,cfg:DualArmScenarioCfg,robot_prim_path: str,articulation: Articulation):
    #     super().__init__(cfg,robot_prim_path,articulation)
    
    def convert_ik_target(self,ik_target):
        ik_target[0] += self.eef_x_offset
        ik_target[1] += self.eef_y_offset
        ik_target[2] += self.eef_z_offset
        return ik_target

    def convert_forward_kinematics(self,pose):
        pose[0] -= self.eef_x_offset
        pose[1] -= self.eef_y_offset
        pose[2] -= self.eef_z_offset
        return pose

    def eef_euler_yz_to_quats(self,euler_yz):
        return - Arm.eef_euler_yz_to_quats(self,euler_yz)




class Hand:
    # joint_indices = np.zeros([11])
    joint_indices = np.array([6])
    joint_indices_11 = np.array([6])
    joint_lower_limits = np.array([0.0400, 1.7500, 1.7100, 1.7700, 1.7300,  # 手 5 个手指的第一个关节
                                            # -0.5000, 0.0284, 0.0688, 0.0600, 0.0900, # 手 5 个手指的第一个关节
                                            0.0000]) # 大拇指旋转
    joint_upper_limits = np.array([0.6416, 3.1100, 3.0700, 3.0800, 3.0500,  # 手 5 个手指的第一个关节
                                        # 0.0400, 1.5700, 1.5700, 1.5700, 1.5700,  # 手 5 个手指的第二个关节
                                        1.5700]) # 大拇指旋转

    joint_scale = joint_upper_limits - joint_lower_limits

    joint_11_to_6_indices = np.array([1,3,5,7,9,11])
    joint_6_to_3_indices = np.array([0,1,-1])

    prim_paths_expr = None
    
    def __init__(self,cfg:ScenarioCfg, robot_prim_path: str,articulation: Articulation):
        self.cfg = cfg
        self.robot_prim_path = robot_prim_path
        self.articulation = articulation
        self.default_joint_pos = np.array(cfg.default_joint_pos)
        self.joint_target = self.default_joint_pos[self.joint_indices]

        self.joint_normalizer_N1_1 = Normalizer_N1_1(self.joint_lower_limits, self.joint_upper_limits)
        self.joint_normalizer_0_1 = Normalizer_0_1(self.joint_lower_limits, self.joint_upper_limits)


        rim_paths_expr = list(map(lambda x: robot_prim_path + x, self.prim_paths_expr))
        self.xform_view = XFormPrimView(prim_paths_expr=rim_paths_expr,name="xform_prim_view")
        self.joint_controller = JointController(self)

    def setup(self):
        pass

    
    def set_joint_real_target_6(self,joint_target_real_6): # real 是 -1 - 1
        '''
        设置手关节目标
        输入  6维 -1到1
        '''
        self.set_articulation_action_real_from6(joint_target_real_6)
        # joint_real_target_11 = self.action_6_to_11(joint_target_real_6)
        # self.set_articulation_action_real(joint_real_target_11)

    # 通过 3 维 0,1,-1 位置 获取 6 维 -1到1 位置
    def get_joint_real_target_3(self,joint_target_real_3):
        real_6 = self.action_3_to_6(joint_target_real_3) # 0-1
        abs_6 = self.joint_normalizer_0_1.denormalize(real_6) # 绝对值
        # abs_6 = self.joint_normalizer_N1_1.denormalize(real_6)
        return abs_6
    
    def set_joint_real_target_3(self,joint_target_real_3):
        '''
        设置手关节目标
        输入  3维 -1到1  【0,1,-1】位置
        '''
        self.set_articulation_action_real_from6(self.get_joint_real_target_3(joint_target_real_3))

    # def set_articulation_action_abs_from6(self,joint_abs_target_6):
    #     self.joint_controller.set_joint_target(self.joint_abs_target_6)

        
    def set_articulation_action_real_from6(self,joint_real_target_6):
        '''
        设置手关节目标
        输入  6维 -1到1
        转为实际关节角度
        '''
        self.joint_target = self.real_action_to_abs(joint_real_target_6)
        # print(f"joint_real_target_6: {joint_real_target_6}, joint_target: {self.joint_target}")
        self.joint_controller.set_joint_target(self.joint_target)


    # def set_articulation_action_real(self,joint_real_target_11):
    #     self.joint_target = self.real_action_to_abs(joint_real_target_11)
    #     self.joint_controller.set_joint_target(self.joint_target)

    # def action_3_to_11(self,action_3):
    #     return np.array([action_3[0],action_3[1],action_3[1],action_3[1],action_3[1],
    #                     action_3[0],action_3[1],action_3[1],action_3[1],action_3[1],
    #                     action_3[2]])

    # def action_3_to_11(self,action_3):
    #     return np.array([action_3[0],action_3[1],action_3[1],action_3[1],action_3[1],
    #                     action_3[0],action_3[1],action_3[1],action_3[1],action_3[1],
    #                     action_3[2]])

    def action_3_to_6(self,action_3):
        return np.array([action_3[0],action_3[1],action_3[1],action_3[1],action_3[1],
                        action_3[2]])

    # def action_6_to_11(self,action_6):
    #     return np.array([*action_6[:5],*action_6[:5],action_6[5]])

    # def action_11_to_6(self,action_11):
    #     return action_11[self.joint_11_to_6_indices]


    def real_action_to_abs(self,real_action):
        """
        将 real 空间的动作转换为 abs 空间的动作。只接受6维。
        """
        # abs_action = self.joint_normalizer_N1_1.denormalize(real_action)
        abs_action = self.joint_normalizer_0_1.denormalize(real_action)
        return abs_action

    def abs_action_to_real_3(self,abs_action):
        """
        将 abs 空间的动作转换为 real 空间的动作。只接受6维。
        """
        real_action = self.joint_normalizer_0_1.normalize(abs_action)
        return real_action[self.joint_6_to_3_indices]
    
    # 获取手指位置
    def get_finger_positions(self):
        return self.articulation.get_joint_positions(joint_indices=self.joint_indices[:])
    # 获取手指位置
    def get_finger_positions_11(self):
        return self.articulation.get_joint_positions(joint_indices=self.joint_indices_11[:])

    def step(self):
        pass


    # 单独设置和 控制 hand 到指定位置 joint pos
    def set_joint_pos(self,joint_pos):
        self.joint_target = joint_pos
        self.joint_controller.set_joint_target(self.joint_target)
        
        
    def reach_joint_target(self):
        return self.joint_controller.reach_joint_target()
    
    def get_hand_joint_policy_3(self):
        hand_joint = self.get_finger_positions()
        hand_joint_normalized = self.joint_normalizer_N1_1.normalize(hand_joint)
        # hand_joint_normalized = self.joint_normalizer_0_1.normalize(hand_joint)
        hand_joint_3 = hand_joint_normalized[[0,1,-1]]
        return hand_joint_3

    def get_hand_joint_policy_6(self):
        hand_joint = self.get_finger_positions()
        hand_joint_normalized = self.joint_normalizer_N1_1.normalize(hand_joint)
        # hand_joint_normalized = self.joint_normalizer_0_1.normalize(hand_joint)
        hand_joint_3 = hand_joint_normalized[[0,1,2,3,4,5]]
        return hand_joint_3

    def get_hand_joint_real_3(self):
        hand_joint = self.get_finger_positions()
        # hand_joint_normalized = self.joint_normalizer_N1_1.normalize(hand_joint)
        hand_joint_normalized = self.joint_normalizer_0_1.normalize(hand_joint)
        hand_joint_3 = hand_joint_normalized[[0,1,-1]]
        return hand_joint_3
    
    def get_hand_joint_real_6(self):
        hand_joint = self.get_finger_positions()
        # hand_joint_normalized = self.joint_normalizer_N1_1.normalize(hand_joint)
        hand_joint_normalized = self.joint_normalizer_0_1.normalize(hand_joint)
        hand_joint_6 = hand_joint_normalized[[0,1,2,3,4,5]]
        return hand_joint_6

class LeftHand(Hand):
    # joint_indices =  [24,15,16,17,18,  34,25,26,27,28,  14 ] # 统一为 5个 + 5个 + 1 个旋转
    joint_indices = [24,15,16,17,18,  14 ] 
    # joint_indices_11 =  [24,15,16,17,18,  34,25,26,27,28,  14 ]
    joint_indices_11 =  [14,24,34,15,25,16,26,17,27,18,28]  # urdf
    joint_11_to_6_indices = [24,15,16,17,18, 14]
    prim_paths_expr = [
        "/hand1_link_base/hand1_center",
        "/hand1_link_1_4",
        "/hand1_link_[2-5]_3"
    ]
# 14: 'hand1_joint_link_1_1'
# 15: 'hand1_joint_link_2_1'
# 16: 'hand1_joint_link_3_1'
# 17: 'hand1_joint_link_4_1'
# 18: 'hand1_joint_link_5_1'
# 19: 'hand2_joint_link_1_1'
# 20: 'hand2_joint_link_2_1'
# 21: 'hand2_joint_link_3_1'
# 22: 'hand2_joint_link_4_1'
# 23: 'hand2_joint_link_5_1'
# 24: 'hand1_joint_link_1_2'
# 25: 'hand1_joint_link_2_2'
# 26: 'hand1_joint_link_3_2'
# 27: 'hand1_joint_link_4_2'
# 28: 'hand1_joint_link_5_2'
# 29: 'hand2_joint_link_1_2'
# 30: 'hand2_joint_link_2_2'
# 31: 'hand2_joint_link_3_2'
# 32: 'hand2_joint_link_4_2'
# 33: 'hand2_joint_link_5_2'
# 34: 'hand1_joint_link_1_3'
# 35: 'hand2_joint_link_1_3'

class RightHand(Hand):
    # joint_indices = [29,20,21,22,23,  35,30,31,32,33,  19 ] # 统一为 5个 + 5个 + 1 个旋转 action 的实际输出顺序
    joint_indices = [29,20,21,22,23,  19 ] 
    # joint_indices_11 =  [29,20,21,22,23,  35,30,31,32,33,  19 ]
    joint_indices_11 =  [19,29,35,20,30,21,31,22,32,23,33] # urdf
    joint_11_to_6_indices = [29, 20, 21, 22, 23, 19]
    prim_paths_expr = [
        "/hand2_link_base/hand2_center",
        "/hand2_link_1_4",
        "/hand2_link_[2-5]_3"
        ]
class ArmHand:
    arm:Arm
    hand:Hand
    
    arm_cls = Arm
    hand_cls = Hand
    xyz_lower_limit = np.array([-0.8,       -0.8,    0.98]) 
    xyz_upper_limit = np.array([0.8,        -0.1,    1.5])


    def __init__(self,cfg:ScenarioCfg, robot_prim_path: str,articulation: Articulation):
        self.cfg = cfg
        self.robot_prim_path = robot_prim_path
        self.articulation = articulation
        self.arm = self.arm_cls(cfg, robot_prim_path, articulation)
        self.hand = self.hand_cls(cfg, robot_prim_path, articulation)
    
        self.joint_indices = np.concatenate([self.arm.joint_indices,self.hand.joint_indices])
        self.joint_lower_limits = np.concatenate([self.arm.joint_lower_limits,self.hand.joint_lower_limits])
        self.joint_upper_limits = np.concatenate([self.arm.joint_upper_limits,self.hand.joint_upper_limits])
        self.articulation_action = None

        self.joint_controller = JointController(self)
        self.eef_forward_direction = np.array([0,0,1])


    def setup(self):
        self.arm.setup()
        self.hand.setup()


    def get_joint_target_real_ik_action_9(self,real_ik_action):
        success,arm_real_action = self.arm.get_ik_target_real(real_ik_action[:6])
        hand_real_action = self.hand.get_joint_real_target_3(real_ik_action[6:]) # 先操作手
        
        # print("hand_real_action", hand_real_action)
        if not success:
            return False,None
        
        joint_target = np.concatenate([arm_real_action,hand_real_action])
        return True,joint_target
    
    def set_real_ik_action_9(self,real_ik_action):
        # print("real_ik_action", real_ik_action)
        success,joint_target = self.get_joint_target_real_ik_action_9(real_ik_action)
        
        # print("joint_target", joint_target)
        
        if not success:
            return False
        self.joint_controller.set_joint_target(joint_target)
        return True,joint_target

    # ToS 有问题
    # def set_real_ik_action_6(self,real_ik_action):
    #     # print("real_ik_action", real_ik_action)
    #     # 代表后面 3 位用原来的
    #     real_ik_action = np.concatenate([real_ik_action[:],self.hand.get_hand_joint_real_3()])
    #     success,joint_target = self.get_joint_target_real_ik_action_9(real_ik_action)
    #     if not success:
    #         return False
    #     self.joint_controller.set_joint_target(joint_target)
    #     return True,joint_target
    
    # ToS
    # 只有 6 个 IK 动作
    def set_real_ik_action_6(self,real_ik_action):
        self.arm.set_ik_target_real(real_ik_action[:6])

        if self.arm.joint_target is None:
            return False
        self.joint_target = np.concatenate([self.arm.joint_target,self.hand.joint_target])
        self.joint_controller.set_joint_target(self.joint_target)
        return True


    # def set_real_ik_action_12(self,real_ik_action):
    #     self.arm.set_ik_target_real(real_ik_action[:6])
    #     self.hand.set_joint_real_target_6(real_ik_action[6:])
    #     self.joint_target = np.concatenate([self.arm.joint_target,self.hand.joint_target])
    #     self.joint_controller.set_joint_target(self.joint_target)

    def check_joint_target_valid(self):
        target_mixed_pose = self.arm.get_mixed_pose_joint(self.arm.joint_target)
        # target_mixed_pose = self.arm.convert_forward_kinematics(target_mixed_pose)
        if (target_mixed_pose[:3] < self.xyz_lower_limit).any() or (target_mixed_pose[:3] > self.xyz_upper_limit).any():
            print_red(f"target_mixed_pose: {target_mixed_pose[:3]}, arm_hand: {self.__class__.__name__}")
            return False
        return True

    def reach_joint_target(self):
        if not self.check_joint_target_valid():
            return None, JointControlStatus.OUT_OF_LIMIT
        
        return self.joint_controller.reach_joint_target()   

    def get_joint_positions(self):
        return self.articulation.get_joint_positions(joint_indices=self.joint_indices)

    def get_end_effector_pos_quats(self):
        self.eef_xform_view = XFormPrimView(prim_paths_expr=[self.robot_prim_path +"/"+ self.arm.end_effector_frame_name],name="xform_prim_view")
        return np.squeeze(self.eef_xform_view.get_world_poses()[0][:]),np.squeeze(self.eef_xform_view.get_world_poses()[1][:])

    def get_link_pos_quats(self,link_name): # 根据 link_name 获取位置和姿态
        self.link_xform_view = XFormPrimView(prim_paths_expr=[self.robot_prim_path +"/"+ link_name],name="xform_prim_view")
        return np.squeeze(self.link_xform_view.get_world_poses()[0][:]),np.squeeze(self.link_xform_view.get_world_poses()[1][:])


    def get_real_action_9(self):
        return np.concatenate([self.arm.get_mixed_pose(),self.hand.get_hand_joint_policy_3()])

    def get_real_action_4(self):
        return self.get_real_action_9()[[0,1,2,6]]

    def get_real_action_5_safe(self):
        return self.get_real_action_9()[[0,1,2,4,6]]

    def get_real_action_5(self,left=False):
        return self.get_real_action_9()[[0,1,2,6,7]]

    def get_real_action_3(self):
        return self.get_real_action_9()[[0,1,2]]


class LeftArmHand(ArmHand):
    arm_cls = LeftArm
    hand_cls = LeftHand

    def __init__(self,cfg: ScenarioCfg, robot_prim_path: str,articulation: Articulation):
        super().__init__(cfg, robot_prim_path, articulation)
        self.xyz_lower_limit = cfg.l_xyz_lower_limit
        self.xyz_upper_limit = cfg.l_xyz_upper_limit

class RightArmHand(ArmHand):
    arm_cls = RightArm
    hand_cls = RightHand

    def __init__(self,cfg:ScenarioCfg, robot_prim_path: str,articulation: Articulation):
        super().__init__(cfg, robot_prim_path, articulation)
        self.xyz_lower_limit = cfg.r_xyz_lower_limit
        self.xyz_upper_limit = cfg.r_xyz_upper_limit

    
    
class DualArmHand:
    def __init__(self,cfg:ScenarioCfg, robot_prim_path: str,articulation: Articulation):
        self.cfg = cfg
        self.robot_prim_path = robot_prim_path
        self.articulation = articulation
        self.left_arm_hand = LeftArmHand(cfg, robot_prim_path, articulation)
        self.right_arm_hand = RightArmHand(cfg, robot_prim_path, articulation)
        self.joint_indices = np.concatenate([self.left_arm_hand.joint_indices,self.right_arm_hand.joint_indices])
        self.joint_lower_limits = np.concatenate([self.left_arm_hand.joint_lower_limits,self.right_arm_hand.joint_lower_limits])
        self.joint_upper_limits = np.concatenate([self.left_arm_hand.joint_upper_limits,self.right_arm_hand.joint_upper_limits])
        self.joint_controller = JointController(self)


    def setup(self):
        self.left_arm_hand.setup()
        self.right_arm_hand.setup()


    def set_real_ik_action_18(self,real_ik_action):
        left_success,left_joint_target = self.left_arm_hand.get_joint_target_real_ik_action_9(real_ik_action[:9])
        right_success,right_joint_target = self.right_arm_hand.get_joint_target_real_ik_action_9(real_ik_action[9:])
        if not left_success or not right_success:
            return False,None
        dual_joint_target = np.concatenate([left_joint_target,right_joint_target])
        self.joint_controller.set_joint_target(dual_joint_target)
        return True, dual_joint_target

    def reach_joint_target(self):
        if not self.left_arm_hand.check_joint_target_valid():
            return None, JointControlStatus.OUT_OF_LIMIT
        if not self.right_arm_hand.check_joint_target_valid():
            return None, JointControlStatus.OUT_OF_LIMIT
        return self.joint_controller.reach_joint_target()
   
        
class Robot:
    def __init__(self,cfg:ScenarioCfg, prim_path="/Dex"):
        self.cfg = cfg
        path_to_robot_usd = os.path.join(global_config.assets_path, "psi_robot/PsiRobot_DC_01.usd")
        self.stage = get_current_stage()
        add_reference_to_stage(path_to_robot_usd, prim_path)
        self.prim_path = prim_path
        self.xform_prim = XFormPrim(prim_path,orientation=utils.rot.euler_angles_to_quat(np.array([0,0,-np.pi/2])))
        self.articulation:Articulation = Articulation(prim_path)
        self.articulation_controller = self.articulation.get_articulation_controller()

        prim = get_prim_at_path(Sdf.Path(prim_path + "/base_link"))
        
        # self.left_arm_hand = LeftArmHand(cfg, self.prim_path, self.articulation)
        # self.right_arm_hand = RightArmHand(cfg, self.prim_path, self.articulation)
        self.dual_arm_hand = DualArmHand(cfg, self.prim_path, self.articulation)
        self.left_arm_hand = self.dual_arm_hand.left_arm_hand
        self.right_arm_hand = self.dual_arm_hand.right_arm_hand
        self.default_joint_pos = cfg.default_joint_pos

        self.arm_name_dict:Dict[str, Arm] = {
            'Left': self.left_arm_hand.arm,
            'Right': self.right_arm_hand.arm,
        }
        self.arm_hand_name_dict:Dict[str, ArmHand] = {
            'Left': self.left_arm_hand,
            'Right': self.right_arm_hand,
        }
        self.arm_hands: List[ArmHand] = [self.right_arm_hand,self.left_arm_hand]

        self.right_pose_forward_direction = np.array([0,0,1])


    def setup(self):
        self.articulation_controller.set_gains(kps=self.cfg.controller_kps,kds=self.cfg.controller_kds)

        for arm_hand in self.arm_hands:
            arm_hand.setup()
        self.dual_arm_hand.setup()
        
    def reset(self):
        # print(f"self.default_joint_pos: {self.default_joint_pos}")
        self.articulation.set_joint_positions(self.default_joint_pos)
        
        # joint_pos = self.articulation.get_joint_positions()
        # print(f"joint_pos: {joint_pos}")
        
        # 使用各个手部设置的 joint_target，而不是直接使用 default_joint_pos
        # left_hand_joints = self.left_arm_hand.hand.joint_target
        # right_hand_joints = self.right_arm_hand.hand.joint_target
        
        # # 创建完整的关节位置数组
        # full_joint_pos = self.default_joint_pos.copy()
        # # 设置左手关节位置
        # full_joint_pos[self.left_arm_hand.hand.joint_indices] = left_hand_joints
        # # 设置右手关节位置  
        # full_joint_pos[self.right_arm_hand.hand.joint_indices] = right_hand_joints
        # self.articulation.set_joint_positions(full_joint_pos)
        
        
    @property
    def left_arm(self):
        return self.left_arm_hand.arm

    @property
    def right_arm(self):
        return self.right_arm_hand.arm

    @property
    def left_hand(self):
        return self.left_arm_hand.hand

    @property
    def right_hand(self):
        return self.right_arm_hand.hand
    
    
    
class Scenario:
    cfg_cls: ScenarioCfg = ScenarioCfg
    cfg: ScenarioCfg
    robot_cls:Robot = Robot
    world:World
    stage: Usd.Stage

    def __init__(self,cfg:ScenarioCfg):
        self.cfg = cfg
        self.task_obj_list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_world(self, world:World):
        self.world = world

    def load_assets(self):
        self.stage = get_current_stage()

        self.load_ground_plane()
        self.load_robot()
        self.load_objects()
        self.load_vision()

    def load_vision(self):
        if self.cfg.enable_vision:
            self.top_camera = Camera(
                prim_path = "/Dex/base_camera_rgb/base_camera_rgb",
            )
            self.top_camera.initialize()
            self.top_camera.add_motion_vectors_to_frame()
            
            self.wrist_camera = Camera(
                prim_path = "/Dex/arm2_camera_rgb/arm2_camera_rgb",
            )
            self.wrist_camera.initialize()
            self.wrist_camera.add_motion_vectors_to_frame()
            
            self.top_camera_window = create_viewport_window(
                "Top Camera",
                camera_path="/Dex/base_camera_rgb/base_camera_rgb",
                width=128,
                height=128
            )
            self.wrist_camera_window = create_viewport_window(
                "Wrist Camera",
                camera_path="/Dex/arm2_camera_rgb/arm2_camera_rgb",
                width=128,
                height=128
            )            

            if not self.cfg.enable_vision:
                carb.log_info("enable_vision is not set in scenario cfg, set it to True")
                self.cfg.enable_vision = True
                self.load_vision() 

    def load_ground_plane(self, color=None, floor_type="default"):
        """
        加载地板平面
        Args:
            color: 地板颜色，可以是 RGB 元组 (r, g, b) 或颜色名称字符串
                  例如: (0.2, 0.8, 0.2) 或 "green" 或 "wooden_yellow"
            floor_type: 地板类型，可选值：
                       - "default": 默认平面地板
                       - "gridroom_curved": Isaac Sim内置的网格房间地板（带弯曲效果）
                       - "gridroom": Isaac Sim内置的网格房间地板
                       - "gridroom_curved_white": 白色网格房间地板
                       - "gridroom_curved_black": 黑色网格房间地板
        """
        if floor_type == "default":
            # 使用默认的GroundPlane
            self.ground_plane = GroundPlane("/World/Ground")
            self.world.scene.add(self.ground_plane)
            
            # 如果指定了颜色，设置地板材质
            if color is not None:
                # 尝试使用Isaac Sim的内置方法
                if self._set_ground_color_simple(color):
                    print(f"使用简单方法成功设置地板颜色: {color}")
        else:
            # 使用Isaac Sim内置的地板
            self._load_isaac_floor(floor_type)
    
    def _load_isaac_floor(self, floor_type):
        """
        加载Isaac Sim内置的地板
        """
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.core.utils.prims import get_prim_at_path
            from pxr import Sdf
            import carb
            import os
            
            # 获取data_path
            from dexrl.global_config import data_path
            
            # 本地地板资源路径映射
            local_floor_assets = {
                "gridroom_curved": f"{data_path}/Assets/IsaacSim/Assets/Isaac/4.2/Isaac/Environments/Grid/gridroom_curved.usd",
            }
            if floor_type not in local_floor_assets:
                print(f"未知的地板类型: {floor_type}，使用默认gridroom_curved")
                floor_type = "gridroom_curved"
            
            # 检查是否已经存在地板
            existing_ground = get_prim_at_path("/World/Ground")
            if existing_ground:
                print("删除现有的地板...")
                # 删除现有的地板
                from omni.isaac.core.utils.prims import delete_prim
                delete_prim("/World/Ground")
            
            # 尝试加载本地地板资源
            floor_asset_path = local_floor_assets[floor_type]
            
            print(f"正在加载Isaac Sim内置地板: {floor_type}")
            print(f"尝试路径: {floor_asset_path}")
            
            # 检查文件是否存在
            if os.path.exists(floor_asset_path):
                print(f"✓ 找到本地资源文件: {floor_asset_path}")
                try:
                    add_reference_to_stage(floor_asset_path, "/World/Ground")
                    
                    # 检查是否成功加载
                    ground_prim = get_prim_at_path("/World/Ground")
                    if ground_prim:
                        print(f"✓ 成功加载Isaac Sim内置地板: {floor_type}")
                        return True
                    else:
                        print(f"✗ 加载失败: 无法找到Ground prim")
                        return False
                        
                except Exception as load_error:
                    print(f"✗ 加载本地资源失败: {load_error}")
                    return self._fallback_to_default_floor()
                
        except Exception as e:
            print(f"✗ 加载Isaac Sim内置地板时出错: {e}")
            print("回退到默认地板...")
            return self._fallback_to_default_floor()
    
    def _fallback_to_default_floor(self):
        """
        回退到默认地板
        """
        try:
            print("创建默认地板...")
            from omni.isaac.core.objects import GroundPlane
            from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim
            
            # 删除可能存在的失败加载
            existing_ground = get_prim_at_path("/World/Ground")
            if existing_ground:
                delete_prim("/World/Ground")
            
            # 创建默认地板
            self.ground_plane = GroundPlane("/World/Ground")
            self.world.scene.add(self.ground_plane)
            
            print("✓ 成功创建默认地板")
            return True
            
        except Exception as e:
            print(f"✗ 创建默认地板失败: {e}")
            return False


    def _set_ground_color_simple(self, color):
        """
        使用Isaac Sim内置方法设置地板颜色（更简单可靠）
        """
        try:
            from pxr import UsdShade, Sdf, Gf
            
            # 颜色映射字典
            color_map = {
                # 基础颜色
                "red": (1.0, 0.0, 0.0),
                "green": (0.0, 1.0, 0.0),
                "blue": (0.0, 0.0, 1.0),
                "yellow": (1.0, 1.0, 0.0),
                "cyan": (0.0, 1.0, 1.0),
                "magenta": (1.0, 0.0, 1.0),
                "white": (1.0, 1.0, 1.0),
                "black": (0.0, 0.0, 0.0),
                "gray": (0.5, 0.5, 0.5),
                "grey": (0.5, 0.5, 0.5),
                "orange": (1.0, 0.5, 0.0),
                "purple": (0.5, 0.0, 0.5),
                "brown": (0.6, 0.4, 0.2),
                "pink": (1.0, 0.75, 0.8),
                "light_blue": (0.5, 0.8, 1.0),
                "light_green": (0.7, 1.0, 0.7),
                "light_gray": (0.8, 0.8, 0.8),
                "dark_gray": (0.2, 0.2, 0.2),
                
                # 木制地板颜色
                "wooden_yellow": (0.8, 0.6, 0.3),      # 木制黄色
                "wooden_brown": (0.6, 0.4, 0.2),       # 木制棕色
                # "wooden_dark": (0.4, 0.25, 0.1),       # 深木色
                "wooden_dark": (0.25, 0.2, 0.1),       # 深木色
                "wooden_light": (0.9, 0.7, 0.4),       # 浅木色
                "wooden_oak": (0.7, 0.5, 0.3),         # 橡木色
                "wooden_pine": (0.85, 0.65, 0.35),     # 松木色
                "wooden_walnut": (0.5, 0.3, 0.15),     # 胡桃木色
                "wooden_teak": (0.75, 0.55, 0.25),     # 柚木色
            }
            
            # 处理颜色输入
            if isinstance(color, str):
                if color.lower() in color_map:
                    rgb_color = color_map[color.lower()]
                else:
                    print(f"未知颜色名称: {color}，使用默认木制黄色")
                    rgb_color = (0.8, 0.6, 0.3)  # 默认木制黄色
            elif isinstance(color, (list, tuple)) and len(color) == 3:
                rgb_color = tuple(color)
            else:
                print(f"无效的颜色格式: {color}，使用默认木制黄色")
                rgb_color = (0.8, 0.6, 0.3)  # 默认木制黄色
            
            # 获取stage
            stage = get_current_stage()
            
            # 直接使用GroundPlane的prim路径
            ground_prim = get_prim_at_path("/World/Ground")
            if not ground_prim:
                print("无法找到Ground prim")
                return False
            
            # 创建材质
            material_path = "/World/Ground/Material"
            material = UsdShade.Material.Define(stage, material_path)
            
            # 创建着色器
            shader_path = f"{material_path}/Shader"
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            
            # 设置颜色
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb_color))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            
            # 根据是否为木制材质调整粗糙度
            if "wooden" in str(color).lower():
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)  # 木制材质较粗糙
            else:
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            
            # 连接着色器到材质
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            # 直接绑定到Ground prim
            UsdShade.MaterialBindingAPI(ground_prim).Bind(material)
            
            print(f"简单方法设置地板颜色成功: {color} (RGB: {rgb_color})")
            return True
            
        except Exception as e:
            print(f"简单方法设置地板颜色失败: {e}")
            return False


    def _print_prim_children(self, prim, depth=0, max_depth=2):
        """递归打印prim的子结构"""
        if depth > max_depth:
            return
            
        indent = "  " * depth
        print(f"{indent}- {prim.GetPath()}")
        
        for child in prim.GetChildren():
            self._print_prim_children(child, depth + 1, max_depth)

    def _add_grid_texture(self, geometry_prim, base_color):
        """
        为地板添加网格纹理
        Args:
            geometry_prim: 几何体prim
            base_color: 基础颜色 (R, G, B)
        """
        try:
            from pxr import UsdShade, Sdf, Gf, UsdGeom
            
            # 获取stage
            stage = get_current_stage()
            
            # 创建网格材质
            material_path = "/World/Ground/GridMaterial"
            material = UsdShade.Material.Define(stage, material_path)
            
            # 创建着色器
            shader_path = f"{material_path}/Shader"
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            
            # 设置基础颜色
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            
            # 创建网格纹理着色器
            grid_shader_path = f"{material_path}/GridShader"
            grid_shader = UsdShade.Shader.Define(stage, grid_shader_path)
            grid_shader.CreateIdAttr("UsdUVTexture")
            
            # 创建UV坐标着色器
            uv_shader_path = f"{material_path}/UVShader"
            uv_shader = UsdShade.Shader.Define(stage, uv_shader_path)
            uv_shader.CreateIdAttr("UsdPrimvarReader_float2")
            uv_shader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            
            # 创建网格纹理
            grid_texture_path = f"{material_path}/GridTexture"
            grid_texture = UsdShade.Shader.Define(stage, grid_texture_path)
            grid_texture.CreateIdAttr("UsdUVTexture")
            
            # 设置网格纹理参数
            grid_size = 10.0  # 网格大小
            grid_width = 0.02  # 网格线宽度
            grid_color = (0.3, 0.3, 0.3)  # 网格线颜色（深灰色）
            
            # 创建程序化网格纹理
            grid_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("")
            grid_texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            grid_texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            
            # 连接UV坐标到网格纹理
            grid_texture.CreateInput("st", Sdf.ValueTypeNames.Token).ConnectToSource(uv_shader.ConnectableAPI(), "result")
            
            # 创建混合着色器
            mix_shader_path = f"{material_path}/MixShader"
            mix_shader = UsdShade.Shader.Define(stage, mix_shader_path)
            mix_shader.CreateIdAttr("UsdMix")
            
            # 连接网格纹理到混合着色器
            mix_shader.CreateInput("bg", Sdf.ValueTypeNames.Color3f).ConnectToSource(grid_texture.ConnectableAPI(), "rgb")
            mix_shader.CreateInput("fg", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*grid_color))
            mix_shader.CreateInput("mix", Sdf.ValueTypeNames.Float).Set(0.3)  # 混合强度
            
            # 连接混合结果到主着色器
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(mix_shader.ConnectableAPI(), "result")
            
            # 连接着色器到材质
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            # 绑定材质到几何体
            UsdShade.MaterialBindingAPI(geometry_prim).Bind(material)
            
            print("成功添加网格纹理")
            
        except Exception as e:
            print(f"添加网格纹理时出错: {e}")
            # 如果网格纹理失败，至少确保基础颜色被设置
            try:
                # 重新绑定基础材质
                material_path = "/World/Ground/Material"
                material = UsdShade.Material.Define(stage, material_path)
                shader_path = f"{material_path}/Shader"
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(geometry_prim).Bind(material)
            except Exception as e2:
                print(f"重新设置基础颜色时出错: {e2}")

    def load_robot(self):
        self.robot:Robot = self.robot_cls(self.cfg)
        self.world.scene.add(self.robot.articulation)

    def load_objects(self):
        pass

    def setup(self):
        self.setup_viewport_camera()
        self.robot.setup()
        self.reset()

    def reset(self):
        self.robot.articulation.set_joint_positions(self.robot.default_joint_pos)
        return None,None,None

    def setup_viewport_camera(self):
        set_camera_view(eye=self.cfg.viewport_camera_pos_lookat[:3], target=self.cfg.viewport_camera_pos_lookat[3:])
        # set_camera_view(eye=[2.47,-0.06,1.48], target=[1.56,0.26,1.20], camera_prim_path="/OmniverseKit_Persp")

    # def sample_action(self):
    #     return np.random.rand(len(self.low_limit)) * (self.high_limit - self.low_limit) + self.low_limit

    def get_sim_info(self):
        return {
            "backend": self.world.backend,
            "device": self.world.device
        }

    def get_camera_rgb(self):
        assert self.cfg.enable_vision
        return self.top_camera.get_rgb()

    def get_camera_rgb(self):
        assert self.cfg.enable_vision
        top_rgb = self.top_camera.get_rgb()
        wrist_rgb = self.wrist_camera.get_rgb()
        camera_dict = {
            "top_rgb": top_rgb,
            "wrist_rgb": wrist_rgb
        }
        return camera_dict
    
    def close(self):
        pass

    def get_observation(self):
        return NotImplementedError("get_observation is not implemented")

    def get_reward(self):
        return NotImplementedError("get_reward is not implemented")

    def get_terminated(self):
        return NotImplementedError("get_terminated is not implemented")

    def get_truncated(self):
        return NotImplementedError("get_truncated is not implemented")

    def get_info(self):
        return NotImplementedError("get_info is not implemented")

    def step(self, action: float):
        return NotImplementedError("get_info is not implemented")


# 使用示例：
# 
# # 在子类中重写 load_ground_plane 方法来设置不同的地板
# class MyScenario(Scenario):
#     def load_ground_plane(self):
#         # 使用Isaac Sim内置的网格房间地板（推荐）
#         super().load_ground_plane(floor_type="gridroom_curved")
#         
#         # 其他地板选项：
#         # super().load_ground_plane(floor_type="gridroom")              # 标准网格房间
#         # super().load_ground_plane(floor_type="gridroom_curved_white") # 白色网格房间
#         # super().load_ground_plane(floor_type="gridroom_curved_black") # 黑色网格房间
#         # super().load_ground_plane(floor_type="warehouse")             # 仓库环境
#         # super().load_ground_plane(floor_type="office")                # 办公室环境
#         # super().load_ground_plane(floor_type="lab")                   # 实验室环境
#         
#         # 使用自定义颜色（仅对默认地板有效）
#         # super().load_ground_plane(color="wooden_yellow", floor_type="default")
#         # super().load_ground_plane(color=(0.8, 0.6, 0.3), floor_type="default")
# 
# # 支持的Isaac Sim内置地板：
# # "gridroom_curved" - 网格房间地板（带弯曲效果，推荐）
# # "gridroom" - 标准网格房间地板
# # "gridroom_curved_white" - 白色网格房间地板
# # "gridroom_curved_black" - 黑色网格房间地板
# # "warehouse" - 仓库环境
# # "office" - 办公室环境
# # "lab" - 实验室环境
# 
# # 支持的自定义颜色（仅对默认地板有效）：
# # "wooden_yellow", "wooden_brown", "wooden_dark", "wooden_light"
# # "wooden_oak", "wooden_pine", "wooden_walnut", "wooden_teak"
# # "red", "green", "blue", "yellow", "cyan", "magenta"
# # "white", "black", "gray", "orange", "purple", "brown"
# # 或者使用自定义RGB颜色元组，如 (0.8, 0.6, 0.3)
