import numpy as np
# from dexrl.sim.scenarios.base_scenario import Scenario
from dexrl.sim.scenarios.rl_grasp_5act_sim import RLGraspScenario
from dexrl.sim.scenarios.rl_grasp_5act_sim import RLGraspScenarioCfg
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache

from dexrl.sim.scenarios._cfg import ScenarioCfg
from dexrl.utils import configclass
from dexrl.sim.objects import BoxOfCaneSugar, BaseObject, ThickCoconutMilk,Biscuit
from dexrl.sim import utils
from omni.isaac.core.objects import VisualCuboid
from dexrl.sim.utils import Normalizer_N1_1
from scipy.spatial.transform import Rotation as R
from dexrl.utils import rot

@configclass
class HandoverConfig:
    """Handover任务配置"""
    # 右手相对于物体的偏移
    x_offset_plus = -0.12
    y_offset_plus = -0.03
    
    # 左手位置参数
    x_lh_plus = 0.4
    y_lh_plus = -0.34
    z_lh_plus = 0.77
    
    # 动作执行参数
    action_steps = 20
    approach_distance = 0.02  # 接近物体的最终距离


@configclass
class RLBiHands10ActScenarioCfg(ScenarioCfg):

    
    default_joint_pos = np.array(
        [-0.10114183,0.0230558, 
         -1.61278892,-1.35585899 ,
         0.70649183,1.18980836,
         0.40805796,-0.42680282,
         -2.03365759, 0.0185703 ,
         -2.04454845,-1.82732479, 
         0.7979296,0.03886848,
         0.10,3.11,3.07,3.08,3.05,1.57,3.11,3.07,3.08,3.05,0.64,1.57,1.57,1.57,1.57,0.64,1.57,1.57,1.57,1.57,0.04,0.04]
        )
         
    table_pos = [0,-0.4,0.9]
    table_scale = [1.6,0.7,0.04]


    obj_cls = Biscuit
    # table_pos[2] + table_scale[2]/2 + 176.328*Biscuit.scale_ratio_z/2
    obj_init_pos = [-0.15, -0.38, 0.997]
    # obj_init_pos = [-0.15, -0.4, table_pos[2] + table_scale[2]/2 + 176.328*Biscuit.scale_ratio_z/2]
    obj_init_euler = [0, 1.57, 0] # z改为 -1.57/6 1.57/6
    # obj_delta_euler = 1.57/6
    obj_mass = 0.2 #0.2
 
    target_obj_height_offset = 0.1 # 0.15


    viewport_camera_pos_lookat = -0.14,-2.78,1.69,-0.17,-1.80,1.50

    enable_vision = False
    max_env_step = 600

    # 最后三位尝试直接赋值
    limit_range = 0.05
    hand_limit_range = 0.5 #0.2
    # policy_act_lower_limit = np.array([-limit_range,     -limit_range,   -limit_range,      -limit_range,   -limit_range,     -limit_range,       -1,   -1,  -1]) 
    # policy_act_upper_limit = np.array([limit_range,       limit_range,    limit_range,       limit_range,    limit_range,       limit_range,          1,     1,   1])
    r_policy_act_lower_limit = np.array([-limit_range,     -limit_range,   -limit_range,            -hand_limit_range,   -hand_limit_range]) 
    r_policy_act_upper_limit = np.array([limit_range,       limit_range,    limit_range,             hand_limit_range,     hand_limit_range])
    
    l_policy_act_lower_limit = r_policy_act_lower_limit
    l_policy_act_upper_limit = r_policy_act_upper_limit
    
    policy_act_lower_limit = np.concatenate([r_policy_act_lower_limit,l_policy_act_lower_limit])
    policy_act_upper_limit = np.concatenate([r_policy_act_upper_limit,l_policy_act_upper_limit])
    
    # 设置范围
    right_real_act_lower_limit = np.array([-0.5,       -0.6,    1,                  -1,  -1]) 
    right_real_act_upper_limit = np.array([0.2,        -0.2,    1.35,                 1,  1])

    left_real_act_lower_limit = np.array([-0.2,       -0.6,    1,                 -1,  -1]) 
    left_real_act_upper_limit = np.array([0.49,        -0.2,    1.35,               1,  1])
    
    real_act_lower_limit = np.concatenate([right_real_act_lower_limit,left_real_act_lower_limit])
    real_act_upper_limit = np.concatenate([right_real_act_upper_limit,left_real_act_upper_limit])
    
    rh_rot = -1
    rh_y_euler = 1.57 
    rh_z_euler = -0.6
    
    lh_rot = 2
    lh_y_euler = 1.54
    lh_z_euler = -0.6

    
    hand_rot = 1
        



class RLBiHands10ActScenario(RLGraspScenario):
    cfg: RLBiHands10ActScenarioCfg
    cfg_cls = RLBiHands10ActScenarioCfg

    policy_steps = 0
    env_steps = 0
    accumulated_reward = 0
    obj_stable_steps=0

    # state buffers
    joint_pos_state = np.zeros(18)
    current_action_state = np.zeros(10)
    obj_pos_state = np.zeros(3)
    final_euler = np.zeros(3)
    obj_quat_state = np.zeros(4)
    obj_rot_6d = np.zeros(6)  # 6维旋转表示
    
    right_hand_view_pos = np.zeros((6,3))
    left_hand_view_pos = np.zeros((6,3))
    
    eef_pos = np.zeros(3)
    eef_quat = np.zeros(4)
    obj_init_pos = np.zeros(3)
    obj_init_quat = np.zeros(4)
    
    right_hand_joint_pos = np.zeros(6)
    left_hand_joint_pos = np.zeros(6)
    
    right_mixed_pose = np.zeros(6)
    left_mixed_pose = np.zeros(6)
    
    obj_front_center = np.zeros(3)
    obj_back_center = np.zeros(3)
    obj_right_center = np.zeros(3)
    obj_left_center = np.zeros(3)
    
    task_stage_count = np.zeros(5)
    task_reset_script = None
    task_stage = 0
    task_reset_script_stage = 0
    
    # Handover配置
    handover_cfg = HandoverConfig()
    
    def __init__(self,cfg:  RLBiHands10ActScenarioCfg):
        super().__init__(cfg)
        self.policy_action_normalizer = Normalizer_N1_1(self.cfg.policy_act_lower_limit, self.cfg.policy_act_upper_limit)
        self.right_policy_action_normalizer = Normalizer_N1_1(self.cfg.r_policy_act_lower_limit, self.cfg.r_policy_act_upper_limit)
        self.left_policy_action_normalizer = Normalizer_N1_1(self.cfg.l_policy_act_lower_limit, self.cfg.l_policy_act_upper_limit)

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
            )
        self.obj_prim = self.obj.create_prim()
        
        # 获取物体长宽高
        self.bbox_cache = create_bbox_cache()
        bounds = compute_aabb(self.bbox_cache, str(self.obj_prim.prim_path))
        self.obj_size = bounds[3:6] - bounds[0:3]  # 计算边界框的尺寸  长高(x) 横宽 (y) 厚款(z)
        # 排序 从大到小
        self.obj_size = np.sort(self.obj_size)[::-1]
        
        self.world.scene.add(self.obj_prim)
        self.task_obj_list.append(self.obj)
        
        
        self.vis_cube_right_mix_pose = VisualCuboid(
            name="vis_cube_right_mix_pose",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_right_mix_pose",
            size=0.05,
            color=np.array([0, 0, 1]), # 蓝色
        ) 
        self.vis_cube_left_mix_pose = VisualCuboid(
            name="vis_cube_left_mix_pose",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_left_mix_pose",
            size=0.05,
            color=np.array([0, 0, 1]), # 蓝色
        )  
        self.vis_cube_obj_front = VisualCuboid(
            name="vis_cube_obj_front",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_obj_front",
            size=0.03,
            color=np.array([0, 1, 0]), # 绿色
        )    
        self.vis_cube_obj_back = VisualCuboid(
            name="vis_cube_obj_back",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_obj_back",
            size=0.03,
            color=np.array([1, 0, 0]), # 红色
        ) 
        # self.vis_cube_right = VisualCuboid(
        #     name="vis_cube_right",
        #     position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
        #     prim_path="/World/vis_cube_right",
        #     size=0.03,
        #     color=np.array([0, 1, 1]), # 黄色
        # )
        # self.vis_cube_left = VisualCuboid(
        #     name="vis_cube_left",
        #     position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
        #     prim_path="/World/vis_cube_left",
        #     size=0.03,
        #     color=np.array([0, 0, 1]), # 蓝色
        # ) 

        
    def reset(self, stage=0):
        self.policy_steps = 0
        self.accumulated_reward = 0
        self.obj_stable_steps = 0
        self.task_stage = 0
        self.task_reset_script_stage = stage
        
        self.robot.reset()
        # self.task_reset()
        
        # x_offset = np.random.uniform(-0.15, 0.15)
        # y_offset = np.random.uniform(-0.15, 0.15)
        # delta_euler = np.random.uniform(-1.57/6, 1.57/6) #90度/6
        x_offset = 0
        y_offset = 0
        delta_euler = 0
        self.obj_pos = self.cfg.obj_init_pos + np.array([x_offset, y_offset, 0])
        
        # 修复：正确计算欧拉角，只传递3个值
        self.final_euler = np.array(self.cfg.obj_init_euler) + np.array([0, 0, delta_euler])
        self.obj.set_world_pose(self.obj_pos, 
                                utils.rot.euler_angles_to_quat(self.final_euler))
        
        # 根据阶段设置相应的任务重置脚本
        self.task_reset_script = self._get_stage_script(stage)
        
        return x_offset,y_offset,self.obj_pos
    
    def _get_stage_script(self, stage):
        """根据阶段返回对应的脚本生成器"""
        if stage == 0:
            return None
        elif stage == 1:
            return lambda: self.task_reset_script_1()
        elif stage == 2:
            return lambda: self._execute_stages([1, 2])
        elif stage == 3:
            return lambda: self._execute_stages([1, 2, 3])
        elif stage == 4:
            return lambda: self._execute_stages([1, 2, 3, 4])
        else:
            return None
    
    def _execute_stages(self, stages):
        """执行多个阶段的脚本"""
        for stage in stages:
            if stage == 1:
                for _ in self.task_reset_script_1():
                    yield()
            elif stage == 2:
                for _ in self.task_reset_script_2():
                    yield()
            elif stage == 3:
                for _ in self.task_reset_script_3():
                    yield()
            # elif stage == 4:
            #     for _ in self.task_reset_script_4():
            #         yield()
    
    def reset_and_set_obj_pose(self, obj_pos, obj_quat, stage=0):
        self.obj.set_world_pose(obj_pos, obj_quat)
        x_offset = obj_pos[0] - self.cfg.obj_init_pos[0]
        y_offset = obj_pos[1] - self.cfg.obj_init_pos[1]
        
        # 根据阶段设置相应的任务重置脚本
        self.task_reset_script = self._get_stage_script(stage)
            
        return x_offset,y_offset,obj_pos
    
    def _process_policy_action(self, action, is_left=False):     
        # 反归一化动作
        action = np.array(action)
        if is_left:
            real_delta_action = self.left_policy_action_normalizer.denormalize(action)
        else:
            real_delta_action = self.right_policy_action_normalizer.denormalize(action)
        
        
        # 获取当前状态
        last_mix_pose = self.robot.get_mixed_pose(left=is_left)
        last_hand_joint_3 = self.robot.get_hand_joint_policy_3(left=is_left)
        
        # 计算真实动作
        real_action = np.zeros(9)
        real_action[:3] = last_mix_pose[:3] + real_delta_action[:3] # 改为直接赋值！获取真实的 xyz
        # print(f"real_action[3:6]: {real_action[3:6]}")
        # print(f" [self.cfg.lh_rot, self.cfg.lh_y_euler, self.cfg.lh_z_euler]: { [self.cfg.lh_rot, self.cfg.lh_y_euler, self.cfg.lh_z_euler]}")
        
        if is_left:
            real_action[3:6] = [self.cfg.lh_rot, self.cfg.lh_y_euler, self.cfg.lh_z_euler]
        else:
            real_action[3:6] = [self.cfg.rh_rot, self.cfg.rh_y_euler, self.cfg.rh_z_euler]
        # real_action[6:] = last_hand_joint_3 + real_delta_action[6:] # 获取真实的 灵巧手关节位置
        # real_action[6:] = real_delta_action[6:] # 改为直接赋值！获取真实的 灵巧手关节位置
        
        real_action[6:8] = last_hand_joint_3[:2] + real_delta_action[3:5] # 获取真实的 灵巧手关节位置
        real_action[8:9] = self.cfg.hand_rot
        
        # 应用限制
        # real_action = np.clip(real_action, self.cfg.left_real_act_lower_limit, self.cfg.left_real_act_upper_limit)
        limits = (self.cfg.left_real_act_lower_limit, self.cfg.left_real_act_upper_limit) if is_left else \
                (self.cfg.right_real_act_lower_limit, self.cfg.right_real_act_upper_limit)
        real_action[:3] = np.clip(real_action[:3], limits[0][:3], limits[1][:3])
        real_action[6:8] = np.clip(real_action[6:8], limits[0][3:5], limits[1][3:5])
        
        # 执行动作

        arm_hand = self.robot.left_arm_hand if is_left else self.robot.right_arm_hand
        # if is_left:
        #     print(f"real_action: {real_action[:3]}")
        action_valid = arm_hand.set_real_ik_action_9(real_action)
               
        return action_valid, real_action

    def set_right_policy_action(self, action):
        """设置右手策略动作"""
        return self._process_policy_action(action, is_left=False)
    
    def set_left_policy_action(self, action):
        """设置左手策略动作"""
        return self._process_policy_action(action, is_left=True)

    def set_policy_action(self, action,only_left=False):
        if len(action) == 5 and not only_left:
            return self.set_right_policy_action(action)
        elif len(action) == 5 and only_left:
            return self.set_left_policy_action(action)
        elif len(action) != 10:
            raise ValueError(f"动作长度必须为18，当前为: {len(action)}")
        
        # print(f"action: {action}")
        
        # 分别设置左右手动作
        right_action_valid, right_real_action = self.set_right_policy_action(action[:5])
        left_action_valid, left_real_action = self.set_left_policy_action(action[5:])
        
        # 合并结果
        action_valid = right_action_valid and left_action_valid
        real_action = np.concatenate([right_real_action, left_real_action])
        
        # 更新状态
        self.current_action_state[:] = action
        self.policy_steps += 1
        
        return action_valid, real_action
        




    def get_rl_tuples(self,noise=True):
        self.update_state_buffers()

        observation = self.get_observation(noise)
        reward = self.get_reward()
        # self.accumulated_reward += reward
        terminated = self.get_terminated()
        truncated = self.get_truncated()
        # if terminated:
        #     reward += 100
        if terminated or truncated:
            infos = {
                "final_observation": observation,
                "sim_episode": {
                    # "r": self.accumulated_reward,
                    "r": reward,
                    "l": self.policy_steps
                }
            }
        else:
            infos = {}
        return observation, reward, terminated, truncated, infos

    def update_state_buffers(self):
        self.mixed_pose[:] = self.robot.get_mixed_pose() # 3+4
        self.vis_cube_right_mix_pose.set_world_pose(self.mixed_pose[:3])
        self.left_mixed_pose[:] = self.robot.get_mixed_pose(left=True) # 3+4
        self.vis_cube_left_mix_pose.set_world_pose(self.left_mixed_pose[:3])
        self.obj_pos_state[:], self.obj_quat_state[:] = self.obj.get_world_pose() # 3+4
        # print(f"obj_pos_state: {self.obj_pos_state}, obj_quat_state: {self.obj_quat_state}")
        # 更新6维旋转表示
        self.obj_rot_6d[:] = rot.quat_to_6d(self.obj_quat_state)
        
        self.right_hand_view_pos[:,:] = self.robot.right_arm_hand.hand.xform_view.get_world_poses()[0] # 手指中心
        # 获取 灵巧手 joint_po
        self.right_hand_joint_pos[:] = self.robot.right_arm_hand.get_joint_positions()[7:]
        # 大拇指到物体后方的中心点的距离
        self.right_thumb_eef_pos = self.right_hand_view_pos[1]
        # 无名指到物体前方的中心点的距离
        self.right_index_eef_pos = self.right_hand_view_pos[4] 
        self.right_hand_center = self.right_hand_view_pos[0]
        
        # 左手
        self.left_hand_view_pos[:,:] = self.robot.left_arm_hand.hand.xform_view.get_world_poses()[0] # 手指中心
        # 获取 灵巧手 joint_po
        self.left_hand_joint_pos[:] = self.robot.left_arm_hand.get_joint_positions()[7:]
        # 大拇指到物体后方的中心点的距离
        self.left_thumb_eef_pos = self.left_hand_view_pos[1]
        # 无名指到物体前方的中心点的距离
        self.left_index_eef_pos = self.left_hand_view_pos[4] 
        self.left_hand_center = self.left_hand_view_pos[0]
        
        # 获取物体前后中心点 # 获取物体的左右点
        self.obj_front_center, self.obj_back_center,self.obj_right_center,self.obj_left_center = self.get_obj_centers(
            self.obj_pos_state, self.obj_quat_state, self.obj_init_quat)
        # 更新可视化物体
        self.vis_cube_obj_front.set_world_pose(self.obj_front_center, self.obj_quat_state)
        self.vis_cube_obj_back.set_world_pose(self.obj_back_center, self.obj_quat_state)
        

        if self.policy_steps <=1 and self.task_reset_script_stage == 0:
            self.obj_init_pos[:], self.obj_init_quat[:] = self.obj.get_world_pose() # 3+4
        else:
            self.obj_init_pos[:], self.obj_init_quat[:] = self.obj.get_world_pose()
            self.obj_init_pos[:] = self.cfg.obj_init_pos

    def get_observation(self,noise=True):
        # 创建临时副本，避免直接修改状态缓冲区
        if noise:
            obj_pos_with_noise = self.obj_pos_state + np.random.uniform(-0.02, 0.02, 3)
            # obj_rot_6d_with_noise = self.obj_rot_6d + np.random.uniform(-0.02, 0.02, 6)
            z_euler_with_noise = self.final_euler[2] + np.random.uniform(-1.57/20, 1.57/20) # 4.5度的误差
        else:
            obj_pos_with_noise = self.obj_pos_state
            z_euler_with_noise = self.final_euler[2]
        
        # 给 box
        self.bbox_cache = create_bbox_cache()
        bounds = compute_aabb(self.bbox_cache, str(self.obj_prim.prim_path))
        
        final_euler_with_noise = np.array([self.final_euler[0], self.final_euler[1], z_euler_with_noise])
        # obj_pos_with_noise = self.obj_pos_state 
        obj_rot_6d_with_noise = rot.quat_to_6d(utils.rot.euler_angles_to_quat(final_euler_with_noise))
        
        # abs_hand_action = self.right_hand_joint_pos[[0,1,-1]]
        # 转为 0-1 的real空间
        hand_action_real_3 = self.robot.right_arm_hand.hand.abs_action_to_real_3(self.right_hand_joint_pos)[[0,1,-1]]
        
        state = np.concatenate([
                        # self.eef_pos,self.eef_quat, # 3+4
                        self.mixed_pose, # 6
                        hand_action_real_3,# self.right_hand_joint_pos[[0,1,-1]], # 6 
                        self.current_action_state, # 9
                        # self.right_hand_view_pos[0],手指中心位置
                        obj_pos_with_noise, # 3 (使用带噪声的临时副本)
                        # self.obj_quat_state, # 4
                        obj_rot_6d_with_noise, # 6 (使用带噪声的临时副本)
                        bounds, # 2
                        np.array([self.task_stage]), # 1    
                        ])
        # obs_dict = {
        #     # "image": self.get_camera_rgb(),
        #     "state": state
        # }
        # return obs_dict
        return state

    def get_obj_centers(self, object_position, current_quat, init_quat):
        
        # rot_init = R.from_quat(init_quat)
        # rot_current = R.from_quat(current_quat)
        
        # # 计算从初始姿态到当前姿态的相对旋转
        # relative_rot = rot_init.inv() * rot_current
        
        # # 在初始姿态下的偏移向量
        # init_front_offset = np.array([0, -self.obj_size[2]/2, 0])
        # init_back_offset = np.array([0, self.obj_size[2]/2, 0])
        # init_right_offset = np.array([-self.obj_size[1]/2, 0, 0])
        # init_left_offset = np.array([self.obj_size[1]/2, 0, 0])
        
        # # 将初始世界对齐的偏移量"反向旋转"回规范局部（体）坐标系。
        # # 应用相对旋转到偏移向量
        # rotated_front_offset = relative_rot.apply(init_front_offset)
        # rotated_back_offset = relative_rot.apply(init_back_offset)
        # rotated_right_offset = relative_rot.apply(init_right_offset)
        # rotated_left_offset = relative_rot.apply(init_left_offset)
        
        # # 计算表面中心点
        # obj_front_center = object_position + rotated_front_offset
        # obj_back_center = object_position + rotated_back_offset
        # obj_right_center = object_position + rotated_right_offset
        # obj_left_center = object_position + rotated_left_offset
        
        rot_matrix = utils.rot.quat_to_rot_matrix(current_quat)
        lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,-1,0]))
        x_lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,-1]))
        obj_front_center = object_position + lookat_direction * self.obj_size[2]/2
        obj_back_center = object_position - lookat_direction * self.obj_size[2]/2
        obj_right_center = object_position + x_lookat_direction * self.obj_size[1]/2
        obj_left_center = object_position - x_lookat_direction * self.obj_size[1]/2

        return obj_front_center,obj_back_center,obj_right_center,obj_left_center



    # hand 到 一个位置的距离
    def get_hand_to_position_dist_reward(self,obj_back_center,obj_front_center,left=False):
        if left:
            thumb_dist = np.linalg.norm(self.left_thumb_eef_pos - obj_back_center)
            index_dist = np.linalg.norm(self.left_index_eef_pos - obj_front_center)
            hand_center_dist = np.linalg.norm(
                self.left_hand_center - self.obj_pos_state)
        else:
            thumb_dist = np.linalg.norm(self.right_thumb_eef_pos - obj_back_center)
            index_dist = np.linalg.norm(self.right_index_eef_pos - obj_front_center)
            hand_center_dist = np.linalg.norm(
                self.right_hand_center - self.obj_pos_state)
            
        # 大拇指到物体后方的中心点的距离奖励
        thumb_dist_reward = - np.exp(3 * thumb_dist) + 2
        thumb_dist_reward = np.clip(thumb_dist_reward, 0.0, 1.0)
        # 无名指到物体前方的中心点的距离奖励
        index_dist_reward = - np.exp(2 * index_dist) + 2
        index_dist_reward = np.clip(index_dist_reward, 0.0, 1.0)
        
        # 计算手中心到物体中心的距离
        hand_center_dist_reward = - np.exp(3 * hand_center_dist) + 2
        hand_center_dist_reward = np.clip(hand_center_dist_reward, 0.0, 1.0)
        
        return thumb_dist,index_dist,hand_center_dist,thumb_dist_reward,index_dist_reward,hand_center_dist_reward

    def get_height_reward(self,initial_height,obj_right_center,obj_left_center):
        
        # 计算物体提升高度的奖励
        # 获取物体长宽高
        self.bbox_cache = create_bbox_cache()
        height_reward = min(
            (obj_right_center[2] - initial_height) / self.cfg.target_obj_height_offset,
            (obj_left_center[2] - initial_height) / self.cfg.target_obj_height_offset
        )
        
        height = min(
            (obj_right_center[2] - initial_height) ,
            (obj_left_center[2] - initial_height) 
        )
        # print(f"height: {initial_height}")
        # print(f"height: {height}")
        
        
        height_reward = np.clip(height_reward, 0.0, 1.0)
        
        return height,height_reward

    def get_hand_open_dist_reward(self):

        # 手指打开的奖励 大拇指到无名指的距离
        # 转为 0-1 的real空间
        hand_action_real_3 = self.robot.right_arm_hand.hand.abs_action_to_real_3(self.right_hand_joint_pos)[[0,1,-1]]
        hand_open_dist = hand_action_real_3[0] + hand_action_real_3[1]
        # hand_open_dist = np.linalg.norm(self.right_hand_view_pos[1] - self.right_hand_view_pos[4])
        # 0.07 是关上，0.12是打开 ; 映射到 [0.27  0.62]
        # hand_open_dist_reward = (hand_open_dist - 0.07) / (0.13 - 0.07)
        hand_open_dist_reward = (hand_open_dist) / 2
        hand_open_dist_reward = np.clip(hand_open_dist_reward, 0, 1)
        
        hand_close_dist_reward = (2-hand_open_dist) / (2+1e-6)
        hand_close_dist_reward = np.clip(hand_close_dist_reward, 0, 1)

        return hand_open_dist,hand_open_dist_reward,hand_close_dist_reward

    
    def _get_handover_params(self):
        """获取handover相关的参数"""
        obj_pos = self.obj.get_world_pose()[0]
        return {
            'obj_pos': obj_pos,
            'x_offset_plus': self.handover_cfg.x_offset_plus,
            'y_offset_plus': self.handover_cfg.y_offset_plus,
            'x_lh_plus': self.handover_cfg.x_lh_plus,
            'y_lh_plus': self.handover_cfg.y_lh_plus,
            'z_lh_plus': self.handover_cfg.z_lh_plus,
        }
    
    def _execute_arm_action(self, arm_hand, action, is_left=False, left_arm_hand=None, left_action=None):
        """执行手臂动作的通用方法"""
        if left_arm_hand is not None and left_action is not None:
            # 同时执行左右手动作
            # 设置右手动作
            rot_params = [self.cfg.rh_rot, self.cfg.rh_y_euler, self.cfg.rh_z_euler]
            arm_hand.set_real_ik_action_6(
                np.concatenate([action[:3], np.array(rot_params)])
            )
            arm_hand.hand.set_joint_real_target_3(
                np.concatenate([action[3:], np.array([self.cfg.hand_rot])])
            )
            
            # 设置左手动作
            left_rot_params = [self.cfg.lh_rot, self.cfg.lh_y_euler, self.cfg.lh_z_euler]
            left_arm_hand.set_real_ik_action_6(
                np.concatenate([left_action[:3], np.array(left_rot_params)])
            )
            left_arm_hand.hand.set_joint_real_target_3(
                np.concatenate([left_action[3:], np.array([self.cfg.hand_rot])])
            )
            
            # 等待动作完成
            for _ in range(self.handover_cfg.action_steps):
                arm_hand.arm.reach_joint_target()
                arm_hand.hand.reach_joint_target()
                left_arm_hand.arm.reach_joint_target()
                left_arm_hand.hand.reach_joint_target()
                yield()
        else:
            # 单臂执行（原有逻辑）
            if is_left:
                rot_params = [self.cfg.lh_rot, self.cfg.lh_y_euler, self.cfg.lh_z_euler]
            else:
                rot_params = [self.cfg.rh_rot, self.cfg.rh_y_euler, self.cfg.rh_z_euler]
            
            # 设置手臂位置和旋转
            arm_hand.set_real_ik_action_6(
                np.concatenate([action[:3], np.array(rot_params)])
            )
            
            # 设置手部关节
            if len(action) > 3:
                arm_hand.hand.set_joint_real_target_3(
                    np.concatenate([action[3:], np.array([self.cfg.hand_rot])])
                )
            
            # 等待动作完成
            for _ in range(self.handover_cfg.action_steps):
                arm_hand.arm.reach_joint_target()
                if len(action) > 3:
                    arm_hand.hand.reach_joint_target()
                yield()
    
    def task_reset_script_1(self):
        """阶段1：让手在物体边上"""
        params = self._get_handover_params()
        print(f"x_offset_plus: {params['x_offset_plus']}")
        
        # 右手移动到物体附近
        action_list = [
            [params['obj_pos'][0] + params['x_offset_plus'] - 0.12, params['obj_pos'][1] + params['y_offset_plus'], 1.02, 1, 1],
            [params['obj_pos'][0] + params['x_offset_plus'] - 0.08, params['obj_pos'][1] + params['y_offset_plus'], 1.02, 1, 1],
            [params['obj_pos'][0] + params['x_offset_plus'] - self.handover_cfg.approach_distance, params['obj_pos'][1] + params['y_offset_plus'], 1.02, 1, 1],
        ]
        
        for action in action_list:
            for _ in self._execute_arm_action(self.robot.right_arm_hand, action):
                yield()
    
    def task_reset_script_2(self):
        """阶段2：在阶段1基础上，右手抓住物体并且抬高"""
        params = self._get_handover_params()
        
        # 右手抓住物体并抬高
        grasp_and_lift_actions = [
            [params['obj_pos'][0] + params['x_offset_plus'] - self.handover_cfg.approach_distance, params['obj_pos'][1] + params['y_offset_plus'], 1.02, -1, 0.3],  # 关闭手
            [params['obj_pos'][0] + params['x_offset_plus'] - self.handover_cfg.approach_distance, params['obj_pos'][1] + params['y_offset_plus'], 1.05, -1, 0.3],  # 稍微抬高
            [params['obj_pos'][0] + params['x_offset_plus'] - self.handover_cfg.approach_distance, params['obj_pos'][1] + params['y_offset_plus'], 1.15, -1, 0.3],  # 继续抬高
        ]
        
        for action in grasp_and_lift_actions:
            for _ in self._execute_arm_action(self.robot.right_arm_hand, action):
                # print(f"action: {action}")
                yield()
    
    def task_reset_script_3(self):
        """阶段3：左手移动到物体附近准备接手"""
        params = self._get_handover_params()
        x_offset_plus = params['x_offset_plus']
        y_offset_plus = params['y_offset_plus']

        x_lh_plus = params['x_lh_plus']
        y_lh_plus = params['y_lh_plus']
        z_lh_plus = params['z_lh_plus']
        
        x_offset=0
        y_offset=0
        
        # handover动作序列
        handover_actions = [
                # handover
                [-0.21+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.35,       -1,  0.3] +\
                    [x_lh_plus, y_lh_plus, z_lh_plus + 0.5, 1,1  ],

                 [-0.15+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.35,          -1,  0.3] +\
                    [-0.05+x_lh_plus, y_lh_plus, z_lh_plus + 0.5, 1,1],
                    
                 [-0.15+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.35,          -1,  0.3] +\
                    [-0.1+x_lh_plus, y_lh_plus, z_lh_plus + 0.5, 1,1],
                 
                 [-0.15+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.35,          -1,  0.3] +\
                    [-0.15+x_lh_plus, y_lh_plus, z_lh_plus + 0.5, 1,1],
                                       
                [-0.15+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.35,        -1,  0.3] +\
                    [-0.25+x_lh_plus, y_lh_plus, z_lh_plus + 0.5, 1,1 ],  
                                
                [-0.15+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.35,      -1,  0.3] +\
                    [-0.35+x_lh_plus, y_lh_plus, z_lh_plus + 0.5, 1,1 ], 

                [-0.15+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.35,      -1,  0.3] +\
                    [-0.35+x_lh_plus, y_lh_plus, z_lh_plus + 0.5, -1,0.3 ], 
                
                # right-离开
                [-0.3+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.05, 1,  1] +\
                    [-0.25+x_lh_plus, y_lh_plus, z_lh_plus + 0.5, -1,0.3 ],   
                
                [-0.3+x_offset+x_offset_plus,-0.4+y_offset+y_offset_plus,1.05, 1,  1] +\
                    [x_lh_plus, y_lh_plus, z_lh_plus + 0.5 , -1,0.3  ], 

        ]
        
        for action in handover_actions:
            # 同时设置左右手动作
            right_action = action[:5]
            left_action = action[5:]
            
            # 使用封装的方法同时执行左右手动作
            for _ in self._execute_arm_action(
                self.robot.right_arm_hand, 
                right_action, 
                left_arm_hand=self.robot.left_arm_hand, 
                left_action=left_action
            ):
                yield()
    
    
    def cond_obj_in_which_hand(self):
        '''
        判断物体在哪只手上
        '''
        # 如果物体在左手上并且 物体高度 在 3-10 cm
        left_thumb_dist,left_index_dist,left_hand_center_dist,_,_,_ \
            = self.get_hand_to_position_dist_reward(self.obj_back_center,self.obj_front_center,left=True)
        right_thumb_dist,right_index_dist,right_hand_center_dist,_,_,_ \
            = self.get_hand_to_position_dist_reward(self.obj_back_center,self.obj_front_center,left=False)
        
        if left_thumb_dist < 0.1 and left_index_dist < 0.08 and left_hand_center_dist < 0.1 and left_thumb_dist < right_thumb_dist and left_index_dist < right_index_dist and left_hand_center_dist < right_hand_center_dist:
            return "left"
        elif right_thumb_dist < 0.1 and right_index_dist < 0.08 and right_hand_center_dist < 0.1 and right_thumb_dist < left_thumb_dist and right_index_dist < left_index_dist and right_hand_center_dist < left_hand_center_dist:
            return "right"
        else:
            return None  
        
    def get_task_stage(self):
        '''
        判别所属阶段
        '''
        # 物体在哪只手上
        hand_in_which_hand = self.cond_obj_in_which_hand()
        # 物体高度
        height,_ = self.get_height_reward(self.obj_init_pos[2] ,self.obj_right_center,self.obj_left_center)
        
        if hand_in_which_hand == None:
            return 0
        if hand_in_which_hand == "right" and height < self.cfg.target_obj_height_offset:
            return 1
        if hand_in_which_hand == "right" and height > self.cfg.target_obj_height_offset:
            return 2
        if hand_in_which_hand == "left" :
            return 3

        
    def get_reward(self):
        
        # 计算手到物体距离的奖励
        thumb_dist,index_dist,hand_center_dist,thumb_dist_reward,index_dist_reward,hand_center_dist_reward \
            = self.get_hand_to_position_dist_reward(self.obj_back_center,self.obj_front_center)
        
        # 物体提升高度的奖励
        initial_height = self.obj_init_pos[2]  
        height,height_reward = self.get_height_reward(initial_height,self.obj_right_center,self.obj_left_center)

        # 手指打开的奖励
        hand_open_dist,hand_open_dist_reward,hand_close_dist_reward = \
            self.get_hand_open_dist_reward()

        
         # 判断现在所属的阶段
        # print(f"task_stage: {self.task_stage}, step: {self.policy_steps}, hand_center_dist: {hand_center_dist}, index_dist: {index_dist}, thumb_dist: {thumb_dist}")
        task_stage = self.get_task_stage()
        # 阶段只会增加,不会减少
        if self.task_stage == task_stage:
            pass
                    
        if self.task_stage==1:
            hand_open_dist_reward = 1 # 直接给满
            thumb_dist_reward = 1
            index_dist_reward = 1
        else:
            self.task_stage = 0
            hand_close_dist_reward = 0 # 直接给0
            

        reward = 5 * hand_open_dist_reward + 5 * hand_close_dist_reward + \
                 3 * thumb_dist_reward + 3 * index_dist_reward + 2*hand_center_dist_reward +\
                20 * height_reward 
        # print(f"reward: {reward},task_stage: {self.task_stage}, step: {self.policy_steps}, hand_center_dist: {hand_center_dist}, index_dist: {index_dist}, thumb_dist: {thumb_dist}")
        return reward
    

    def last_return(self):
        observation = self.get_observation()
        reward = self.get_reward()
        self.infos = {"final_observation":[observation],
                      "final_info":[{"sim_episode":{
                        #   "r":self.accumulated_reward,
                          "r": reward,
                          "l":self.policy_steps
                      }}]}
        return True
    
    def get_truncated(self):
        if self.obj_pos_state[2] <= self.cfg.table_pos[2]:
            print("物体掉落")
            return self.last_return()
        if self.env_steps > self.cfg.max_env_step:
            return self.last_return()
        # 如果机器人手臂在机器人后面
        if self.mixed_pose[1] > 0.0:
            return self.last_return()
        # 如果物体倒了也失败
        if self.obj_pos_state[2] < self.obj_init_pos[2] - 0.005:
            return self.last_return()
        # if np.linalg.norm(self.obj_pos_state - self.obj_init_pos) > 1:
        #     print("物体被移动得太远了")
        #     return self.last_return()
        if self.obj_pos_state[0] > 0.25:
            print("物体太靠左了")
            return self.last_return()
        return False
    
    def get_terminated(self):        
        """
        当任务成功时返回 True
        """
        # if self.obj_pos_state[2] - self.obj_init_pos[2] >= self.cfg.target_obj_height_offset:
        #     return self.last_return()
        
        hand_in_which_hand = self.cond_obj_in_which_hand()
        height,height_reward = self.get_height_reward(self.obj_init_pos[2] ,self.obj_right_center,self.obj_left_center)
 
        if hand_in_which_hand == "left" and height > 0.03 and height < 0.1:
            return self.last_return()
        return False

