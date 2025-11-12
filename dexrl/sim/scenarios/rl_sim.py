import numpy as np
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache

from dexrl.utils import configclass
from dexrl.sim.objects import BoxOfCaneSugar, BaseObject, ThickCoconutMilk
from dexrl.sim import utils
from omni.isaac.core.objects import VisualCuboid
from dexrl.sim.utils import Normalizer_N1_1
from scipy.spatial.transform import Rotation


from dexrl.sim.scenarios.base_scenario import *
from dexrl.sim.scenarios._cfg import ScenarioCfg



@configclass
class RLScenarioCfg(ScenarioCfg):
    default_joint_pos = np.array(
        [-0.31,0.92,-1.34,-1.34,2.18,-1.05,1.05,-1.26,-3.07,1.48,-1.67,-2.22,0.93,-0.76,0.10,3.11,3.07,3.08,3.05,1.57,3.11,3.07,3.08,3.05,0.64,1.57,1.57,1.57,1.57,0.64,1.57,1.57,1.57,1.57,0.04,0.04]
        )
    table_pos = [0,-0.4,0.9]
    table_scale = [1.6,0.7,0.04]


    obj_cls = ThickCoconutMilk
    obj_init_pos = [0, -0.35, table_pos[2] + table_scale[2]/2 + 85.6074*ThickCoconutMilk.scale_ratio/2]
    obj_init_euler = [1.57, 0, 0]
    obj_mass = 5 #0.2

    target_obj_height_offset = 0.14 # 0.15


    viewport_camera_pos_lookat = -0.14,-2.78,1.69,-0.17,-1.80,1.50

    enable_vision = False
    max_env_step = 300
    # policy_act_lower_limit = np.array([-0.45,-0.5,  1.02,     -0.5,  -0.5,  -0.5,    0,  0, 0.5]) 
    # 最后三位尝试直接赋值
    limit_range = 0.05

    policy_act_lower_limit = np.array([-limit_range,     -limit_range,   -limit_range,      -limit_range,   -limit_range,     -limit_range,       -1,   -1,  -1]) 
    policy_act_upper_limit = np.array([limit_range,       limit_range,    limit_range,       limit_range,    limit_range,       limit_range,          1,     1,   1])

    
    # 设置范围
    right_real_act_ik_lower_limit = np.array([-0.4,       -0.5,    1,         -3,    1.8,      -0.7,          -1,  -1,  -1]) 
    right_act_upper_limit = np.array([0.2,        -0.2,    1.36,      -0.5,    1.6,      0,            1,  1,       1])

    left_real_act_ik_lower_limit = np.array([-0.2,       -0.5,    1,         -3,    1.8,      -0.7,          -1,  -1,  -1]) 
    left_act_upper_limit = np.array([0.4,        -0.1,    1.3,        3,     2,      -0.5,            1,  1,  1])

    

    
class RLScenario(Scenario):
    cfg: RLScenarioCfg
    cfg_cls = RLScenarioCfg

    policy_steps = 0
    env_steps = 0
    accumulated_reward = 0
    obj_stable_steps=0

    # state buffers
    joint_pos_state = np.zeros(18)
    current_action_state = np.zeros(9)
    obj_pos_state = np.zeros(3)
    obj_quat_state = np.zeros(4)
    right_hand_view_pos = np.zeros((6,3))
    eef_pos = np.zeros(3)
    eef_quat = np.zeros(4)
    obj_init_pos = np.zeros(3)
    obj_init_quat = np.zeros(4)
    obj_pos_state = np.zeros(3)
    obj_quat_state = np.zeros(4)
    obj_rot_6d = np.zeros(6)  # 6维旋转表示
    obj_euler = np.zeros(3)
    right_hand_joint_pos = np.zeros(6)
    mixed_pose = np.zeros(6)
    left_mixed_pose = np.zeros(6)
    task_stage = 0

    def __init__(self,cfg:RLScenarioCfg):
        super().__init__(cfg)
        self.policy_action_normalizer = Normalizer_N1_1(self.cfg.policy_act_lower_limit, self.cfg.policy_act_upper_limit)

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
        
        self.vis_cube_obj_front = VisualCuboid(
            name="vis_cube_obj_front",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_obj_front",
            size=0.02,
            color=np.array([0, 1, 0]), # 绿色
        )
        self.vis_cube_obj_back = VisualCuboid(
            name="vis_cube_obj_back",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_obj_back",
            size=0.02,
            color=np.array([1, 0, 0]), # 红色
        ) 
        self.vis_cube_back = VisualCuboid(
            name="vis_cube_back",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_back",
            size=0.01,
            color=np.array([1, 0, 0]), # 红色
        ) 
        self.vis_cube_right = VisualCuboid(
            name="vis_cube_right",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_right",
            size=0.01,
            color=np.array([0, 0, 1]), # 蓝色
        )   
        self.vis_cube_left = VisualCuboid(
            name="vis_cube_left",
            position=np.array([(0.2878412468910885, -0.39393220715837446, 1.0488733532586387)]),
            prim_path="/World/vis_cube_left",
            size=0.01,
            color=np.array([0, 1, 1]), # 青色
        )   
        # self.thumb_eef_pos, self.index_eef_pos, self.hand_center = self.get_hand_eef_pos(self.mixed_pose)


    def reset(self):
        self.policy_steps = 0
        self.accumulated_reward = 0
        self.obj_stable_steps = 0
        self.task_stage = 0
        
        self.robot.reset()
        x_offset = np.random.uniform(-0.1, 0.1)
        y_offset = np.random.uniform(-0.1, 0.1)
        # x_offset = 0
        # y_offset = 0
        obj_pos = self.cfg.obj_init_pos + np.array([x_offset, y_offset, 0])
        self.obj.set_world_pose(obj_pos, 
                                utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler))
        return x_offset,y_offset,obj_pos
    
    def reset_and_set_obj_pos(self, obj_pos):
        self.obj.set_world_pose(obj_pos, utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler))
        x_offset = obj_pos[0] - self.cfg.obj_init_pos[0]
        y_offset = obj_pos[1] - self.cfg.obj_init_pos[1]
        return x_offset,y_offset,obj_pos


    def get_single_policy_real_action(self, action,arm_hand:ArmHand,lower_limit,upper_limit):
        last_mix_pose = arm_hand.arm.get_mixed_pose()
        last_hand_joint_3 = arm_hand.hand.get_hand_joint_policy_3()
        
        action = np.array(action)
        real_delta_action = self.policy_action_normalizer.denormalize(action)
        # print(action, real_delta_action)
        
        real_action = np.zeros(len(action))
        real_action[:6] = last_mix_pose + real_delta_action[:6] # 获取真实的 xyz
        real_action[6:] = last_hand_joint_3 + real_delta_action[6:] # 获取真实的 灵巧手关节位置
        real_action = np.clip(real_action, lower_limit, upper_limit)
        
        return real_action

    def set_single_policy_action(self, action,arm_hand:ArmHand,lower_limit,upper_limit):
        """
        设置策略动作
        输入：
            action: 9维, 6维机械臂关节位置, 3维手关节位置
        """

        real_action = self.get_single_policy_real_action(action,arm_hand,lower_limit,upper_limit)
        action_valid = arm_hand.set_real_ik_action_9(real_action)
        self.current_action_state[:] = action
        self.policy_steps += 1
        return action_valid,real_action


    def set_left_policy_action(self, action):
        return self.set_single_policy_action(action,self.robot.left_arm_hand,
                                             self.cfg.left_real_act_ik_lower_limit,
                                             self.cfg.left_act_upper_limit)
    
    def set_right_policy_action(self, action):
        return self.set_single_policy_action(action,self.robot.right_arm_hand,
                                             self.cfg.right_real_act_ik_lower_limit,
                                             self.cfg.right_act_upper_limit)


    def set_dual_policy_action(self, action):
        left_action = self.get_single_policy_real_action(action[:9],self.robot.left_arm_hand,
                                                         self.cfg.left_real_act_ik_lower_limit,
                                                         self.cfg.left_act_upper_limit)
        right_action = self.get_single_policy_real_action(action[9:],self.robot.right_arm_hand,
                                                         self.cfg.right_real_act_ik_lower_limit,
                                                         self.cfg.right_act_upper_limit)
        action = np.concatenate([left_action,right_action]) 
        action_valid = self.robot.dual_arm_hand.set_real_ik_action_18(action)
        return action_valid,action
    


    def get_rl_tuples(self):
        self.update_state_buffers()

        observation = self.get_observation()
        reward = self.get_reward()
        # self.accumulated_reward += reward
        terminated = self.get_terminated()
        truncated = self.get_truncated()
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
        self.mixed_pose[:] = self.robot.right_arm.get_mixed_pose() # 3+4
        self.obj_pos_state[:], self.obj_quat_state[:] = self.obj.get_world_pose() # 3+4
        self.right_hand_view_pos[:,:] = self.robot.right_arm_hand.hand.xform_view.get_world_poses()[0] # 手指中心
        
        # 获取 灵巧手 joint_pos
        self.right_hand_joint_pos[:] = self.robot.right_arm_hand.get_joint_positions()[7:]
        
        if self.policy_steps <=1:
            self.obj_init_pos[:], self.obj_init_quat[:] = self.obj.get_world_pose() # 3+4
            self.obj_euler[:] = self.obj.get_world_pose()[1]
    def get_observation(self):
        state = np.concatenate([
                        # self.eef_pos,self.eef_quat, # 3+4
                        self.mixed_pose, # 6
                        self.right_hand_joint_pos[[0,1,-1]], # 6
                        self.current_action_state, # 9
                        # self.right_hand_view_pos[0],手指中心位置
                        self.obj_pos_state, # 3
                        self.obj_quat_state, # 4
                        np.array([self.task_stage]), # 1    
                        ])
        # obs_dict = {
        #     # "image": self.get_camera_rgb(),
        #     "state": state
        # }
        # return obs_dict
        return state

    def get_obj_centers(self, object_position, current_quat, init_quat):
        """
        计算物体前后中心点
        Args:
            object_position: 物体当前位置
            current_quat: 物体当前四元数
            init_quat: 物体初始四元数
        Returns:
            obj_back_center: 物体后方中心点
            obj_front_center: 物体前方中心点
        """
        # 计算物体姿态变化
        # rel_rot = Rotation.from_quat(current_quat) * Rotation.from_quat(init_quat).inv()
        
        
        rot_matrix = utils.rot.quat_to_rot_matrix(current_quat)
        lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([0,0,1]))
        x_lookat_direction = utils.rot.rot_matrix_to_lookat_direction(rot_matrix,forward_direction=np.array([1,0,0]))
        front_pos = object_position + lookat_direction * self.obj_size[2]/2
        back_pos = object_position - lookat_direction * self.obj_size[2]/2

        return front_pos,back_pos ,None,None

    def get_reward(self):
        object_position = self.obj_pos_state
        initial_bottom_height = self.cfg.table_pos[2] + self.cfg.table_scale[2]/2 # self.obj_init_pos[2] - self.obj_size[0]/2
        initial_top_height = self.obj_init_pos[2] + self.obj_size[0]/2
        # target_height = initial_bottom_height + self.cfg.target_obj_height_offset
        

        reward = 0
        
        # 计算手中心到物体中心的距离
        hand_center_dist = np.linalg.norm(
            self.right_hand_view_pos[0] - object_position)
        hand_center_dist_reward = - np.exp(3 * hand_center_dist) + 2
        hand_center_dist_reward = np.clip(hand_center_dist_reward, 0.0, 1.0)
        
        # 获取物体前后中心点 # 获取物体的左右点
        obj_front_center, obj_back_center,obj_left_center,obj_right_center = self.get_obj_centers(
            object_position, self.obj_quat_state, self.obj_init_quat)
        
        # self.vis_cube_front.set_world_pose(obj_front_center, self.obj_quat_state)
        # self.vis_cube_back.set_world_pose(obj_back_center, self.obj_quat_state)
        
        # 大拇指到物体后方的中心点的距离
        thumb_dist = np.linalg.norm(self.right_hand_view_pos[1] - obj_back_center)
        # 无名指到物体前方的中心点的距离
        index_dist = np.linalg.norm(self.right_hand_view_pos[4] - obj_front_center)
        # 大拇指到物体后方的中心点的距离奖励
        thumb_dist_reward = - np.exp(3 * thumb_dist) + 2
        # 无名指到物体前方的中心点的距离奖励
        index_dist_reward = - np.exp(2 * index_dist) + 2
        thumb_dist_reward = np.clip(thumb_dist_reward, 0.0, 1.0)
        index_dist_reward = np.clip(index_dist_reward, 0.0, 1.0)
        
        # 计算物体提升高度的奖励
        # 获取物体长宽高
        self.bbox_cache = create_bbox_cache()
        bounds = compute_aabb(self.bbox_cache, str(self.obj_prim.prim_path))
        obj_z_bottom = bounds[2]
        obj_z_top = bounds[5]
        height_reward = min(
            (obj_z_bottom - initial_bottom_height) / self.cfg.target_obj_height_offset,
            (obj_z_top - initial_top_height) / self.cfg.target_obj_height_offset
        )
        height_reward = np.clip(height_reward, 0.0, 1.0)
        # print(height_reward)

        # 手指打开的奖励 大拇指到无名指的距离
        hand_open_dist = np.linalg.norm(self.right_hand_view_pos[1] - self.right_hand_view_pos[4])
        # 0.07 是关上，0.12是打开 ; 映射到 [0.27  0.62]
        hand_open_dist_reward = (hand_open_dist - 0.07) / (0.13 - 0.07)
        hand_open_dist_reward = np.clip(hand_open_dist_reward, 0, 1)
        
        hand_close_dist_reward = (0.12-hand_open_dist) / (0.12 - 0.07)
        hand_close_dist_reward = np.clip(hand_close_dist_reward, 0, 1)
        
         # 判断现在所属的阶段
        if hand_center_dist < 0.1 and index_dist < 0.08 and thumb_dist < 0.08:
            self.task_stage = 1
            hand_open_dist_reward = 1 # 直接给满
        else:
            self.task_stage = 0
            hand_close_dist_reward = 0 # 直接给0
            
        # reward = 5 * hand_open_dist_reward + 5 * hand_close_dist_reward + \
        #          3 * hand_center_dist_reward +  2* thumb_dist_reward + index_dist_reward + \
        #         10 * height_reward 

        reward = 2 * hand_open_dist_reward + 5 * hand_close_dist_reward + \
                 3 * thumb_dist_reward + 2 * index_dist_reward + \
                10 * height_reward 
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
        #     self.obj_stable_steps += 1
        # else:
        #     self.obj_stable_steps = 0
        # if self.obj_stable_steps >= 10:
        #     return self.last_return()
        
        # if self.obj_pos_state[2] - self.obj_init_pos[2] >= self.cfg.target_obj_height_offset:
        #     return self.last_return()
        
        return False

