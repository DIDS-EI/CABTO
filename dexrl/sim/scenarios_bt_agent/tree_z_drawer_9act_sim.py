import numpy as np
# from dexrl.sim.scenarios.base_scenario import Scenario
from dexrl.sim.scenarios.rl_grasp_5act_sim import RLGraspScenario
from dexrl.sim.scenarios.rl_grasp_5act_sim import RLGraspScenarioCfg
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache

from dexrl.sim.scenarios._cfg import ScenarioCfg
from dexrl.utils import configclass
from dexrl.sim.objects import BoxOfCaneSugar,Milk, Tea, ThickCoconutMilk,Biscuit,Bowl,Cabinet
from dexrl.sim import utils
from omni.isaac.core.objects import VisualCuboid
from dexrl.sim.utils import Normalizer_N1_1
from scipy.spatial.transform import Rotation as R
from dexrl.utils import rot
from dexrl.sim.scenarios_bt_agent.tree_handover_9act_sim import TreeHandover9ActScenarioCfg,TreeHandover9ActScenario


@configclass
class TreeDrawer9ActScenarioCfg(TreeHandover9ActScenarioCfg):

    
    # default_joint_pos = np.array(
    #     [-0.10114183,0.0230558, 
    #      -1.61278892,-1.35585899 ,
    #      0.70649183,1.18980836,
    #      0.40805796,-0.42680282,
    #      -2.03365759, 0.0185703 ,
    #      -2.04454845,-1.82732479, 
    #      0.7979296,0.03886848,
    #      0.10,3.11,3.07,3.08,3.05,1.57,3.11,3.07,3.08,3.05,0.64,1.57,1.57,1.57,1.57,0.64,1.57,1.57,1.57,1.57,0.04,0.04]
    #     )
         
    default_joint_pos = np.array(
        [-0.36629102,0.0230558, 
         -0.9389022,-1.35585899 ,
         -1.5763891,1.18980836,
         -1.8762741,-0.42680282,
         -2.9864855, 0.0185703 ,
         0.35919958,-1.82732479, 
         3.1911237,0.03886848,
         3.11,3.11,
         3.07,3.08,
         3.05,1.57,
         3.11,3.07,
         3.08,3.05,
         0.64,1.57,
         1.57,1.57,
         1.57,0.64,
         1.57,1.57,
         1.57,1.57,
         1.57,0.04]
        )      
         
    table_pos = [0,-0.4,0.9]
    table_scale = [1.6,0.7,0.04]


    obj_cls = Tea  #Biscuit
    obj_init_pos =[0, -0.3777626037597656, 0.9939966797828674] #[-0.43382, -0.55641, 0.99748]
    obj_init_euler = [0, 1.57, -1.57]  #[0, 1.57, 0] 
    obj_mass = 0.2
    
    # obj_cls2 = Milk # Biscuit
    # obj_init_pos2 = [-0.54382, -0.5723, 0.996]
    # obj_init_euler2 = [1.57, 0, 0]  #[0, 1.57, 0] 
    # obj_mass2 = 0.2
    
    obj_cls2 = Biscuit
    obj_init_pos2 = [-0.54382, -0.5723, 0.996]
    obj_init_euler2 = [0, 1.57, 0] # z改为 -1.57/6 1.57/6
    # obj_delta_euler = 1.57/6
    obj_mass2 = 0.2 #0.2
    
    
    # obj_cls3 = Cabinet
    # # obj_init_pos3 = [0.2, -0.45, 0.96] 
    # obj_init_pos3 = [-0.91254, -0.36455, 0.93142] #[0.2, -0.45, 0.96]
    # # obj_init_euler3 = [0, 0, -1.1] 
    # obj_init_euler3 = [0, 0, 1.57] 
    # obj_mass3 = 3


    obj_cls3 = Cabinet
    # obj_init_pos3 = [0.2, -0.45, 0.96] 
    obj_init_pos3 = [0.5691729839339597, -0.7954696188769388, 0.9314200282096862] #[0.2, -0.45, 0.96]
    # obj_init_euler3 = [0, 0, -1.1] 
    obj_init_euler3 = [0, 0, -1.57] 
    obj_mass3 = 3
    
    
    # Task paper
    obj_cls4 = Bowl
    obj_init_pos4 = [0.21, -0.4, 0.96] #[0.2, -0.45, 0.96]
    obj_init_euler4 = [0, 3.14, 0] 
    obj_mass4 = 0.2


    # target_pad_pos = [0.04847, -0.40342, 0.9235406158103272]
    target_pad_pos = [-0.41159366280838505, -0.5880072354597021, 0.9235405921936037]
    # 活动范围 [-0.45- 0.1,  -0.25--0.6   ]
    target_pad_scale = [0.15, 0.15, 0.01]

class TreeDrawer9ActScenario(TreeHandover9ActScenario):
    cfg: TreeDrawer9ActScenarioCfg
    cfg_cls = TreeDrawer9ActScenarioCfg

    policy_steps = 0
    env_steps = 0
    accumulated_reward = 0
    obj_stable_steps=0

    # state buffers
    joint_pos_state = np.zeros(18)
    current_action_state = np.zeros(9)
    future_action_state = np.zeros(len(current_action_state))
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
    
    task_index = 4 # 当前任务索引
    task_start_stage = 2 # 当前任务开始阶段
    task_stage = task_start_stage # 当前任务阶段, 默认开始的任务阶段
    task_reset_script_stage = 0 # 当前任务重置脚本阶段
    task_reset_script = None
    task_stage_count = np.zeros(5)
    
    
    def __init__(self,cfg):
        super().__init__(cfg)
        # self.policy_action_normalizer = Normalizer_N1_1(self.cfg.policy_act_lower_limit, self.cfg.policy_act_upper_limit)
        self.right_policy_action_normalizer = Normalizer_N1_1(self.cfg.r_policy_act_lower_limit, self.cfg.r_policy_act_upper_limit)
        # self.left_policy_action_normalizer = Normalizer_N1_1(self.cfg.l_policy_act_lower_limit, self.cfg.l_policy_act_upper_limit)

        # self.target_pos = np.array([self.cfg.obj_init_pos2[0]-0.12,self.cfg.obj_init_pos2[1],self.cfg.obj_init_pos2[2]+0.1])
        # self.target_quat = rot.euler_angles_to_quat(\
        #     [self.cfg.obj_init_euler2[0],self.cfg.obj_init_euler2[1]+0.785,self.cfg.obj_init_euler2[2]])
        # self.target_quat = rot.euler_angles_to_quat(\
        #     [self.cfg.obj_init_euler2[0],self.cfg.obj_init_euler2[1]+2.355,self.cfg.obj_init_euler2[2]])
        
        self.target_pos = np.array([0.10173384845256805, -0.4291098713874817, 1.1290359497070312])
        self.target_quat = np.array([-0.0636, 0.88173, -0.28996, 0.36665])
        

    def load_objects(self):
        
        self.target_pad = FixedCuboid(
            name="target_pad",
            position=np.array([self.cfg.target_pad_pos]),
            size=1,
            scale=self.cfg.target_pad_scale,
            color=np.array([0, 1, 0]), # 绿色
            prim_path="/World/target_pad",
        )
        self.world.scene.add(self.target_pad)
        
        self.obj4: BaseObject = self.cfg.obj_cls4(
            position=self.cfg.obj_init_pos4,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler4),
            mass=self.cfg.obj_mass4,
            )
        self.obj_prim4 = self.obj4.create_prim()
        self.world.scene.add(self.obj_prim4)
        
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
        
        self.obj2: BaseObject = self.cfg.obj_cls2(
            position=self.cfg.obj_init_pos2,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler2),
            mass=self.cfg.obj_mass2,
            )
        self.obj_prim2 = self.obj2.create_prim()
        
        self.obj3: BaseObject = self.cfg.obj_cls3(
            position=self.cfg.obj_init_pos3,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler3),
            mass=self.cfg.obj_mass3,
            )
        self.obj_prim3 = self.obj3.create_prim()
        # 获取物体长宽高
        self.bbox_cache = create_bbox_cache()
        bounds = compute_aabb(self.bbox_cache, str(self.obj_prim.prim_path))
        self.obj_size = bounds[3:6] - bounds[0:3]  # 计算边界框的尺寸  长高(x) 横宽 (y) 厚款(z)
        # 排序 从大到小
        self.obj_size = np.sort(self.obj_size)[::-1]
        
        self.world.scene.add(self.obj_prim)
        self.world.scene.add(self.obj_prim2)
        self.task_obj_list.append(self.obj)
        self.task_obj_list.append(self.obj2)
        
        
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

    def task_reset_script_1(self,random_init=True):
        """阶段1: 让手在物体边上"""
        
        params = self._get_pickup_params()

        # action_list = [
        #     [params['obj_pos'][0] + params['x_offset_plus'] - 0.12, params['obj_pos'][1] + params['y_offset_plus'], -1, -1, 1], 
        # ]

        # 右手移动到物体附近
        action_list = [
            [params['obj_pos'][0] + params['x_offset_plus'] - 0.12, params['obj_pos'][1] + params['y_offset_plus'], 1, 1, 1],
            [params['obj_pos'][0] + params['x_offset_plus'] - 0.08, params['obj_pos'][1] + params['y_offset_plus'], 1, 1, 1],
            [params['obj_pos'][0] + params['x_offset_plus'] + params['approach_distance'], params['obj_pos'][1] + params['y_offset_plus'], 1.02, 1, 1],
            [params['obj_pos'][0] + params['x_offset_plus'] + params['approach_distance'] + 0.02, params['obj_pos'][1] + params['y_offset_plus'], \
                1.02, -1, 0.3],  # 关闭手    
        ]
        # print(f"params: {params}")
        # print(f"action_list: {action_list}")
        for action in action_list:
            for _ in self._execute_arm_action(self.robot.right_arm_hand, action):
                yield()

    def reset(self, stage=None,**kwargs):
        result = super().reset(stage,**kwargs)
        # 再 reset 碗
        self.obj2.set_world_pose(
            position=self.cfg.obj_init_pos2,
            orientation=utils.rot.euler_angles_to_quat(self.cfg.obj_init_euler2),
        )
        return result
    
        
    def task_reset_script_2(self,random_init=True):
        """阶段2:在阶段1基础上, 右手抓住物体并且抬高"""
        pk_params = self._get_pickup_params()
        # ho_params = self._get_handover_params()
        
        # 右手抓住物体并抬高
        grasp_and_lift_actions = [
            [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
                1.02, -1, 0.3],  # 关闭手
            # [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
            #     1.05, -1, 0.3],  # 稍微抬高
            # [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
            #     1.15, -1, 0.3],  # 1.15 继续抬高
            # [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
            #     1.02, -1, 0.3],  # 1.33 继续抬高
        ]
        
        # 动作序列最后一项, xyz 三个方向的扰动
        x_offset = np.random.uniform(-0.05, 0.05)
        # x_offset = 0
        y_offset = np.random.uniform(-0.05, 0.05)
        # y_offset = 0
        z_offset = np.random.uniform(-0.01, 0.1)
        lift_random_action = [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'] + x_offset, \
            pk_params['obj_pos'][1] + pk_params['y_offset_plus'] + y_offset, \
            1.02 + z_offset, \
            -1, 0.3]
        
        grasp_and_lift_actions.append(lift_random_action)
        
        
        for action in grasp_and_lift_actions:
            for _ in self._execute_arm_action(self.robot.right_arm_hand, action):
                yield()

    
    def _get_pour_params(self):
        """获取pour相关的参数"""
        return {
            'max_z': 1.35,
        }
    
        
    def get_task_stage(self):
        # 如果在 目标 位置上空半径 10 cm 范围内 并且高度 >=10 cm
        height,_ = self.get_height_reward(self.obj_init_pos[2],self.obj_right_center,self.obj_left_center)
        obj2_pos_state = self.cfg.obj_init_pos2
        dis = np.linalg.norm(self.obj_pos_state - obj2_pos_state)
        # print(f"height: {height}, dis: {dis}")
        if height >= 0.05 and dis <= 0.3:
            # print("task: pour")
            return self.task_index
        else:
            return 1
        
        
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
        if task_stage >= self.task_stage:
            self.task_stage = task_stage
        
        if self.task_stage == self.task_index:
            height_reward = 1
            
        # 获取物体quat 转为欧拉角度
        # obj_euler = rot.quat_to_euler(self.obj_quat_state)
        # print(f"obj_euler: {obj_euler}")
        
        if self.task_stage == self.task_index:
            # 此时左手 (0.19618141651153564, -0.34419456124305725, 1.2942111492156982)
            # 距离目标的奖励  pad\ bowl上方歇着 \ left, 包括物体倾斜度?
            _,_,target_pos_reward = self.get_target_pose_reward(self.target_pos,self.target_quat)
            distance_reward = 10* target_pos_reward
        else:
            distance_reward = 0
                       

        # 抓住物体的奖励
        grasp_reward = 5 * hand_close_dist_reward + 2 * thumb_dist_reward + 2 * index_dist_reward + 1 * hand_center_dist_reward
        # 抬高至少 10 cm 的奖励
        height_reward = 10 * height_reward
        
        reward = grasp_reward + height_reward + distance_reward
        
        # print(f"grasp_reward: {grasp_reward}, height_reward: {height_reward}, distance_reward: {distance_reward}")
        return reward   

    
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
        if self.cfg.env_args.get("collect_data",True):
            height,_ = self.get_height_reward(self.obj_init_pos[2],self.obj_right_center,self.obj_left_center)
            dis,angle,_ = self.get_target_pose_reward(self.target_pos,self.target_quat)
            # print(f"height: {height}, dis: {dis}, angle: {angle}")
            if height >= 0.05 and dis <= 0.06 and angle <= 0.5:
                return self.last_return()
        return False


