import numpy as np
# from dexrl.sim.scenarios.base_scenario import Scenario
# from dexrl.sim.scenarios.rl_grasp_5act_sim import RLGraspScenario
# from dexrl.sim.scenarios.rl_grasp_5act_sim import RLGraspScenarioCfg
from omni.isaac.core.objects import FixedCuboid  # pyright: ignore[reportMissingImports]
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache  # pyright: ignore[reportMissingImports]

from dexrl.sim.scenarios._cfg import ScenarioCfg
from dexrl.utils import configclass
from dexrl.sim.objects import BoxOfCaneSugar, BaseObject, ThickCoconutMilk,Biscuit
from dexrl.sim import utils
from omni.isaac.core.objects import VisualCuboid  # pyright: ignore[reportMissingImports]
from dexrl.sim.utils import Normalizer_N1_1
from scipy.spatial.transform import Rotation as R
from dexrl.utils import rot
# from dexrl.sim.scenarios.tree_base_sim import TreeBaseScenarioCfg,TreeBaseScenario
# from dexrl.sim.scenarios.tree_9act_sim import Tree9ActScenarioCfg,Tree9ActScenario

from dexrl.sim.scenarios_tree.tree_pickup_gc_9act_sim import TreePickupGc9ActScenarioCfg,TreePickupGc9ActScenario

@configclass
class TreeHandover9ActScenarioCfg(TreePickupGc9ActScenarioCfg):

    
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


    obj_cls = Biscuit
    # table_pos[2] + table_scale[2]/2 + 176.328*Biscuit.scale_ratio_z/2
    obj_init_pos = [-0.15, -0.35, 0.997]
    # obj_init_pos = [-0.15, -0.4, table_pos[2] + table_scale[2]/2 + 176.328*Biscuit.scale_ratio_z/2]
    obj_init_euler = [0,0,0] #[0, 1.57, 0] # z改为 -1.57/6 1.57/6
    obj_init_quat = utils.rot.euler_angles_to_quat(np.array(obj_init_euler))
    # obj_delta_euler = 1.57/6
    obj_mass = 0.2 #0.2
 
    enable_right_hand = True
    enable_left_hand = False 

class TreeHandover9ActScenario(TreePickupGc9ActScenario):
    cfg: TreeHandover9ActScenarioCfg
    cfg_cls = TreeHandover9ActScenarioCfg

    
    task_index = 2 # 当前任务索引
    task_start_stage = 2 # 当前任务开始阶段
    task_stage = task_start_stage # 当前任务阶段, 默认开始的任务阶段
    task_reset_script_stage = 0 # 当前任务重置脚本阶段
    task_reset_script = None
    task_stage_count = np.zeros(5)
    
    
    def __init__(self,cfg):
        super().__init__(cfg)

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


  
    def task_reset_script_2(self,random_init=True):
        """阶段2:在阶段1基础上, 右手抓住物体并且抬高"""
        print("Handover: pickup task_reset_script_2")
        pk_params = self._get_pickup_params()
        # ho_params = self._get_handover_params()
        
        # 右手抓住物体并抬高
        grasp_and_lift_actions = [
            [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
                1.02, -1, 0.3],  # 关闭手
            # [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
            #     1.05, -1, 0.3],  # 稍微抬高
            # [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
            #     1.15, -1, 0.3],  # 继续抬高
            # [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'], pk_params['obj_pos'][1] + pk_params['y_offset_plus'], \
            #     1.2, -1, 0.3],  # .2 1.33 继续抬高
        ]
        
        
        # 动作序列最后一项, xyz 三个方向的扰动
        x_offset = np.random.uniform(-0.05, 0.05)
        # x_offset = 0
        y_offset = np.random.uniform(-0.05, 0.05)
        # y_offset = 0
        z_offset = np.random.uniform(0, 0.3)
        
        # z_offset = 0
        
        lift_random_action = [pk_params['obj_pos'][0] + pk_params['x_offset_plus'] + pk_params['approach_distance'] + x_offset, \
            pk_params['obj_pos'][1] + pk_params['y_offset_plus'] + y_offset, \
            1.02 + z_offset, \
            -1, 0.3]
        
        grasp_and_lift_actions.append(lift_random_action)
        
        
        for action in grasp_and_lift_actions:
            for _ in self._execute_arm_action(self.robot.right_arm_hand, action):
                yield()
    
    def _get_handover_params(self):
        """获取handover相关的参数"""
        return {
            'max_z': 1.35,
            'x_lh_plus': 0.4,
            'y_lh_plus': -0.34,
            'z_lh_plus': 0.77,
        }
    
    
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
        
        hands = [] 
        # print(f"left_thumb_dist: {left_thumb_dist}, left_index_dist: {left_index_dist}, left_hand_center_dist: {left_hand_center_dist}")
        if left_thumb_dist < 0.1 and left_index_dist < 0.08 and left_hand_center_dist < 0.1:
            hands.append("left")
        if right_thumb_dist < 0.1 and right_index_dist < 0.08 and right_hand_center_dist < 0.1:
            hands.append("right")
        
        # if left_thumb_dist < 0.1 and left_index_dist < 0.08 and left_hand_center_dist < 0.1 and left_thumb_dist < right_thumb_dist and left_index_dist < right_index_dist and left_hand_center_dist < right_hand_center_dist:
        #     return "left"
        # if right_thumb_dist < 0.1 and right_index_dist < 0.08 and right_hand_center_dist < 0.1 and right_thumb_dist < left_thumb_dist and right_index_dist < left_index_dist and right_hand_center_dist < left_hand_center_dist:
        #     return "right"
        return hands
   
    def reset(self, stage=None,**kwargs):
        random_init = self.cfg.env_args.get("random_init",True)
        
        self.policy_steps = 0
        self.accumulated_reward = 0
        self.obj_stable_steps = 0
        if stage is not None:
            self.task_start_stage = stage
        self.task_reset_script_stage = self.task_start_stage # 任务重置从第几步开始 1/2
        
        if self.task_start_stage == 2:
            self.task_stage  = self.task_index
        else:
            self.task_stage = 1
        # 0-1
        # 0-1-2
        # 0-1-3
        # 0-1-4
        
        self.update_state_buffers()
        self.robot.reset()
        
        if random_init: 
            # print("随机初始化")
            x_offset = np.random.uniform(-0.05, 0.05)
            y_offset = np.random.uniform(-0.05, 0.05)
            delta_euler = np.random.uniform(-1.57/30, 1.57/30) #90度/6
        else:
            x_offset = 0
            y_offset = 0
            delta_euler = 0
            
        self.obj_pos = self.cfg.obj_init_pos + np.array([x_offset, y_offset, 0])
        
        # 修复：正确计算欧拉角，只传递3个值
        self.final_euler = np.array(self.cfg.obj_init_euler) + np.array([0, 0, delta_euler])
        self.obj.set_world_pose(self.obj_pos, 
                                utils.rot.euler_angles_to_quat(self.final_euler))
        # obj_pose = self.obj.get_world_pose()
        # self.obj_pos_state[:] = obj_pose[0]
        # self.obj_quat_state[:] = obj_pose[1]
        
        # 根据阶段设置相应的任务重置脚本
        self.task_reset_script = self._get_stage_script(self.task_reset_script_stage )
        
        return x_offset,y_offset,self.obj.get_world_pose() 
        
    def get_task_stage(self):
        left_hand_h = 1.3
        
        # 在左手高度的上下 10cm 高度范围内
        # 0722
        if self.obj_pos_state[2] >= left_hand_h - 0.1 and self.obj_pos_state[2] <= left_hand_h + 0.1:
        
        # 左手切换区调整范围大一点？
        # 0727
        # if self.obj_pos_state[2] >= left_hand_h - 0.15 and self.obj_pos_state[2] <= left_hand_h + 0.15:
            return 2
        else:
            return 1
        
    def update_state_buffers(self):
        # self.goal_state = np.zeros(self.goal_len)
        # self.goal_mask =  np.zeros(self.goal_len)
        return super().update_state_buffers()
        
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
            # 此时左手 (0.19618141651153564, -0.34419456124305725, 1.2942111492156982)
            target_pos = np.array([((0.04627397283911705, -0.3909094035625458, 1.330304503440857))])
            # target_quat = np.array([-0.08102, 0.72921, 0.72921, 0.66845])
            target_quat = self.obj_init_quat
            _,_,target_pos_reward,target_quat_reward= self.get_target_pose_reward(target_pos,target_quat)
            distance_reward = 10* target_pos_reward + 10 * target_quat_reward
        else:
            distance_reward = 0
        
        # 抓住物体的奖励
        grasp_reward = 5 * hand_close_dist_reward + 2 * thumb_dist_reward + 2 * index_dist_reward + 1 * hand_center_dist_reward
        # 抬高至少 10 cm 的奖励
        height_reward = 10 * height_reward
        
        reward = grasp_reward + height_reward + distance_reward # max=20
        
        return reward
    

   
    def get_terminated(self):           
        # 到左手的距离小于一定值
        # print(f"hands: {hands}, height: {height}")
        if self.cfg.env_args.get("collect_data",True):
            hands = self.cond_obj_in_which_hand()
            height,_ = self.get_height_reward(self.obj_init_pos[2] ,self.obj_right_center,self.obj_left_center)
            if "left" in hands and height >= self.cfg.target_obj_height_offset:
                return self.last_return()
        return False
