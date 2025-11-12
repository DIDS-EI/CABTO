from re import T
import torch
import torch as th
from dexrl.utils import configclass
import numpy as np

@configclass
class ScenarioCfg:
    right_arm_ik_task_lower_limits = (0.0400, 1.7500, 1.7100, 1.7700, 1.7300)  # 手 5 个手指的第一个关节
    default_joint_pos = [-0.31,0.92,
                         -1.34,-1.34,
                         2.18,-1.05,
                         1.05,-1.26,
                         -3.07,1.48,
                         -1.67,-2.22,
                         0.93,-0.76,
                         0.10,3.11,3.07,3.08,3.05,1.57,3.11,3.07,3.08,3.05,0.64,1.57,1.57,1.57,1.57,0.64,1.57,1.57,1.57,1.57,0.04,0.04]
    
    # default_joint_pos = np.array(
    #     [-0.02, 0.45601963, 
    #      -1.51, -1.24047527 ,
    #      1.61, 0.12672836,
    #      1.63,-0.83182392,
    #      0.46, 1.05829787 ,
    #      -0.72,-1.51843645, 
    #      -2.36, -0.33150783,
    #      0.10,3.11,3.07,3.08,3.05,1.57,3.11,3.07,3.08,3.05,0.64,1.57,1.57,1.57,1.57,0.64,1.57,1.57,1.57,1.57,0.04,0.04]
    #     )
                          
    table_pos = [0.4,0,0.9]
    table_scale = [0.6,1.6,0.04]
    init_obj_scale = [0.6, 1, 1.4]

    init_obj_pos = [0.52, -0.268, 1.047]
    init_obj_euler = [0., 0, 1.57]
    # ik_lower_limit = [0.15,  -0.25,   1.0,   0.0,   0.0,  0.0] # 标准范围-更小范围
    # ik_upper_limit = [0.25,  -0.35,   1.15,  0.0,   0.0 , 0.0]

    # cys 修改为较大的抬起范围
    # 修改 reward_ik_limit
    ik_lower_limit = [0.15,  -0.25,   1.0,   -0.5,   -0.2,  -0.3] # 标准范围-更小范围
    ik_upper_limit = [0.35,  -0.35,   1.25,  0.5,   0.2 , 0.2]


    # controller_kps = 1e7
    # controller_kds = 2.5e6 #1e6 #2.5e6 # 越小移动步数越大
    # controller_kps = 1e7
    # controller_kds = 1e6 #2e6 #2.5e6 
    
    controller_kps = 2.5e7
    controller_kds = 2.5e6
    
    # controller_kps 比例增益决定了控制器对当前误差的响应强度。它与关节当前位置和目标位置之间的误差成正比。
    # 数值越大，控制器对位置偏差的纠正作用越强，系统反应更灵敏，但过大可能导致震荡。
    
    # controller_kds 阻尼增益用于抑制系统的震荡和过冲，类似于物理中的阻尼力。
    # 数值越大，系统运动越平稳，但过大会导致响应迟缓。
    # 它与关节运动的速度成正比，用于减缓运动速度。


    viewport_camera_pos_lookat = [2.47,-0.06,1.48, 1.56,0.26,1.20]

    enable_vision = True
    
    r_xyz_lower_limit = np.array([-0.8,       -0.8,    0.98]) 
    r_xyz_upper_limit = np.array([0.8,        -0.1,    1.5])

    l_xyz_lower_limit = np.array([-0.8,       -0.8,    0.98]) 
    l_xyz_upper_limit = np.array([0.8,         -0.1,   1.5])


    device = "cpu"
    # device = "cuda:0"