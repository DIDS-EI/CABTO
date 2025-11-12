import numpy as np
from dexrl.sim import utils
from dexrl import global_config
import os

########################################################## 
def collect_data(transition_data_list=[],file_name="demo_data.pkl"):
    
    import pickle as pkl
    
    # 保存 joint_pos_list
    floder_name = f"{global_config.root_path}/outputs/pkl/{file_name}"
    if not os.path.exists(floder_name):
        os.makedirs(floder_name)
    
    with open(f"{floder_name}/demo_data.pkl", "wb") as f:
        pkl.dump(transition_data_list, f)
    print(f"saved transition_data_list to {floder_name}/demo_data.pkl")
    
    # # 统计 action_agent_num 的数量 通过遍历 transition_data_list 数据
    # unique_task_stages = set()
    # for transition_data in transition_data_list:
    #     if "task_stage" in transition_data:
    #         unique_task_stages.add(transition_data["task_stage"])
    
    # # 如果没有 task_stage 字段，默认为 1
    # if not unique_task_stages:
    #     action_agent_num = 1
    #     print("Warning: No task_stage found in transition_data_list, using default action_agent_num = 1")
    # else:
    #     action_agent_num = max(unique_task_stages) + 1  # task_stage 从 0 开始，所以需要 +1
    #     print(f"Found {len(unique_task_stages)} unique task stages: {sorted(unique_task_stages)}")
    #     print(f"action_agent_num = {action_agent_num}")
    
    # # 保存 action 数据
    # action_transition_data = [[] for _ in range(action_agent_num)]
    
    # for transition_data in transition_data_list:
    #     if "task_stage" in transition_data:
    #         task_stage = transition_data["task_stage"]
    #         if task_stage < action_agent_num:
    #             action_transition_data[task_stage].append(transition_data)
    #         else:
    #             print(f"Warning: task_stage {task_stage} exceeds action_agent_num {action_agent_num}")
    #     else:
    #         # 如果没有 task_stage，放入第一个 agent
    #         action_transition_data[0].append(transition_data)
    
    # for i in range(action_agent_num):
    #     with open(f"{floder_name}/act_{i}.pkl", "wb") as f:
    #         pkl.dump(action_transition_data[i], f)
    #     print(f"saved action_transition_data to {floder_name}/act_{i}.pkl (count: {len(action_transition_data[i])})")
########################################################## 