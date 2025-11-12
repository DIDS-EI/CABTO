
import os
from btgym.behavior_tree.behavior_trees.BehaviorTree import BehaviorTree
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btgym.algos.bt_planning.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str
import pathlib  
import time
DIR = pathlib.Path(__file__).parent



def validate_bt_fun(behavior_lib_path, goal_str,cur_cond_set,output_dir=None):
    ######################
    # 1. import behavior lib 导入行为库
    ######################
    # behavior_lib_path = os.path.join(DIR, '../exec_lib_demo/exec_lib')
    behavior_lib = ExecBehaviorLibrary(behavior_lib_path)


    ######################
    # 2. run planning algorithm, return planning success or not 运行规划算法，返回规划成功与否
    ######################
    # goal_str = "IsHolding(apple)"
    goal_set = goal_transfer_str(goal_str)

    # cur_cond_set = {'IsHandEmpty()'}

    algo = BTExpInterface(behavior_lib, cur_cond_set=cur_cond_set,
                        selected_algorithm="obtea")

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    act_num = -1
    expanded_num = -1
    planning_time_total = end_time - start_time

    time_limit_exceeded = algo.algo.time_limit_exceeded

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
        "\x1b[31mERROR\x1b[0m" if error else "",
        "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)


    ######################
    # 3. output algorithm result 输出算法结果
    ######################
    if output_dir!=None:
        file_path = os.path.join(output_dir)
        with open(file_path, 'w') as file:
            file.write(ptml_string)
        # read and execute
        bt = BehaviorTree(file_path, behavior_lib)
        bt.print()
        
        # 提取文件名（包括扩展名）
        file_name_with_extension = os.path.basename(output_dir)
        # 去掉扩展名
        file_name, _ = os.path.splitext(file_name_with_extension)
        bt.draw(target_directory=os.path.join(os.path.dirname(output_dir),""),file_name=file_name)
        
    return error, algo.algo.bt,expanded_num,act_num,record_act_ls,ptml_string
    '''测试 BT 的正确性'''


if __name__ == "__main__":      
    # behavior_lib_path = os.path.join(DIR, '../exec_lib_demo/exec_lib')
    behavior_lib_path = os.path.join(DIR, './exec_lib')
    goal_str = "IsHolding(apple)"
    cur_cond_set = {'IsHandEmpty()'}

    output_dir = os.path.join(DIR, './a_exp1_llm_bt/tree.btml')

    validate_bt_fun(behavior_lib_path=behavior_lib_path,goal_str=goal_str,cur_cond_set=cur_cond_set,output_dir=output_dir)
