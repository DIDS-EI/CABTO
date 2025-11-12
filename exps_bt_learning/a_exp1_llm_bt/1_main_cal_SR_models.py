import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from datetime import datetime
from exps_bt_learning.tools import parse_bddl
from exps_bt_learning.llm_generate_lib_func import llm_generate_behavior_lib
from exps_bt_learning.validate_bt_fun import validate_bt_fun

task2name = {
    "task1":"Cover",
    "task2":"Blocks",
    "task3":"Clean",
}
task2objects = {
    "task1":['a','place_a','b','place_b','c','place_c'],
    "task2":['a','b','c','d'],
    "task3":['left_franka','right_franka','left_lego1','left_lego2','left_lego3','right_lego1','right_lego2','right_lego3'],
}
task2start_state = {
    "task1":{'IsHandEmpty()','IsEmpty(place_a)','IsEmpty(place_b)','IsEmpty(place_c)','On(a,table)','On(b,table)','On(c,table)'},
    "task2":{'IsHandEmpty()','On(a,table)','On(b,table)','On(c,table)','On(d,table)'},
}
task2goal_str = {
    "task1":['On(a,place_a)','On(a,place_a) & On(b,place_b)','On(a,place_a) & On(b,place_b) & On(c,place_c)'],
    "task2":['On(a,b)','On(a,b) & On(b,c)','On(a,b) & On(b,c) & On(c,d)'],
}

total_try_times = 2  # 每个任务跑5次

model = "gpt-4o"

# 获取当前日期
current_date = datetime.now().strftime("%Y%m%d")

# 创建结果目录
result_dir = os.path.join(DIR, "a_exp1_llm_bt_results")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 存储所有任务的结果
all_results = []

for task_id in range(1, 3):  # task1 和 task2
    # 1. set task
    task_name = f"task{task_id}"
    
    behavior_lib_path = os.path.join(DIR, f"a_exp1_llm_bt_tasks/{task_name}/exec_lib")
    
    start_state = task2start_state[task_name].copy()
    objects = set(task2objects[task_name])
    goal_str_list = task2goal_str[task_name]  # [easy, medium, hard]
    
    print("="*60)
    print(f"Task: {task_name}")
    print("objects:", objects)
    print("start_state:", start_state)
    print("goal_str_list (easy, medium, hard):", goal_str_list)
    print("="*60)

    # 为每次尝试创建目录保存输入输出
    for try_idx in range(total_try_times):
        print(f"\n{'='*60}")
        print(f"Task: {task_name}, Try {try_idx + 1}/{total_try_times}")
        print(f"{'='*60}\n")
        
        # 创建保存输入输出的目录
        save_io_dir = os.path.join(result_dir, f"{task_name}_try_{try_idx + 1}_{current_date}")
        os.makedirs(save_io_dir, exist_ok=True)
        
        # 1. 生成行为库（传入三个goal）
        print("Generating behavior lib with all three goals...")
        try:
            llm_generate_behavior_lib(
                goal_str_list=goal_str_list,
                objects=objects,
                start_state=start_state,
                behavior_lib_path=behavior_lib_path,
                model=model,
                save_io_dir=save_io_dir
            )
        except Exception as e:
            print(f"Error generating behavior lib: {e}")
            # 如果生成失败，记录所有goal都失败
            for difficulty, goal_str in zip(['easy', 'medium', 'hard'], goal_str_list):
                all_results.append({
                    'task_name': task_name,
                    'try_idx': try_idx + 1,
                    'difficulty': difficulty,
                    'goal': goal_str,
                    'success': False,
                    'error': str(e)
                })
            continue
        
        # 2. 分别对每个goal进行验证
        difficulties = ['easy', 'medium', 'hard']
        all_goals_success = True
        
        for difficulty, goal_str in zip(difficulties, goal_str_list):
            print(f"\n--- Validating {difficulty} goal: {goal_str} ---")
            
            # 为每个goal创建单独的输出目录
            output_dir = os.path.join(save_io_dir, f"bt_{difficulty}.btml")
            
            try:
                error, bt, expanded_num, act_num, record_act_ls, ptml_string = validate_bt_fun(
                    behavior_lib_path=behavior_lib_path,
                    goal_str=goal_str,
                    cur_cond_set=start_state,
                    output_dir=output_dir
                )
                
                success = (error == 0)
                if not success:
                    all_goals_success = False
                
                print(f"{difficulty} goal result: {'SUCCESS' if success else 'FAILED'}")
                
                # 记录结果
                all_results.append({
                    'task_name': task_name,
                    'try_idx': try_idx + 1,
                    'difficulty': difficulty,
                    'goal': goal_str,
                    'success': success,
                    'expanded_num': expanded_num,
                    'act_num': act_num,
                    'error': None
                })
                
            except Exception as e:
                all_goals_success = False
                print(f"Error validating {difficulty} goal: {e}")
                
                # 记录失败结果
                all_results.append({
                    'task_name': task_name,
                    'try_idx': try_idx + 1,
                    'difficulty': difficulty,
                    'goal': goal_str,
                    'success': False,
                    'expanded_num': -1,
                    'act_num': -1,
                    'error': str(e)
                })
        
        # 统计这次尝试三个goal是否都成功
        print(f"\n{'='*60}")
        print(f"Task {task_name}, Try {try_idx + 1}: All goals success = {all_goals_success}")
        print(f"{'='*60}\n")
        
        # 输出生成的action lib和condition lib数量
        try:
            action_lib_num = len([f for f in os.listdir(os.path.join(behavior_lib_path, 'Action')) if f.endswith('.py')])
            condition_lib_num = len([f for f in os.listdir(os.path.join(behavior_lib_path, 'Condition')) if f.endswith('.py')])
            print(f"Action lib num: {action_lib_num}")
            print(f"Condition lib num: {condition_lib_num}")
        except Exception as e:
            print(f"Error counting lib files: {e}")

# 保存结果到CSV
df = pd.DataFrame(all_results)
csv_filename = os.path.join(result_dir, f"exp1_results_{current_date}.csv")
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"\n{'='*60}")
print(f"Results saved to: {csv_filename}")
print(f"{'='*60}\n")

# 统计结果
print("\n=== 统计结果 ===")
for task_name in ["task1", "task2"]:
    task_df = df[df['task_name'] == task_name]
    if len(task_df) == 0:
        continue
    
    print(f"\n{task_name}:")
    for difficulty in ['easy', 'medium', 'hard']:
        diff_df = task_df[task_df['difficulty'] == difficulty]
        if len(diff_df) > 0:
            success_count = diff_df['success'].sum()
            total_count = len(diff_df)
            success_rate = success_count / total_count * 100
            print(f"  {difficulty}: {success_count}/{total_count} ({success_rate:.2f}%)")
    
    # 统计所有goal都成功的次数
    all_success_count = 0
    for try_idx in range(1, total_try_times + 1):
        try_df = task_df[task_df['try_idx'] == try_idx]
        if len(try_df) == 3 and try_df['success'].all():
            all_success_count += 1
    print(f"  所有goal都成功: {all_success_count}/{total_try_times} ({all_success_count/total_try_times*100:.2f}%)")

print("\n完成！")
