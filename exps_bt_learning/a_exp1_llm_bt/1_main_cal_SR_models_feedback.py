import os
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from datetime import datetime
from exps_bt_learning.tools import parse_bddl
from exps_bt_learning.llm_generate_lib_func import llm_generate_behavior_lib,llm_generate_behavior_lib_need_feedback
from exps_bt_learning.validate_bt_fun import validate_bt_fun
from btgym.llm.llm_gpt import LLM
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
import traceback

'''
大模型直接输出的大概率有问题，如果教它，如何设计反馈？
有哪些问题，可以做一下总结
根据这些 insight 设计反馈

1. 要说明白规划的是 goal  是干吗用的，不是让它一步完成的
增加了互斥的删除效果
        # Remove all On(arg[0], other locations) states
        all_locations = {args[1] for args in cls.valid_args}
        info["del_set"] = {f"On({arg[0]},{loc})" for loc in all_locations if loc != arg[1]}
后面都能答对
'''

def check_syntax_errors(behavior_lib_path):
    """
    检查行为库是否有语法错误
    返回: (has_error, error_message)
    """
    try:
        behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
        return False, None
    except SyntaxError as e:
        return True, f"SyntaxError: {str(e)}"
    except Exception as e:
        # 检查是否是导入错误（可能是语法错误导致的）
        error_str = str(e)
        if "SyntaxError" in error_str or "invalid syntax" in error_str.lower():
            return True, f"SyntaxError: {error_str}"
        # 其他错误（如导入错误）也视为语法相关错误
        if "ImportError" in error_str or "ModuleNotFoundError" in error_str:
            return True, f"ImportError: {error_str}"
        # 其他异常也返回，但标记为语法错误以便重试
        return True, f"Error: {error_str}"

def generate_behavior_lib_with_syntax_retry(goal_str_list, objects, initial_state, behavior_lib_path, 
                                            llm, messages, clear_lib, save_io_dir, max_syntax_retries=3):
    """
    生成行为库，如果出现语法错误则自动重试（最多max_syntax_retries次）
    这个重试不计入常规的尝试次数和反馈次数
    """
    for syntax_retry in range(max_syntax_retries):
        try:
            # 生成行为库
            messages = llm_generate_behavior_lib_need_feedback(
                goal_str_list=goal_str_list,
                objects=objects,
                initial_state=initial_state,
                behavior_lib_path=behavior_lib_path,
                llm=llm,
                messages=messages,
                clear_lib=clear_lib,
                save_io_dir=save_io_dir
            )
            
            # 检查语法错误
            has_error, error_message = check_syntax_errors(behavior_lib_path)
            if not has_error:
                # 没有语法错误，返回
                if syntax_retry > 0:
                    print(f"\033[92m✓ Successfully generated behavior library after {syntax_retry + 1} attempts (syntax retries)\033[0m")
                return messages
            else:
                # 有语法错误，继续重试
                print(f"\033[93m⚠ Syntax error detected (attempt {syntax_retry + 1}/{max_syntax_retries}): {error_message}\033[0m")
                print(f"\033[93m  Retrying generation...\033[0m")
                # 添加语法错误反馈到消息历史
                syntax_error_prompt = f"""The generated behavior library contains syntax errors and cannot be imported. Error: {error_message}

Please regenerate the behavior library with correct Python syntax. Ensure:
1. All Python code follows correct syntax rules
2. All class definitions are complete
3. All required attributes are properly defined
4. No syntax errors in the code

Please provide the complete updated Python code for the entire behavior library directly, without additional explanations."""
                messages.append({"role": "user", "content": syntax_error_prompt})
                
        except Exception as e:
            # 生成过程中出现异常
            error_str = str(e)
            print(f"\033[93m⚠ Error during generation (attempt {syntax_retry + 1}/{max_syntax_retries}): {error_str}\033[0m")
            if syntax_retry < max_syntax_retries - 1:
                print(f"\033[93m  Retrying generation...\033[0m")
                # 添加错误反馈到消息历史
                error_prompt = f"""An error occurred during behavior library generation: {error_str}

Please regenerate the behavior library. Ensure all code is syntactically correct and complete.

Please provide the complete updated Python code for the entire behavior library directly, without additional explanations."""
                if messages is None:
                    messages = []
                messages.append({"role": "user", "content": error_prompt})
            else:
                # 最后一次重试也失败了，抛出异常
                raise
    
    # 所有重试都失败了
    raise Exception(f"Failed to generate behavior library without syntax errors after {max_syntax_retries} attempts")

def validate_all_goals(behavior_lib_path, goal_str_list, initial_state, save_io_dir, difficulties=['easy', 'medium', 'hard']):
    """
    验证所有goal，返回验证结果列表
    
    Returns:
        results: list of dict, 每个dict包含验证结果
        all_success: bool, 是否所有goal都成功
    """
    results = []
    all_success = True
    
    for difficulty, goal_str in zip(difficulties, goal_str_list):
        print(f"\n--- Validating {difficulty} goal: {goal_str} ---")
        
        # 为每个goal创建单独的输出目录
        output_dir = os.path.join(save_io_dir, f"bt_{difficulty}.btml")
        
        try:
            error, bt, expanded_num, act_num, record_act_ls, ptml_string = validate_bt_fun(
                behavior_lib_path=behavior_lib_path,
                goal_str=goal_str,
                cur_cond_set=initial_state,
                output_dir=output_dir
            )
            
            success = (error == 0)
            if not success:
                all_success = False
            
            print(f"{difficulty} goal result: {'SUCCESS' if success else 'FAILED'}")
            
            # 记录结果
            result = {
                'difficulty': difficulty,
                'goal': goal_str,
                'success': success,
                'expanded_num': expanded_num,
                'act_num': act_num,
                'error': None,
                'ptml_string': ptml_string,
                'record_act_ls': record_act_ls
            }
            results.append(result)
            
        except Exception as e:
            all_success = False
            print(f"Error validating {difficulty} goal: {e}")
            
            # 记录失败结果
            result = {
                'difficulty': difficulty,
                'goal': goal_str,
                'success': False,
                'expanded_num': -1,
                'act_num': -1,
                'error': str(e),
                'ptml_string': None,
                'record_act_ls': None
            }
            results.append(result)
    
    return results, all_success

def get_lib_counts(behavior_lib_path):
    """获取行为库中Action和Condition的数量"""
    try:
        action_lib_num = len([f for f in os.listdir(os.path.join(behavior_lib_path, 'Action')) if f.endswith('.py')])
        condition_lib_num = len([f for f in os.listdir(os.path.join(behavior_lib_path, 'Condition')) if f.endswith('.py')])
        return action_lib_num, condition_lib_num
    except Exception as e:
        print(f"Error counting lib files: {e}")
        return 0, 0

def build_feedback_prompt(failed_results, goal_str_list, initial_state, difficulties=['easy', 'medium', 'hard']):
    """
    构建反馈prompt
    
    Args:
        failed_results: list of dict, 失败的验证结果
        goal_str_list: list, 所有goal列表
        initial_state: set, 初始状态
        difficulties: list, 难度列表
    """
    feedback_prompt = "The current behavior library fails to generate valid behavior trees that can achieve the specified goals, or encounters errors during the planning process. The following are the failed cases:\n\n"
    
    for result in failed_results:
        difficulty = result['difficulty']
        goal_str = result['goal']
        ptml_string = result.get('ptml_string', None)
        expanded_num = result.get('expanded_num', -1)
        act_num = result.get('act_num', -1)
        record_act_ls = result.get('record_act_ls', None)
        
        feedback_prompt += f"[Failed Case - {difficulty.capitalize()} Goal]\n"
        feedback_prompt += f"Initial State: {initial_state}\n"
        feedback_prompt += f"Target Goal: {goal_str}\n"
        
        if ptml_string is not None:
            # Limit btml to maximum 50 lines
            btml_lines = ptml_string.split('\n')
            if len(btml_lines) > 50:
                btml_display = '\n'.join(btml_lines[:50]) + f"\n... (Total {len(btml_lines)} lines, showing first 50 lines only)"
            else:
                btml_display = ptml_string
            
            feedback_prompt += f"Generated Behavior Tree (BTML):\n{btml_display}\n"
            feedback_prompt += f"Number of nodes expanded during planning: {expanded_num}\n"
            if record_act_ls is not None:
                feedback_prompt += f"Action sequence executed: {record_act_ls}\n"
        else:
            error = result.get('error', 'Unknown error')
            feedback_prompt += f"Failed to generate behavior tree. Error: {error}\n"
            # If the error message contains critical information, emphasize it
            if 'valid_args' in str(error) or 'attribute' in str(error) or 'Error' in str(error):
                feedback_prompt += f"IMPORTANT: This error indicates that the behavior library has structural issues. "
                feedback_prompt += f"Please ensure all action and condition classes have the required attributes (valid_args, num_args, can_be_expanded, etc.) properly defined.\n"
        
        feedback_prompt += "\n"
    
    feedback_prompt += """
Please analyze the failed cases above and the behavior trees generated by the current behavior library, which are described in BTML (Behavior Tree Markup Language). Key concepts:

1. **Behavior Trees (BTML)** serve as the planning blueprint for finding an **executable path** from the **Initial State** to the **Goal**.
2. A **`selector`** node succeeds upon the first successful child; a **`sequence`** node requires all children to succeed sequentially.
3. Planning fails when, starting from the Initial State, the tree **lacks any traversable path** where all subsequent conditions (`cond`) evaluate to true, thus failing to reach an executable action (`act`).

**IMPORTANT**: You must regenerate the ENTIRE behavior library from scratch. The behavior library must be capable of successfully generating behavior trees that can achieve all the specified goals. All previous code will be cleared.

**Requirements for the regenerated behavior library:**

1. **Completeness**: Generate all action and condition nodes necessary to achieve the goals.

2. **Structural Correctness**: 
   - Ensure all classes have the correct structure with all required attributes (valid_args, num_args, can_be_expanded, etc.).
   - Make sure valid_args is properly defined for each action/condition class.
   - Avoid any structural errors that would prevent the library from loading.

3. **Action Design Principles**:
   - **Actor-Object Distinction**: Clearly distinguish between actors (robots/agents) and objects. Actors are the entities that perform actions (e.g., left_franka, right_franka), while objects are the targets being manipulated (e.g., items, containers, locations). If there are multiple actors, define corresponding actions for each actor separately.
   - **State Consistency**: The previous action's `add` or `del` effects must fully satisfy the next action's required `pre` preconditions, both structurally and logically. State predicates must not contain structural ambiguities.
   - **Initial State Executability**: At least one action must be executable in the initial state (its preconditions must be satisfied by the initial state).
   - **Logical Consistency**: Avoid logically contradictory condition combinations.

4. **Planning Feasibility**: The behavior library must facilitate successful planning by ensuring that a valid execution path can always be initiated from the initial state.

Please provide the complete updated Python code for the entire behavior library directly, without additional explanations.

"""
    
    return feedback_prompt


task2name = {
    "task1":"Cover",
    "task2":"Blocks",
    
    "task3":"Clean",
    "task4":"Handover",
    "task5":"Pour",  # 左边牛奶倒入右边杯子，右边牛奶倒入左边杯子

    "task6":"HomeRearrangement",  # 涉及到 开关铰链抽屉 和 放置物品
    "task7":"MealPreparation",    # 涉及到 开关电器门 和 启动电器 和 开关灯
}
task2objects = {
    "task1":['a','place_a','b','place_b','c','place_c'],
    "task2":['a','b','c','d'],

    "task3":['left_franka','right_franka','left_lego','right_lego',"center_big_box","box_board","left_table","right_table","center_table"],

    # myx
    "task4":['left_franka','right_franka','left_box1','left_box2','right_box',"left_table","right_table"],

    "task5":['left_franka','right_franka','left_cup','right_cup','right_milk','right_milk','left_table','right_table'], # 杯子需要手扶住

    "task6":['fridge','pen','cabinet',"book","banana","table","breakfast_table"],  # 冰箱和抽屉初始是关闭的，需要打开才能放入物品，分别把苹果放进冰箱，pen放进抽屉
    "task7":['oven','chickenleg','soup','microwave','light',"radio",'table',"pie","breakfast_table"], # 烤箱、微波炉需要启动，需要开灯

}
task2initial_state = {
    "task1":{'IsHandEmpty()','IsEmpty(place_a)','IsEmpty(place_b)','IsEmpty(place_c)','On(a,table)','On(b,table)','On(c,table)'},
    "task2":{'IsHandEmpty()','On(a,table)','On(b,table)','On(c,table)','On(d,table)'},

    "task3":{'IsHandEmpty(left_franka)','IsHandEmpty(right_franka)',\
        'On(left_franka,left_table)','On(right_franka,right_table)',\
        'On(left_lego,left_table)','On(right_lego,right_table)',\
            'On(center_big_box,center_table)','On(box_board,center_table)',"BigBoxNeedTwoFrankaHoldTogether()"},
    
    "task4":{'IsHandEmpty(left_franka)','IsHandEmpty(right_franka)',\
        'On(left_franka,left_table)','On(right_franka,right_table)',\
        'On(left_box1,left_table)','On(left_box2,left_table)','On(right_box,right_table)'},

    "task5":{
            'IsHandEmpty(left_franka)', 'IsHandEmpty(right_franka)',
            # 'On(left_franka,left_table)', 'On(right_franka,right_table)',
            'On(left_cup,left_table)', 'On(right_cup,right_table)',
            'On(left_milk,left_table)', 'On(right_milk,right_table)',
            
            'IsFull(left_milk)', 'IsFull(right_milk)',# 牛奶容器是满的
            'IsEmpty(left_cup)',  'IsEmpty(right_cup)', # 杯子是空的

            'CupNeedGraspAndSupport()', # 杯子需要支撑
            'CanGrasp(left_franka,left_milk)', 'CanGrasp(right_franka,right_milk)'}, # 左手拿左边

    "task6":{'IsHandEmpty()', 'On(pen,breakfast_table)','On(book,table)','On(banana,table)',\
            'IsClosed(fridge)', 'IsClosed(cabinet)', },

    "task7":{'IsHandEmpty()', 'On(chickenleg,table)', 'On(soup,table)', \
        'IsSwitchedOff(oven)', 'IsSwitchedOff(microwave)', 'IsSwitchedOff(light)',"IsSwitchedOff(radio)",'IsClosed(oven)', 'IsClosed(microwave)','On(pie,table)'},
    }
task2goal_str = {
    "task1":['On(a,place_a)','On(b,place_b) & On(c,place_c)','On(b,place_a) & On(a,place_c) & On(c,place_b)'], #'On(b,place_a) & On(a,place_b) & On(c,place_c)'
    # "task2":['On(a,b)','On(a,b) & On(b,c)','On(c,b) & On(b,a) & On(a,d)'],
    "task2":['On(a,b)','On(c,b) & On(b,a)','On(a,c) & On(c,b) & On(b,d)'], #'On(a,c) & On(c,b) & On(b,d)'

    # "task3":['In(left_lego,center_big_box)','In(right_lego,center_big_box)&In(left_lego,center_big_box)','On(center_big_box,box_board)'] # 前两个可以被完成,
    # "task3":['On(center_big_box,box_board)','On(center_big_box,box_board)','On(center_big_box,box_board)'], # 可以完成
    # "task3":['On(center_big_box,box_board)','On(center_big_box,box_board)&On(left_lego,center_big_box)','On(center_big_box,box_board)&On(left_lego,center_big_box)&On(right_lego,center_big_box)'],
    "task3":['In(left_lego,center_big_box)','In(right_lego,center_big_box)','On(center_big_box,box_board)'],
    # &IsHolding(right_franka,center_big_box)


    # "task4":['On(left_box,right_table)','On(left_box,right_table)&On(right_box,left_table)','On(left_box,right_table) & On(right_box,left_box)'],  # 左box放右桌子，右box放左box，右box放左box
    "task4":['On(left_box1,right_table)','On(left_box2,right_table)&On(right_box,left_table)','IsHolding(left_franka,left_box1)&IsHolding(right_franka,right_box)'],  # 左box放右桌子，右box放左box，右box放左box

    # 1.右手扶住右杯，将左牛奶倒一半到右杯中 (单向倒，单手协作扶杯)。
    # 2.完成左右牛奶的交换倾倒，两个杯子都倒至半满
    # 3.完成双向倾倒，并且所有物体都回到初始位置
    "task5":['IsHalfFull(right_cup)','IsHalfFull(right_cup) & IsHalfFull(left_cup)',
    'IsHalfFull(right_cup) & IsHalfFull(left_cup) & IsHandEmpty(left_franka) & IsHandEmpty(right_franka) & On(left_milk,left_table) & On(right_milk,right_table)'],
    # "task5":['IsFull(right_cup)','IsFull(left_cup)',
    # 'IsFull(right_cup) & On(right_cup,right_table)'],



    # "task6":[ # 放一个,放2个，放两个+恢复/关上
    #     'On(book,breakfast_table) & Open(cabinet)', 'In(book,cabinet) & In(pen,cabinet)', 
    #     'In(banana,fridge)  & IsClosed(fridge) & In(wine,fridge)'],

    # "task6":[ # 放一个,放2个，放两个+恢复/关上
    #     'In(book,cabinet) & IsOpened(cabinet)', 'On(book,breakfast_table) & In(pen,cabinet)', 
    #     'In(banana,fridge)  & IsClosed(fridge)'],

    "task6":[ # 放一个,放2个，放两个+恢复/关上
        'In(book,cabinet) & IsOpened(cabinet)', 'On(book,breakfast_table)', 
        'In(banana,fridge)  & IsClosed(fridge)'],


    # "task6":[ # 放一个,放2个，放两个+恢复/关上
    #     'In(apple,fridge) ', 'In(apple,fridge) & IsClosed(fridge)', 
    #     'In(pen,cabinet)& IsClosed(cabinet)'],

    # "task7":[ # 放鸡腿+开烤箱，放鸡腿+开烤箱+开灯
    #     'In(chickenleg,oven) & IsSwitchedOn(oven)',
    #     'In(chickenleg,oven) & IsSwitchedOn(oven) & IsSwitchedOn(microwave) & In(soup,microwave)',
    #     'In(chickenleg,oven) & IsSwitchedOn(oven) & IsSwitchedOn(microwave) & In(soup,microwave) & IsSwitchedOn(light)',
    # ]
    "task7":[ # 放鸡腿+开烤箱，放鸡腿+开烤箱+开灯
        'In(chickenleg,oven) & IsSwitchedOn(oven) & In(pie,oven)',
        'IsSwitchedOn(microwave) & In(soup,microwave) ',
        'IsOpened(microwave) & IsOpened(oven) & IsSwitchedOn(light) & IsSwitchedOn(radio) & On(pie,breakfast_table)'
        
    ]
}

total_try_times = 10  # 每个任务跑5次
max_feedback_times = 3

model = "gpt-4o" #"gpt-4o" #"gpt-4o-mini" #"gemini-2.0-flash-exp"#"gpt-4o-mini" #"gpt-3.5-turbo" #"gpt-4o-mini"

# 获取当前日期
current_date = datetime.now().strftime("%Y%m%d%H%M")

# 创建结果目录
result_dir = os.path.join(DIR, "a_exp1_llm_bt_results")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

just_validate = False

# 存储所有任务的结果
all_results = []

# 参与验证的任务列表
# task_names = ["task1","task2","task5","task4","task3","task7"]
# task_names = ["task1","task2"]
# task_names = ["task7"]
# task_names = ["task2"]

task_names = ["task1","task2","task3","task4","task5","task6","task7"]

for task_name in task_names:
    # 1. set task
    
    behavior_lib_path = os.path.join(DIR, f"a_exp1_llm_bt_tasks/{task_name}/exec_lib")
    
    initial_state = task2initial_state[task_name].copy()
    objects = set(task2objects[task_name])
    goal_str_list = task2goal_str[task_name]  # [easy, medium, hard]
    
    print("="*60)
    print(f"Task: {task_name}")
    print("objects:", objects)
    print("initial_state:", initial_state)
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
        messages = None
        llm = None
        try:
            if not just_validate:
                # 创建LLM对象
                llm = LLM(request_model=model)
                # 初始生成时清空lib（每次运行脚本时从干净状态开始）
                # 反馈时会补充新文件（clear_lib=False）
                # 使用语法错误自动重试的包装函数
                messages = generate_behavior_lib_with_syntax_retry(
                    goal_str_list=goal_str_list,
                    objects=objects,
                    initial_state=initial_state,
                    behavior_lib_path=behavior_lib_path,
                    llm=llm,
                    messages=None,  # 初始生成时messages为None
                    clear_lib=True,  # 初始生成时清空
                    save_io_dir=save_io_dir,
                    max_syntax_retries=3
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
                    'expanded_num': -1,
                    'act_num': -1,
                    'action_lib_num': 0,
                    'condition_lib_num': 0,
                    'feedback_times': 0,
                    'is_after_feedback': False,
                    'error': str(e)
                })
            continue
    
        # 2. 验证所有goal（初始验证，反馈前）
        difficulties = ['easy', 'medium', 'hard']
        print("\n=== 初始验证（反馈前）===")
        initial_results, all_goals_success = validate_all_goals(
            behavior_lib_path=behavior_lib_path,
            goal_str_list=goal_str_list,
            initial_state=initial_state,
            save_io_dir=save_io_dir,
            difficulties=difficulties
        )
        
        # 获取行为库数量
        action_lib_num, condition_lib_num = get_lib_counts(behavior_lib_path)
        print(f"Action lib num: {action_lib_num}")
        print(f"Condition lib num: {condition_lib_num}")
        
        # 记录初始验证结果（反馈前）
        for result in initial_results:
            all_results.append({
                'task_name': task_name,
                'try_idx': try_idx + 1,
                'difficulty': result['difficulty'],
                'goal': result['goal'],
                'success': result['success'],
                'expanded_num': result['expanded_num'],
                'act_num': result['act_num'],
                'action_lib_num': action_lib_num,
                'condition_lib_num': condition_lib_num,
                'feedback_times': 0,
                'is_after_feedback': False,
                'error': result.get('error')
            })
        
        # 统计这次尝试三个goal是否都成功
        print(f"\n{'='*60}")
        print(f"Task {task_name}, Try {try_idx + 1}: All goals success (初始) = {all_goals_success}")
        print(f"{'='*60}\n")

        # 3. 如果失败了，进行反馈循环
        # 如果验证失败但还没有LLM对象，创建它以便进行反馈
        if not all_goals_success and (messages is None or llm is None):
            if llm is None:
                print("Creating LLM object for feedback...")
                try:
                    llm = LLM(request_model=model)
                except Exception as e:
                    print(f"Error creating LLM object: {e}")
                    llm = None
            # 如果messages是None，创建一个空的messages列表
            if messages is None:
                messages = []
                # 添加初始prompt（如果需要的话）
                initial_state_str = "{" + ", ".join([f'"{s}"' for s in initial_state]) + "}"
                objects_str = "{" + ", ".join([f'"{o}"' for o in objects]) + "}"
                goal_list_str = "[" + ", ".join([f'"{g}"' for g in goal_str_list]) + "]"
                goal_str_combined = f"initial_state = {initial_state_str}\nobjects = {objects_str}\ngoal = {goal_list_str}"
                from exps_bt_learning.tools import build_prompt
                initial_prompt = build_prompt(goal=goal_str_combined, objects=objects, initial_state=initial_state)
                messages.append({"role": "user", "content": initial_prompt})
                # 添加一个占位的assistant回复（因为行为库已经存在）
                messages.append({"role": "assistant", "content": "Behavior library already exists."})
        
        record_feedback_times = 0
        final_results = initial_results.copy()
        
        while not all_goals_success and record_feedback_times < max_feedback_times and llm is not None:
            record_feedback_times += 1
            print(f"\033[95m=== 反馈第 {record_feedback_times} 次 ===\033[0m")
            
            # 确保messages已初始化
            if messages is None:
                print("Warning: messages is None, initializing...")
                messages = []
            
            # 收集失败的案例
            failed_results = [r for r in final_results if not r['success']]
            
            # 构建反馈prompt
            feedback_prompt = build_feedback_prompt(
                failed_results=failed_results,
                goal_str_list=goal_str_list,
                initial_state=initial_state,
                difficulties=difficulties
            )
            
            # 添加反馈到消息历史
            messages.append({"role": "user", "content": feedback_prompt})
            
            # 生成新的行为库（使用语法错误自动重试的包装函数）
            try:
                messages = generate_behavior_lib_with_syntax_retry(
                    goal_str_list=goal_str_list,
                    objects=objects,
                    initial_state=initial_state,
                    behavior_lib_path=behavior_lib_path,
                    llm=llm,
                    messages=messages,
                    clear_lib=True,
                    save_io_dir=save_io_dir,
                    max_syntax_retries=3
                )
            except Exception as e:
                print(f"Error generating behavior lib in feedback: {e}")
                break
            
            # 更新行为库数量
            action_lib_num, condition_lib_num = get_lib_counts(behavior_lib_path)
            print(f"Action lib num: {action_lib_num}")
            print(f"Condition lib num: {condition_lib_num}")
            
            # 重新验证所有goal
            print(f"\n=== 反馈后验证（第 {record_feedback_times} 次反馈）===")
            final_results, all_goals_success = validate_all_goals(
                behavior_lib_path=behavior_lib_path,
                goal_str_list=goal_str_list,
                initial_state=initial_state,
                save_io_dir=save_io_dir,
                difficulties=difficulties
            )
            
            # 记录反馈后的验证结果
            for result in final_results:
                all_results.append({
                    'task_name': task_name,
                    'try_idx': try_idx + 1,
                    'difficulty': result['difficulty'],
                    'goal': result['goal'],
                    'success': result['success'],
                    'expanded_num': result['expanded_num'],
                    'act_num': result['act_num'],
                    'action_lib_num': action_lib_num,
                    'condition_lib_num': condition_lib_num,
                    'feedback_times': record_feedback_times,
                    'is_after_feedback': True,
                    'error': result.get('error')
                })
            
            print(f"\n{'='*60}")
            print(f"Task {task_name}, Try {try_idx + 1}: All goals success (反馈后) = {all_goals_success}")
            print(f"{'='*60}\n")
            
            if all_goals_success:
                print(f"\033[92m所有goal在反馈后成功！\033[0m")
                break


# 保存结果到CSV
df = pd.DataFrame(all_results)
csv_filename = os.path.join(result_dir, f"exp1_results_{current_date}.csv")
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"\n{'='*60}")
print(f"Results saved to: {csv_filename}")
print(f"{'='*60}\n")

# 统计结果
print("\n" + "="*80)
print("=== 统计结果 ===")
print("="*80)

# 用于存储汇总表格数据
summary_data = []

for task_name in task_names:
    task_df = df[df['task_name'] == task_name]
    if len(task_df) == 0:
        continue
    
    print(f"\n{'='*80}")
    print(f"Task: {task_name}")
    print(f"{'='*80}")
    
    # 初始化汇总数据
    summary_row = {
        'task_name': task_name,
        # 反馈前的统计（只统计成功案例的平均值）
        'action_lib_num_no_feedback': 0.0,
        'condition_lib_num_no_feedback': 0.0,
        'expanded_num_no_feedback': 0.0,
        'act_num_no_feedback': 0.0,
        # 反馈后的统计（只统计成功案例的平均值）
        'action_lib_num_feedback': 0.0,
        'condition_lib_num_feedback': 0.0,
        'expanded_num_feedback': 0.0,
        'act_num_feedback': 0.0,
        # 成功率统计
        'easy_success_rate_no_feedback': 0.0,
        'medium_success_rate_no_feedback': 0.0,
        'hard_success_rate_no_feedback': 0.0,
        'all_success_rate_no_feedback': 0.0,
        'easy_success_rate_feedback': 0.0,
        'medium_success_rate_feedback': 0.0,
        'hard_success_rate_feedback': 0.0,
        'all_success_rate_feedback': 0.0,
        'average_feedback_times': 0.0
    }
    
    # 1. 反馈前的统计
    print("\n【反馈前统计】")
    before_feedback_df = task_df[task_df['is_after_feedback'] == False]
    
    # 收集所有成功案例的数据用于计算平均值
    all_success_before = before_feedback_df[before_feedback_df['success'] == True]
    
    if len(before_feedback_df) > 0:
        for difficulty in ['easy', 'medium', 'hard']:
            diff_df = before_feedback_df[before_feedback_df['difficulty'] == difficulty]
            if len(diff_df) > 0:
                success_count = diff_df['success'].sum()
                total_count = len(diff_df)
                success_rate = success_count / total_count * 100
                print(f"  {difficulty}: {success_count}/{total_count} ({success_rate:.2f}%)")
                
                # 保存成功率
                if difficulty == 'easy':
                    summary_row['easy_success_rate_no_feedback'] = success_rate
                elif difficulty == 'medium':
                    summary_row['medium_success_rate_no_feedback'] = success_rate
                elif difficulty == 'hard':
                    summary_row['hard_success_rate_no_feedback'] = success_rate
                
                # 计算成功情况下的平均值
                success_df = diff_df[diff_df['success'] == True]
                if len(success_df) > 0:
                    avg_acts = success_df['action_lib_num'].mean()
                    avg_conds = success_df['condition_lib_num'].mean()
                    avg_expands = success_df['expanded_num'].mean()
                    avg_steps = success_df['act_num'].mean()
                    print(f"    成功案例平均值: Acts={avg_acts:.2f}, Conds={avg_conds:.2f}, Expands={avg_expands:.2f}, Steps={avg_steps:.2f}")
        
        # 统计所有goal都成功的次数（反馈前）
        all_success_count_before = 0
        for try_idx in range(1, total_try_times + 1):
            try_df = before_feedback_df[before_feedback_df['try_idx'] == try_idx]
            if len(try_df) == 3 and try_df['success'].all():
                all_success_count_before += 1
        all_success_rate_before = all_success_count_before / total_try_times * 100
        summary_row['all_success_rate_no_feedback'] = all_success_rate_before
        print(f"  所有goal都成功: {all_success_count_before}/{total_try_times} ({all_success_rate_before:.2f}%)")
        
        # 计算反馈前所有成功案例的平均值（只统计成功案例）
        if len(all_success_before) > 0:
            summary_row['action_lib_num_no_feedback'] = all_success_before['action_lib_num'].mean()
            summary_row['condition_lib_num_no_feedback'] = all_success_before['condition_lib_num'].mean()
            summary_row['expanded_num_no_feedback'] = all_success_before['expanded_num'].mean()
            summary_row['act_num_no_feedback'] = all_success_before['act_num'].mean()
            print(f"  反馈前成功案例平均值: Acts={summary_row['action_lib_num_no_feedback']:.2f}, Conds={summary_row['condition_lib_num_no_feedback']:.2f}, Expands={summary_row['expanded_num_no_feedback']:.2f}, Steps={summary_row['act_num_no_feedback']:.2f}")
    
    # 2. 反馈后的统计
    # 重要：反馈后的统计应该包含所有尝试的最终结果
    # - 如果某个尝试在反馈前就全部成功（没有进入反馈循环），使用反馈前的结果
    # - 如果某个尝试有反馈，使用最后一次反馈的结果
    # 这样确保反馈后的成功率 >= 反馈前的成功率
    print("\n【反馈后统计】")
    after_feedback_df = task_df[task_df['is_after_feedback'] == True]
    
    # 构建所有尝试的最终结果（反馈后）
    # 对于每个尝试，取最终的结果：
    # 1. 如果有反馈后的数据，使用最后一次反馈的结果
    # 2. 如果没有反馈后的数据（说明反馈前就全部成功），使用反馈前的结果
    final_results_after = []
    for try_idx in range(1, total_try_times + 1):
        # 先检查是否有反馈后的数据
        try_df_after = after_feedback_df[after_feedback_df['try_idx'] == try_idx]
        if len(try_df_after) > 0:
            # 有反馈数据，取最后一次反馈的结果
            max_feedback = try_df_after['feedback_times'].max()
            final_try_df = try_df_after[try_df_after['feedback_times'] == max_feedback]
            final_results_after.extend(final_try_df.to_dict('records'))
        else:
            # 没有反馈数据，说明反馈前就全部成功了（没有进入反馈循环）
            # 使用反馈前的结果，这样这些成功的案例也会被计入反馈后的统计
            try_df_before = before_feedback_df[before_feedback_df['try_idx'] == try_idx]
            if len(try_df_before) > 0:
                final_results_after.extend(try_df_before.to_dict('records'))
            else:
                # 理论上不应该出现这种情况，但为了安全起见，记录警告
                print(f"\033[93m  WARNING: No data found for try_idx {try_idx} in both before and after feedback\033[0m")
    
    # 转换为DataFrame以便统计
    all_success_after = pd.DataFrame()  # 初始化为空DataFrame
    if len(final_results_after) > 0:
        final_results_after_df = pd.DataFrame(final_results_after)
        
        # 验证：确保所有尝试都被包含在反馈后的统计中
        unique_try_indices = final_results_after_df['try_idx'].unique()
        if len(unique_try_indices) != total_try_times:
            print(f"\033[93m  WARNING: Expected {total_try_times} tries, but found {len(unique_try_indices)} tries in feedback results\033[0m")
            print(f"    Missing try indices: {set(range(1, total_try_times + 1)) - set(unique_try_indices)}")
        
        # 验证：确保每个尝试都有所有三个难度的结果
        for try_idx in range(1, total_try_times + 1):
            try_df = final_results_after_df[final_results_after_df['try_idx'] == try_idx]
            if len(try_df) != 3:
                print(f"\033[93m  WARNING: Try {try_idx} has {len(try_df)} difficulty results, expected 3 (easy, medium, hard)\033[0m")
                existing_difficulties = set(try_df['difficulty'].unique())
                missing_difficulties = {'easy', 'medium', 'hard'} - existing_difficulties
                if len(missing_difficulties) > 0:
                    print(f"    Missing difficulties: {missing_difficulties}")
        
        # 收集所有成功案例的数据用于计算平均值
        all_success_after = final_results_after_df[final_results_after_df['success'] == True]
        
        for difficulty in ['easy', 'medium', 'hard']:
            diff_df = final_results_after_df[final_results_after_df['difficulty'] == difficulty]
            if len(diff_df) > 0:
                success_count = diff_df['success'].sum()
                total_count = len(diff_df)
                success_rate = success_count / total_count * 100 if total_count > 0 else 0.0
                
                # 验证：确保每个难度都包含了所有尝试的结果
                if total_count != total_try_times:
                    print(f"\033[93m  WARNING: {difficulty} difficulty has {total_count} results, expected {total_try_times}\033[0m")
                    missing_try_indices = set(range(1, total_try_times + 1)) - set(diff_df['try_idx'].unique())
                    if len(missing_try_indices) > 0:
                        print(f"    Missing try indices: {missing_try_indices}")
                
                print(f"  {difficulty}: {success_count}/{total_count} ({success_rate:.2f}%)")
                
                # 保存成功率
                if difficulty == 'easy':
                    summary_row['easy_success_rate_feedback'] = success_rate
                elif difficulty == 'medium':
                    summary_row['medium_success_rate_feedback'] = success_rate
                elif difficulty == 'hard':
                    summary_row['hard_success_rate_feedback'] = success_rate
                
                # 计算成功情况下的平均值
                success_df = diff_df[diff_df['success'] == True]
                if len(success_df) > 0:
                    avg_acts = success_df['action_lib_num'].mean()
                    avg_conds = success_df['condition_lib_num'].mean()
                    avg_expands = success_df['expanded_num'].mean()
                    avg_steps = success_df['act_num'].mean()
                    print(f"    成功案例平均值: Acts={avg_acts:.2f}, Conds={avg_conds:.2f}, Expands={avg_expands:.2f}, Steps={avg_steps:.2f}")
        
        # 统计所有goal都成功的次数（反馈后）
        all_success_count_after = 0
        for try_idx in range(1, total_try_times + 1):
            try_df = final_results_after_df[final_results_after_df['try_idx'] == try_idx]
            if len(try_df) == 3 and try_df['success'].all():
                all_success_count_after += 1
        all_success_rate_after = all_success_count_after / total_try_times * 100
        summary_row['all_success_rate_feedback'] = all_success_rate_after
        print(f"  所有goal都成功: {all_success_count_after}/{total_try_times} ({all_success_rate_after:.2f}%)")
        
        # 验证：反馈后的成功率应该 >= 反馈前的成功率（理论上）
        # 如果出现下降，可能是反馈机制引入了新的错误，或者统计逻辑有问题
        if len(before_feedback_df) > 0:
            for difficulty in ['easy', 'medium', 'hard']:
                before_diff_df = before_feedback_df[before_feedback_df['difficulty'] == difficulty]
                if len(before_diff_df) > 0:
                    before_success_rate = before_diff_df['success'].sum() / len(before_diff_df) * 100
                    after_success_rate = summary_row.get(f'{difficulty}_success_rate_feedback', 0.0)
                    if after_success_rate < before_success_rate:
                        print(f"\033[93m  WARNING: {difficulty} success rate decreased after feedback: {before_success_rate:.2f}% -> {after_success_rate:.2f}%\033[0m")
                        print(f"    This may indicate that feedback introduced errors or there's a statistical issue.")
            
            # 检查所有goal都成功的成功率
            if all_success_rate_after < all_success_rate_before:
                print(f"\033[93m  WARNING: All goals success rate decreased after feedback: {all_success_rate_before:.2f}% -> {all_success_rate_after:.2f}%\033[0m")
                print(f"    This may indicate that feedback introduced errors or there's a statistical issue.")
        
        # 3. 平均反馈次数
        # 对每个try_idx，取最大的feedback_times（如果没有反馈数据，则为0）
        feedback_times_list = []
        for try_idx in range(1, total_try_times + 1):
            try_df = after_feedback_df[after_feedback_df['try_idx'] == try_idx]
            if len(try_df) > 0:
                max_feedback = try_df['feedback_times'].max()
                feedback_times_list.append(max_feedback)
            else:
                # 如果没有反馈数据，说明初始就成功了，反馈次数为0
                feedback_times_list.append(0)
        if len(feedback_times_list) > 0:
            avg_feedback_times = sum(feedback_times_list) / len(feedback_times_list)
            summary_row['average_feedback_times'] = avg_feedback_times
            print(f"\n  平均反馈次数: {avg_feedback_times:.2f} (共{total_try_times}次尝试)")
        else:
            print(f"\n  平均反馈次数: 0.00 (共{total_try_times}次尝试)")
    else:
        print("  无反馈数据")
    
    # 计算反馈后所有成功案例的平均值（只统计成功案例）
    # all_success_after 包含了所有最终成功的案例：
    #   - 如果某个尝试在反馈前就成功，使用反馈前的结果
    #   - 如果某个尝试经过反馈后成功，使用反馈后的结果
    if len(all_success_after) > 0:
        # 使用反馈后的成功案例平均值（all_success_after 已经是 DataFrame，只包含成功的案例）
        summary_row['action_lib_num_feedback'] = all_success_after['action_lib_num'].mean()
        summary_row['condition_lib_num_feedback'] = all_success_after['condition_lib_num'].mean()
        summary_row['expanded_num_feedback'] = all_success_after['expanded_num'].mean()
        summary_row['act_num_feedback'] = all_success_after['act_num'].mean()
        print(f"  反馈后成功案例平均值: Acts={summary_row['action_lib_num_feedback']:.2f}, Conds={summary_row['condition_lib_num_feedback']:.2f}, Expands={summary_row['expanded_num_feedback']:.2f}, Steps={summary_row['act_num_feedback']:.2f}")
    elif len(all_success_before) > 0:
        # 如果反馈后没有成功案例，但反馈前有，使用反馈前的（这种情况理论上不应该发生）
        summary_row['action_lib_num_feedback'] = all_success_before['action_lib_num'].mean()
        summary_row['condition_lib_num_feedback'] = all_success_before['condition_lib_num'].mean()
        summary_row['expanded_num_feedback'] = all_success_before['expanded_num'].mean()
        summary_row['act_num_feedback'] = all_success_before['act_num'].mean()
        print(f"  反馈后成功案例平均值（使用反馈前的数据）: Acts={summary_row['action_lib_num_feedback']:.2f}, Conds={summary_row['condition_lib_num_feedback']:.2f}, Expands={summary_row['expanded_num_feedback']:.2f}, Steps={summary_row['act_num_feedback']:.2f}")
    
    # 添加到汇总数据
    summary_data.append(summary_row)

print("\n" + "="*80)
print("完成！")
print("="*80)

# 生成汇总表格
if len(summary_data) > 0:
    summary_df = pd.DataFrame(summary_data)
    
    # 定义列名（按要求的顺序）
    columns = [
        'task_name',
        # 反馈前的统计
        'action_lib_num_no_feedback',
        'condition_lib_num_no_feedback',
        'expanded_num_no_feedback',
        'act_num_no_feedback',
        # 反馈后的统计
        'action_lib_num_feedback',
        'condition_lib_num_feedback',
        'expanded_num_feedback',
        'act_num_feedback',
        # 成功率统计
        'easy_success_rate_no_feedback',
        'medium_success_rate_no_feedback',
        'hard_success_rate_no_feedback',
        'all_success_rate_no_feedback',
        'easy_success_rate_feedback',
        'medium_success_rate_feedback',
        'hard_success_rate_feedback',
        'all_success_rate_feedback',
        'average_feedback_times'
    ]
    
    # 重新排列列
    summary_df = summary_df[columns]
    
    # 重命名列为更友好的名称
    summary_df.columns = [
        'task name',
        # 反馈前的统计
        'action lib num no feedback',
        'condition lib num no feedback',
        'expanded num no feedback',
        'act num (steps) no feedback',
        # 反馈后的统计
        'action lib num feedback',
        'condition lib num feedback',
        'expanded num feedback',
        'act num (steps) feedback',
        # 成功率统计
        'easy success rate no feedback',
        'medium success rate no feedback',
        'hard success rate no feedback',
        'all success rate no feedback',
        'easy success rate feedback',
        'medium success rate feedback',
        'hard success rate feedback',
        'all success rate feedback',
        'average feedback times'
    ]
    
    # 保存为CSV文件
    summary_csv_filename = os.path.join(result_dir, f"summary_results_{model}_{current_date}.csv")
    summary_df.to_csv(summary_csv_filename, index=False, encoding='utf-8-sig')
    
    # 保存为TSV文件（制表符分隔，可直接复制到Excel）
    summary_tsv_filename = os.path.join(result_dir, f"summary_results_{model}_{current_date}.tsv")
    summary_df.to_csv(summary_tsv_filename, index=False, sep='\t', encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("=== 汇总表格（可直接复制到Excel）===")
    print("="*80)
    
    # 输出制表符分隔的格式，方便直接复制到Excel
    # 先输出表头
    header = '\t'.join(summary_df.columns)
    print(header)
    
    # 输出每一行数据
    for _, row in summary_df.iterrows():
        # 将每行的值转换为字符串，并用制表符连接
        row_str = '\t'.join([str(val) for val in row.values])
        print(row_str)
    
    print("="*80)
    print(f"\n汇总表格已保存到:")
    print(f"  CSV格式: {summary_csv_filename}")
    print(f"  TSV格式（可直接复制到Excel）: {summary_tsv_filename}")
    print("\n提示：上面的表格可以直接复制粘贴到Excel中")
    print("="*80)