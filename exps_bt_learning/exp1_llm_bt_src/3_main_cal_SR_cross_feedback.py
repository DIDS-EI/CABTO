import os
import base64
import pandas as pd
from btgym.llm.llm_gpt import LLM
from openai import OpenAI
from exps_bt_learning.llm_generate_lib_func import llm_generate_behavior_lib_need_feedback
from exps_bt_learning.validate_bt_fun import validate_bt_fun
from exps_bt_learning.tools import build_prompt
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def encode_image(image_path):
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

def build_behavior_lib_prompt_with_image(goal_str, objects, initial_state, action_name=None, action_pre=None, action_add=None, action_del=None, use_image=True):
    """
    构建生成行为库的prompt（可选结合图片）
    
    Args:
        goal_str: 单个goal字符串
        objects: 对象集合
        initial_state: 初始状态集合
        action_name: 动作名称（可选）
        action_pre: 动作的前置条件（可选）
        action_add: 动作的添加效果（可选）
        action_del: 动作的删除效果（可选）
        use_image: 是否使用图片（默认True）
    """
    # 格式化 initial_state 为 Python 集合格式
    initial_state_str = "{" + ", ".join([f'"{s}"' for s in initial_state]) + "}"
    # 格式化 objects 为 Python 集合格式
    objects_str = "{" + ", ".join([f'"{o}"' for o in objects]) + "}"
    # 格式化 goal 为字符串格式
    goal_str_combined = f"initial_state = {initial_state_str}\nobjects = {objects_str}\ngoal = \"{goal_str}\""
    
    # 使用build_prompt构建基础prompt
    base_prompt = build_prompt(goal=goal_str_combined, objects=objects, initial_state=initial_state)
    
    # 根据是否使用图片添加不同的说明
    if use_image:
        # 添加图片相关的说明
        image_prompt = f"""
**重要提示：**
请仔细观察提供的图片，理解场景中的物体、状态和关系。基于图片中的实际情况，生成完整的行为库（包括所有Action和Condition节点）。

图片将帮助你：
1. 理解物体的实际状态和位置关系
2. 识别动作执行的前置条件和效果
3. 确保生成的行为库能够正确规划从初始状态到目标状态的路径

{base_prompt}"""
    else:
        # 不使用图片时的说明
        image_prompt = f"""
**重要提示：**
请根据提供的初始状态、对象和目标，生成完整的行为库（包括所有Action和Condition节点）。

{base_prompt}"""
    
    # 如果有动作信息，在最后添加动作定义和错误分析
    if action_name is not None and action_pre is not None and action_add is not None and action_del is not None:
        # 格式化动作的 pre、add、del
        pre_list = list(action_pre) if isinstance(action_pre, set) else action_pre
        add_list = list(action_add) if isinstance(action_add, set) else action_add
        del_list = list(action_del) if isinstance(action_del, set) else action_del
        
        action_info = f"""

当前的动作 {action_name} 的定义是：
- 前置条件（pre）：{pre_list}
- 添加效果（add）：{add_list}
- 删除效果（del）：{del_list}

但是执行失败了，请你帮忙分析 pre、add 和 del 是否有错误，动作 {action_name} 需要重新生成 pre、add 和 del，用 Python 格式给出。另外也需要给出其它所有的动作和条件。只是需要注意的是，
{action_name} 的 pre、add、del 需要检查和修正，因为 {action_name} 执行失败了，是不是有些条件没有满足。或者它的 pre、add、del 不符合实际情况。"""
        
        if use_image:
            action_info += "\n\n请仔细观察图片，理解场景中的实际情况，确保动作定义符合图片中的场景。"
        
        # 比如放入前要检查有没有打开柜子
        image_prompt += action_info
    
    return image_prompt

def validate_goal(behavior_lib_path, goal_str, initial_state, save_io_dir):
    """
    验证单个goal，返回验证结果
    
    Returns:
        result: dict, 包含验证结果
        success: bool, 是否成功
    """
    print(f"\n--- Validating goal: {goal_str} ---")
    
    # 创建输出目录
    output_dir = os.path.join(save_io_dir, "bt.btml")
    
    try:
        error, bt, expanded_num, act_num, record_act_ls, ptml_string = validate_bt_fun(
            behavior_lib_path=behavior_lib_path,
            goal_str=goal_str,
            cur_cond_set=initial_state,
            output_dir=output_dir
        )
        
        success = (error == 0)
        print(f"Goal result: {'SUCCESS' if success else 'FAILED'}")
        
        # 记录结果
        result = {
            'goal': goal_str,
            'success': success,
            'expanded_num': expanded_num,
            'act_num': act_num,
            'error': None,
            'ptml_string': ptml_string,
            'record_act_ls': record_act_ls
        }
        
        return result, success
        
    except Exception as e:
        print(f"Error validating goal: {e}")
        
        # 记录失败结果
        result = {
            'goal': goal_str,
            'success': False,
            'expanded_num': -1,
            'act_num': -1,
            'error': str(e),
            'ptml_string': None,
            'record_act_ls': None
        }
        
        return result, False

def get_lib_counts(behavior_lib_path):
    """获取行为库中Action和Condition的数量"""
    try:
        action_lib_num = len([f for f in os.listdir(os.path.join(behavior_lib_path, 'Action')) if f.endswith('.py')])
        condition_lib_num = len([f for f in os.listdir(os.path.join(behavior_lib_path, 'Condition')) if f.endswith('.py')])
        return action_lib_num, condition_lib_num
    except Exception as e:
        print(f"Error counting lib files: {e}")
        return 0, 0

def build_feedback_prompt_with_image(failed_result, goal_str, initial_state, use_image=True):
    """
    构建反馈prompt（可选结合图片）
    
    Args:
        failed_result: dict, 失败的验证结果
        goal_str: str, goal字符串
        initial_state: set, 初始状态
        use_image: bool, 是否使用图片（默认True）
    """
    if use_image:
        feedback_prompt = "The current behavior library fails to generate valid behavior trees that can achieve the specified goal, or encounters errors during the planning process. Please observe the provided image carefully to understand the scene and correct the behavior library. The following is the failed case:\n\n"
    else:
        feedback_prompt = "The current behavior library fails to generate valid behavior trees that can achieve the specified goal, or encounters errors during the planning process. Please correct the behavior library. The following is the failed case:\n\n"
    
    goal_str = failed_result['goal']
    ptml_string = failed_result.get('ptml_string', None)
    expanded_num = failed_result.get('expanded_num', -1)
    act_num = failed_result.get('act_num', -1)
    record_act_ls = failed_result.get('record_act_ls', None)
    
    feedback_prompt += f"[Failed Case]\n"
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
        error = failed_result.get('error', 'Unknown error')
        feedback_prompt += f"Failed to generate behavior tree. Error: {error}\n"
        # If the error message contains critical information, emphasize it
        if 'valid_args' in str(error) or 'attribute' in str(error) or 'Error' in str(error):
            feedback_prompt += f"IMPORTANT: This error indicates that the behavior library has structural issues. "
            feedback_prompt += f"Please ensure all action and condition classes have the required attributes (valid_args, num_args, can_be_expanded, etc.) properly defined.\n"
    
    feedback_prompt += "\n"
    
    feedback_prompt += """
Please analyze the failed cases above and the behavior trees generated by the current behavior library, which are described in BTML (Behavior Tree Markup Language)."""
    
    if use_image:
        feedback_prompt += """ Observe the provided image carefully to understand the scene, objects, and their relationships."""
    
    feedback_prompt += """ Key concepts:

1. **Behavior Trees (BTML)** serve as the planning blueprint for finding an **executable path** from the **Initial State** to the **Goal**.
2. A **`selector`** node succeeds upon the first successful child; a **`sequence`** node requires all children to succeed sequentially.
3. Planning fails when, starting from the Initial State, the tree **lacks any traversable path** where all subsequent conditions (`cond`) evaluate to true, thus failing to reach an executable action (`act`).

**IMPORTANT**: You must regenerate the ENTIRE behavior library from scratch. The behavior library must be capable of successfully generating behavior trees that can achieve the specified goal. All previous code will be cleared."""
    
    if use_image:
        feedback_prompt += """ Use the image to understand the actual scene and ensure the behavior library matches the reality."""
    else:
        feedback_prompt += """ Ensure the behavior library matches the requirements based on the initial state, objects, and goal."""
    
    feedback_prompt += """

**Requirements for the regenerated behavior library:**

1. **Completeness**: Generate all action and condition nodes necessary to achieve the goal.

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

def request_with_image(llm_client, image_path, prompt_text):
    """
    使用图片和文本发送请求到LLM
    
    Args:
        llm_client: LLM客户端对象
        image_path: 图片路径
        prompt_text: 提示文本
    """
    # 编码图片
    img_base64 = encode_image(image_path)
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        }
    ]
    
    # 发送请求（使用支持视觉的模型）
    try:
        response = llm_client.request(messages)
        return response
    except Exception as e:
        print(f"请求失败: {e}")
        return None


# 定义5个示例案例
example_cases = [
    # 案例1: PutIn(apple,cabinet) - pre缺少IsOpen(cabinet)
    {
        "image_name": "putin_failure.png",
        "action_name": "PutIn(apple,cabinet)",

        # "initial_state": {"IsHoliding(apple)"},
        # "objects": {"apple", "cabinet"},
        # "goal_str": "In(apple,cabinet)",

        "initial_state": {"IsHandEmpty()","On(apple,table)"},
        "objects": {"apple", "table","cabinet"},
        "goal_str": "In(apple,cabinet)",

        "current_pre": {"Holding(apple)"},
        "current_add": {"In(apple,cabinet)"},
        "current_del": {"Holding(apple)"},


        "error_type": "missing",
        "error_description": "Pre : IsOpen(cabinet) missing",
        "correct_pre": {"Holding(apple)", "IsOpen(cabinet)"},
        "correct_add": {"In(apple,cabinet)"},
        "correct_del": {"Holding(apple)"}
    },
    
    # 案例2: Stack(red,green) - pre缺少Clear(green)
    {
        "image_name": "stack_failure.png",
        "action_name": "Stack(red,green)",


        "initial_state": {"IsHandEmpty()", "On(yellow,green)", "On(red,table)", "On(green,table)"},
        "objects": {"red", "green", "yellow", "table"},
        "goal_str": "On(red,green)",


        "current_pre": {"IsHolding(red)"},
        "current_add": {"On(red,green)"},
        "current_del": {"IsHolding(red)"},


        "error_type": "missing",
        "error_description": "Pre : Clear(green) missing",
        "correct_pre": {"IsHolding(red)", "Clear(green)"},
        "correct_add": {"On(red,green)"},
        "correct_del": {"IsHolding(red)"}
    },
    
    # 案例3: Lift(big_box,board) - pre缺少IsHolding(leftrobot,big_box)
    {
        "image_name": "lift_failure.png",
        "action_name": "Lift(big_box,board)",

        "initial_state": {"IsHandEmpty(right_robot)","IsHandEmpty(left_robot)", "On(right_robot,right_table)", "On(left_robot,left_table)", "On(big_box,center_table)"},
        "objects": {"big_box", "right_robot", "left_robot", "right_table", "left_table", "center_table"},
        "goal_str": "On(big_box,board)",

        "current_pre": {"On(right_robot,right_table)", "On(big_box,center_table)", "IsHolding(left_robot,big_box)"},
        "current_add": { "On(big_box,center_table)"},
        "current_del": {"IsHandEmpty(left_robot)", "On(big_box,center_table)"},


        "error_type": "missing",
        "error_description": "Pre : IsHolding(left_robot,big_box) missing",
        "correct_pre": {"IsHolding(left_robot,big_box)"},
        "correct_add": {"On(big_box,right_table)"},
        "correct_del": {"On(big_box,center_table)"}
    },
    
    # 案例4: Pick(right_robot,left_green_tea) - add和del错误地添加了抓取效果，但实际上图片里是够不着的
    {
        "image_name": "pick_failure.png",
        "action_name": "Pick(right_robot,left_green_tea)",


        "initial_state": {"IsHandEmpty(right_robot)","IsHandEmpty(left_franka)", "On(left_green_tea,left_table)"},
        "objects": {"right_robot", "left_franka", "left_green_tea", "left_table", "right_table"},
        "goal_str": "On(left_green_tea,right_table)",


        "current_pre": {"IsHandEmpty(right_robot)", "On(left_green_tea,left_table)"},
        "current_add": {"Holding(right_robot,left_green_tea)"},
        "current_del": {"IsHandEmpty(right_robot)", "On(left_green_tea,left_table)"},


        "error_type": "incorrect",
        "error_description": "Add and Del : Holding effect incorrectly added, but object is unreachable in image",
        "correct_pre": {"IsHandEmpty()", "On(left_green_tea,table)"},
        "correct_add": set(),  # 因为够不着，所以不应该添加Holding
        "correct_del": set()   # 因为够不着，所以不应该删除任何状态
    },
    
    # 案例5: Put(apple,table) - del里没有删除apple在其它所有位置
    {
        "image_name": "put_failure.png",
        "action_name": "Put(apple,table)",


        "initial_state": {"Holiding(apple)", "On(apple,plate)"},
        "objects": {"apple", "cabinet"},
        "goal_str": "On(apple,table)",

        "current_pre": {"Holiding(apple)"},
        "current_add": {"On(apple,table)"},
        "current_del": {"Holiding(apple)"},


        "error_type": "missing",
        "error_description": "Del : On(apple,*) positions missing (should delete all On(apple,location) where location != table)",
        "correct_pre": {"Holding(apple)"},
        "correct_add": {"On(apple,table)"},
        "correct_del": {"Holding(apple)", "On(apple,other_surface)", "On(apple,cabinet)", "On(apple,board)"}  # 应该删除所有On(apple,*)的位置（除了table）
    }
]

def process_case_with_image(case, image_path, behavior_lib_path, llm, output_dir, max_feedback_times=3, try_idx=None, case_name=None, use_image=True):
    """
    处理案例，使用图片生成行为库并验证
    
    Args:
        case: 案例字典，包含 initial_state, objects, goal (或 goal_str_list)
        image_path: 图片路径
        behavior_lib_path: 行为库路径
        llm: LLM对象
        output_dir: 输出目录
        max_feedback_times: 最大反馈次数（默认3次）
        try_idx: 运行次数索引（用于区分多次运行）
        case_name: 案例名称（如果未提供，则从image_name生成）
        use_image: 是否使用图片（默认True）
    
    Returns:
        dict: 结果字典
    """
    if case_name is None:
        case_name = case['image_name'].replace('.png', '').replace('.jpg', '')
    if try_idx is not None:
        case_name = f"{case_name}_try{try_idx}"
    
    # 从case中获取必要信息
    initial_state = case.get("initial_state", set())
    objects = case.get("objects", set())
    # 获取单个goal
    if "goal_str" in case:
        # goal_str 是字符串
        goal_str = case["goal_str"]
    elif "goal" in case:
        # goal 可能是 set 或 str
        goal_str = list(case["goal"])[0] if isinstance(case["goal"], set) else case["goal"]
    elif "goal_str_list" in case:
        # 如果提供了goal_str_list，使用第一个
        goal_str = case["goal_str_list"][0] if isinstance(case["goal_str_list"], list) else case["goal_str_list"]
    else:
        raise ValueError("Case must have either 'goal_str', 'goal' or 'goal_str_list'")
    
    # 为了兼容llm_generate_behavior_lib_need_feedback，将单个goal转换为列表
    goal_str_list = [goal_str]
    
    # 创建LLM对象（如果还没有）
    if llm is None:
        llm = LLM(request_model="gpt-4o")
    
    messages = None
    feedback_times = 0
    goal_success = False
    final_result = None
    
    # 编码图片（如果需要使用图片）
    img_base64 = None
    if use_image:
        img_base64 = encode_image(image_path)
    
    # 初始生成行为库
    print(f"\n=== 初始生成行为库 ===")
    try:
        # 从case中获取动作信息（如果存在）
        action_name = case.get("action_name", None)
        action_pre = case.get("current_pre", None)
        action_add = case.get("current_add", None)
        action_del = case.get("current_del", None)
        
        # 构建初始prompt（可选结合图片）
        prompt = build_behavior_lib_prompt_with_image(
            goal_str=goal_str,
            objects=objects,
            initial_state=initial_state,
            action_name=action_name,
            action_pre=action_pre,
            action_add=action_add,
            action_del=action_del,
            use_image=use_image
        )
        
        # 打印发送给大模型的 prompt
        print("\n" + "="*80)
        print("发送给大模型的 Prompt:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        # 构建消息（根据是否使用图片）
        if use_image and img_base64:
            # 构建包含图片的消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
        else:
            # 构建纯文本消息（不使用图片）
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        
        # 使用语法错误自动重试的包装函数生成行为库
        # 注意：这里需要修改llm_generate_behavior_lib_need_feedback以支持图片消息
        # 暂时先使用标准方式，后续可以扩展
        messages = generate_behavior_lib_with_syntax_retry(
            goal_str_list=goal_str_list,
            objects=objects,
            initial_state=initial_state,
            behavior_lib_path=behavior_lib_path,
            llm=llm,
            messages=messages,
            clear_lib=True,
            save_io_dir=output_dir,
            max_syntax_retries=3
        )
    except Exception as e:
        print(f"Error generating behavior lib: {e}")
        # 如果生成失败，记录失败结果
        final_result = {
            'goal': goal_str,
            'success': False,
            'expanded_num': -1,
            'act_num': -1,
            'error': str(e),
            'ptml_string': None,
            'record_act_ls': None
        }
        goal_success = False
    
    # 验证goal（初始验证，反馈前）
    if final_result is None:
        print("\n=== 初始验证（反馈前）===")
        final_result, goal_success = validate_goal(
            behavior_lib_path=behavior_lib_path,
            goal_str=goal_str,
            initial_state=initial_state,
            save_io_dir=output_dir
        )
    
    # 获取行为库数量
    action_lib_num, condition_lib_num = get_lib_counts(behavior_lib_path)
    print(f"Action lib num: {action_lib_num}")
    print(f"Condition lib num: {condition_lib_num}")
    
    # 反馈循环
    while not goal_success and feedback_times < max_feedback_times:
        feedback_times += 1
        print(f"\033[95m=== 反馈第 {feedback_times} 次 ===\033[0m")
        
        # 确保messages已初始化
        if messages is None:
            print("Warning: messages is None, initializing...")
            messages = []
        
        # 如果goal失败，构建反馈prompt（结合图片）
        if not final_result['success']:
            feedback_prompt = build_feedback_prompt_with_image(
                failed_result=final_result,
                goal_str=goal_str,
                initial_state=initial_state,
                use_image=use_image
            )
            
            # 打印发送给大模型的反馈 prompt
            print("\n" + "="*80)
            print(f"发送给大模型的反馈 Prompt (第 {feedback_times} 次反馈):")
            print("="*80)
            print(feedback_prompt)
            print("="*80 + "\n")
        else:
            # 如果goal成功，不应该进入反馈循环
            break
        
        # 构建反馈消息（根据是否使用图片）
        if use_image and img_base64:
            # 构建包含图片的反馈消息
            feedback_message_content = [
                {
                    "type": "text",
                    "text": feedback_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        else:
            # 构建纯文本反馈消息（不使用图片）
            feedback_message_content = feedback_prompt
        
        # 添加反馈到消息历史
        messages.append({"role": "user", "content": feedback_message_content})
        
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
                save_io_dir=output_dir,
                max_syntax_retries=3
            )
        except Exception as e:
            print(f"Error generating behavior lib in feedback: {e}")
            break
        
        # 更新行为库数量
        action_lib_num, condition_lib_num = get_lib_counts(behavior_lib_path)
        print(f"Action lib num: {action_lib_num}")
        print(f"Condition lib num: {condition_lib_num}")
        
        # 重新验证goal
        print(f"\n=== 反馈后验证（第 {feedback_times} 次反馈）===")
        final_result, goal_success = validate_goal(
            behavior_lib_path=behavior_lib_path,
            goal_str=goal_str,
            initial_state=initial_state,
            save_io_dir=output_dir
        )
        
        print(f"\n{'='*60}")
        print(f"Case {case_name}: Goal success (反馈后) = {goal_success}")
        print(f"{'='*60}\n")
        
        if goal_success:
            print(f"\033[92mGoal在反馈后成功！\033[0m")
            break
    
    # 构建最终结果字典
    result = {
        "image_name": case["image_name"],
        "case_name": case_name,
        "initial_state": str(initial_state),
        "objects": str(objects),
        "goal": goal_str,
        "feedback_times": feedback_times,
        "success": goal_success,
        "expanded_num": final_result.get('expanded_num', -1),
        "act_num": final_result.get('act_num', -1),
        "action_lib_num": action_lib_num,
        "condition_lib_num": condition_lib_num,
        "try_idx": try_idx
    }
    
    return result

def main():
    """主函数"""
    # 配置
    model = "gpt-4o"  # 使用支持视觉的模型
    image_dir = os.path.join(DIR, "a_exp3_pic")  # 图片目录
    output_dir = os.path.join(DIR, "a_exp3_cross_feedback_results")
    
    total_try_times = 10  # 每个案例跑几次，取平均值
    max_feedback_times = 3  # 最大反馈次数
    use_image = True  # 是否使用图片（True=传入图片，False=不传入图片）
    case_ls = [2]  # 可以选择要处理的案例，例如：[0, 1, 2] 或 ["PutIn(apple,cabinet)", "Stack(red,green)"]
                  # 如果为空列表，则处理所有案例
                  # 可以通过索引（0-4）或案例名称来选择
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # 创建LLM对象
    llm = LLM(request_model=model)
    
    # 根据 case_ls 过滤案例
    def filter_cases(all_cases, case_ls):
        """根据 case_ls 过滤案例，支持全局索引（0-4）、案例名称或图片名称"""
        if not case_ls:
            return all_cases
        
        filtered = []
        for i, case in enumerate(all_cases):
            # 检查是否通过全局索引匹配（0-4）
            if i in case_ls:
                filtered.append(case)
            # 检查是否通过案例名称匹配
            elif case["action_name"] in case_ls:
                filtered.append(case)
            # 检查是否通过图片名称匹配
            elif case["image_name"] in case_ls:
                filtered.append(case)
        return filtered
    
    # 根据 case_ls 过滤所有案例（使用全局索引）
    filtered_all_cases = filter_cases(example_cases, case_ls)
    
    # 打印将要处理的案例
    print("\n" + "="*80)
    print("案例选择")
    print("="*80)
    if case_ls:
        print(f"选择的案例: {case_ls}")
        print(f"\n将处理的案例 ({len(filtered_all_cases)} 个):")
        for case in filtered_all_cases:
            print(f"    - {case['action_name']} ({case['image_name']})")
    else:
        print("将处理所有案例")
        print(f"  总案例数: {len(filtered_all_cases)}")
    
    all_results = []
    
    # 处理所有案例（带反馈循环）
    print("="*80)
    print("处理案例（使用图片生成行为库并验证）")
    print("="*80)
    print(f"每个案例将运行 {total_try_times} 次")
    print(f"使用图片: {'是' if use_image else '否'}")
    
    for case_idx, case in enumerate(filtered_all_cases, start=1):
        image_path = os.path.join(image_dir, case["image_name"])
        
        # 如果使用图片，检查图片是否存在
        if use_image and not os.path.exists(image_path):
            print(f"跳过案例 {case['action_name']}：图片不存在 {image_path}")
            continue
        
        # 为每个案例创建行为库路径，使用简单的 case1, case2 等命名
        case_name = f"case{case_idx}"
        behavior_lib_path = os.path.join(output_dir, f"behavior_lib_{case_name}")
        os.makedirs(behavior_lib_path, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"处理案例: {case['action_name']}")
        print(f"图片: {case['image_name']}")
        print(f"行为库路径: {behavior_lib_path}")
        print(f"{'='*80}")
        
        # 对每个案例运行 total_try_times 次
        for try_idx in range(1, total_try_times + 1):
            print(f"\n--- 第 {try_idx}/{total_try_times} 次运行 ---")
            
            # 为每次运行创建单独的行为库路径
            try_behavior_lib_path = os.path.join(behavior_lib_path, f"try_{try_idx}")
            os.makedirs(try_behavior_lib_path, exist_ok=True)
            
            # 为每次运行创建单独的输出目录
            try_output_dir = os.path.join(output_dir, f"{case_name}_try_{try_idx}")
            os.makedirs(try_output_dir, exist_ok=True)
            
            result = process_case_with_image(
                case=case,
                image_path=image_path,
                behavior_lib_path=try_behavior_lib_path,
                llm=llm,
                output_dir=try_output_dir,
                max_feedback_times=max_feedback_times,
                try_idx=try_idx,
                case_name=case_name,
                use_image=use_image
            )
            
            all_results.append(result)
            
            print(f"第 {try_idx} 次运行完成: 成功={result['success']}, 反馈次数={result['feedback_times']}")
        
        # 统计这个案例的成功率
        case_results = [r for r in all_results if r.get("case_name", "").startswith(case_name)]
        success_count = sum(1 for r in case_results if r.get("success", False))
        success_rate = success_count / len(case_results) * 100 if case_results else 0.0
        avg_feedback = sum(r.get("feedback_times", 0) for r in case_results) / len(case_results) if case_results else 0.0
        print(f"\n案例 {case['action_name']} 统计: 成功 {success_count}/{len(case_results)} ({success_rate:.2f}%), 平均反馈次数: {avg_feedback:.2f}")
    
    # 保存结果到CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {csv_file}")
        
        # 统计成功率（按案例分组，计算平均成功率）
        print("\n" + "="*80)
        print("成功率统计（每个案例运行 {} 次）".format(total_try_times))
        print("="*80)
        
        # 按案例分组统计
        from collections import defaultdict
        case_groups = defaultdict(list)
        for result in all_results:
            case_key = result.get("case_name", result.get("image_name", "unknown"))
            case_groups[case_key].append(result)
        
        # 按案例统计
        case_success_rates = []
        for case_name, case_results in sorted(case_groups.items()):
            success_count = sum(1 for r in case_results if r.get("success", False))
            total_runs = len(case_results)
            case_success_rate = success_count / total_runs * 100 if total_runs > 0 else 0.0
            case_success_rates.append(case_success_rate)
            avg_feedback = sum(r.get("feedback_times", 0) for r in case_results) / total_runs if total_runs > 0 else 0.0
            
            # 统计反馈次数为 0 时的成功率
            no_feedback_results = [r for r in case_results if r.get("feedback_times", 0) == 0]
            no_feedback_success = sum(1 for r in no_feedback_results if r.get("success", False))
            no_feedback_total = len(no_feedback_results)
            no_feedback_success_rate = no_feedback_success / no_feedback_total * 100 if no_feedback_total > 0 else 0.0
            
            print(f"  {case_name}: 成功 {success_count}/{total_runs} ({case_success_rate:.2f}%), 平均反馈次数: {avg_feedback:.2f}")
            print(f"    - 反馈次数=0时的成功率: {no_feedback_success}/{no_feedback_total} ({no_feedback_success_rate:.2f}%)")
            print(f"    - 包含反馈的整体成功率: {success_count}/{total_runs} ({case_success_rate:.2f}%)")
        
        # 总体统计
        avg_success_rate_all = sum(case_success_rates) / len(case_success_rates) if case_success_rates else 0.0
        total_runs_all = len(all_results)
        total_success_all = sum(1 for r in all_results if r.get("success", False))
        avg_feedback_all = sum(r.get("feedback_times", 0) for r in all_results) / total_runs_all if total_runs_all > 0 else 0.0
        
        # 统计总体反馈次数为 0 时的成功率
        no_feedback_all = [r for r in all_results if r.get("feedback_times", 0) == 0]
        no_feedback_success_all = sum(1 for r in no_feedback_all if r.get("success", False))
        no_feedback_total_all = len(no_feedback_all)
        no_feedback_success_rate_all = no_feedback_success_all / no_feedback_total_all * 100 if no_feedback_total_all > 0 else 0.0
        
        print(f"\n总体统计:")
        print(f"  案例数: {len(case_groups)}")
        print(f"  总运行次数: {total_runs_all}")
        print(f"  总成功次数: {total_success_all}")
        print(f"  按运行次数计算成功率: {total_success_all}/{total_runs_all} ({total_success_all/total_runs_all*100:.2f}%)")
        print(f"  按案例平均成功率: {avg_success_rate_all:.2f}%")
        print(f"  平均反馈次数: {avg_feedback_all:.2f}")
        print(f"\n  反馈次数=0时的成功率: {no_feedback_success_all}/{no_feedback_total_all} ({no_feedback_success_rate_all:.2f}%)")
        print(f"  包含反馈的整体成功率: {total_success_all}/{total_runs_all} ({total_success_all/total_runs_all*100:.2f}%)")
        
        # 打印摘要（按案例分组显示）
        print("\n" + "="*80)
        print("处理摘要")
        print("="*80)
        for case_name, case_results in sorted(case_groups.items()):
            print(f"\n案例: {case_name}")
            for result in sorted(case_results, key=lambda x: x.get('try_idx', 0)):
                status = "✓" if result.get('success', False) else "✗"
                print(f"  运行 {result.get('try_idx', 0)}: {status} 成功={result.get('success', False)}, 反馈次数={result.get('feedback_times', 0)}")
            success_count = sum(1 for r in case_results if r.get('success', False))
            print(f"  总计: {success_count}/{len(case_results)} 成功")
        
        # 生成汇总表格（可复制到CSV）
        print("\n" + "="*80)
        print("汇总统计表格（可复制到CSV）")
        print("="*80)
        
        # 收集汇总数据
        summary_data = []
        # 建立image_name到action_name的映射
        image_to_action = {}
        for case in filtered_all_cases:
            image_name = case.get("image_name", "")
            action_name = case.get("action_name", image_name)
            image_to_action[image_name] = action_name
        
        for case_name, case_results in sorted(case_groups.items()):
            # 获取案例的action_name（用于显示）
            # 从case_results中获取image_name，然后查找对应的action_name
            case_image_name = case_results[0].get("image_name", "")
            case_action_name = image_to_action.get(case_image_name, case_image_name)
            if not case_action_name or case_action_name == case_image_name:
                # 如果找不到，使用case_name作为后备
                case_action_name = case_name
            
            # 统计反馈=0且成功的个数
            no_feedback_results = [r for r in case_results if r.get("feedback_times", 0) == 0]
            no_feedback_success = sum(1 for r in no_feedback_results if r.get("success", False))
            no_feedback_total = len(no_feedback_results)
            no_feedback_success_ratio = f"{no_feedback_success}/{no_feedback_total}" if no_feedback_total > 0 else "0/0"
            no_feedback_success_rate = no_feedback_success / no_feedback_total * 100 if no_feedback_total > 0 else 0.0
            
            # 统计总的成功个数
            total_success = sum(1 for r in case_results if r.get("success", False))
            total_runs = len(case_results)
            total_success_ratio = f"{total_success}/{total_runs}"
            total_success_rate = total_success / total_runs * 100 if total_runs > 0 else 0.0
            
            summary_data.append({
                "案例名称": case_action_name,
                "反馈=0且成功": no_feedback_success_ratio,
                "反馈=0且成功占比(%)": f"{no_feedback_success_rate:.2f}",
                "总成功": total_success_ratio,
                "总成功占比(%)": f"{total_success_rate:.2f}"
            })
        
        # 输出CSV格式的表格
        print("\nCSV格式（可直接复制）:")
        print("-" * 80)
        # 表头
        header = "案例名称,反馈=0且成功,反馈=0且成功占比(%),总成功,总成功占比(%)"
        print(header)
        # 数据行
        for row in summary_data:
            csv_row = f"{row['案例名称']},{row['反馈=0且成功']},{row['反馈=0且成功占比(%)']},{row['总成功']},{row['总成功占比(%)']}"
            print(csv_row)
        print("-" * 80)
        
        # 保存汇总表格到CSV文件
        summary_df = pd.DataFrame(summary_data)
        summary_csv_file = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_csv_file, index=False, encoding='utf-8-sig')
        print(f"\n汇总表格已保存到: {summary_csv_file}")
        
        # 输出格式化的表格（便于阅读）
        print("\n格式化表格:")
        print("-" * 80)
        print(f"{'案例名称':<30} {'反馈=0且成功':<20} {'占比(%)':<12} {'总成功':<15} {'占比(%)':<12}")
        print("-" * 80)
        for row in summary_data:
            print(f"{row['案例名称']:<30} {row['反馈=0且成功']:<20} {row['反馈=0且成功占比(%)']:<12} {row['总成功']:<15} {row['总成功占比(%)']:<12}")
        print("-" * 80)
        
    else:
        print("\n没有成功处理的案例")

if __name__ == "__main__":
    main()

