# from btgym.llm.llm import LLM
from btgym.llm.llm_gpt import LLM
import re
import pathlib
DIR = pathlib.Path(__file__).parent
import os
from exps_bt_learning.tools import parse_bddl,build_prompt,extract_code
from exps_bt_learning.validate_bt_fun import validate_bt_fun
from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
import time


######################################
# 1. 输入: bddl, goal, objects, start_state
# 2. 输出: 行为库 behavior_lib_path
######################################

def llm_generate_behavior_lib(bddl_file=None,goal_str=None,goal_str_list=None,objects=None,start_state=None,behavior_lib_path=None,model="gpt-4o",clear_lib=True,save_io_dir=None):
    """
    生成行为库
    Args:
        goal_str: 单个goal字符串（向后兼容）
        goal_str_list: goal列表，包含三个goal [easy, medium, hard]
        save_io_dir: 保存输入输出的目录
    """
    
    ########################
    # 1. set input BDDL file 设置输入 BDDL 文件
    ########################
    # bddl_file = os.path.join(DIR,"../tasks/task1/problem0.bddl")
    # objects, start_state, goal = parse_bddl(bddl_file)
    # # 把 goal 转换为字符串
    # goal_str = ' '.join(goal)
    # print("objects:",objects)
    # print("start_state:",start_state)
    # print("goal_str:",goal_str)

    # start_state.update({'IsHandEmpty()'})
    # # goal_str = 'IsHolding(apple)'
    # goal_str = 'On(apple,coffee_table)'

        

    # behavior_lib_path = os.path.join(DIR,"../exec_lib")
    # 先收集所有 lib
    # behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
    

    
    # 确保 behavior_lib_path 目录存在
    os.makedirs(behavior_lib_path, exist_ok=True)
    
    # 清空 behavior_lib_path Action 和 Condition 目录下的所有 .py 文件
    action_dir = os.path.join(behavior_lib_path, "Action")
    condition_dir = os.path.join(behavior_lib_path, "Condition")
    
    # 确保目录存在
    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(condition_dir, exist_ok=True)
 
    if clear_lib:
        # 清空 Action 目录
        if os.path.exists(action_dir):
            for file in os.listdir(action_dir):
                file_path = os.path.join(action_dir, file)
                if os.path.isfile(file_path) and file.endswith('.py'):
                    os.remove(file_path)
        
        # 清空 Condition 目录
        if os.path.exists(condition_dir):
            for file in os.listdir(condition_dir):
                file_path = os.path.join(condition_dir, file)
                if os.path.isfile(file_path) and file.endswith('.py'):
                    os.remove(file_path)
    
        print(f"Cleared {action_dir} and {condition_dir}")
    ########################
    # 2. call llm to generate behavior lib 调用大模型生成行为库
    ########################
    llm = LLM(request_model=model) #gpt-3.5-turbo
    
    # 如果提供了goal_str_list，构建包含三个goal的prompt
    if goal_str_list is not None and len(goal_str_list) == 3:
        # 按照示例格式拼接：start_state、objects、goal 都使用 Python 代码格式
        # 格式化 start_state 为 Python 集合格式
        start_state_str = "{" + ", ".join([f'"{s}"' for s in start_state]) + "}"
        # 格式化 objects 为 Python 集合格式
        objects_str = "{" + ", ".join([f'"{o}"' for o in objects]) + "}"
        # 格式化 goal 为 Python 列表格式
        goal_list_str = "[" + ", ".join([f'"{g}"' for g in goal_str_list]) + "]"
        goal_str_combined = f"start_state = {start_state_str}\nobjects = {objects_str}\ngoal = {goal_list_str}"
        prompt = build_prompt(goal=goal_str_combined,objects=objects,start_state=start_state)
    else:
        # 向后兼容，使用单个goal_str
        prompt = build_prompt(goal=goal_str,objects=objects,start_state=start_state)
    
    # 保存输入
    if save_io_dir is not None:
        os.makedirs(save_io_dir, exist_ok=True)
        input_file = os.path.join(save_io_dir, "llm_input.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("=== LLM Input ===\n\n")
            f.write(f"Objects: {objects}\n\n")
            f.write(f"Start State: {start_state}\n\n")
            if goal_str_list is not None:
                f.write(f"Easy Goal: {goal_str_list[0]}\n")
                f.write(f"Medium Goal: {goal_str_list[1]}\n")
                f.write(f"Hard Goal: {goal_str_list[2]}\n\n")
            else:
                f.write(f"Goal: {goal_str}\n\n")
            f.write("=== Prompt ===\n\n")
            f.write(prompt)
        print(f"Saved LLM input to {input_file}")
    
    # 蓝色打印
    print("\033[94m",f"Requesting llm...","\033[0m")
    start_time = time.time()
    answer = llm.request(prompt)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    # 绿色打印
    print("\033[92m",answer,"\033[0m")
    
    # 保存输出
    if save_io_dir is not None:
        output_file = os.path.join(save_io_dir, "llm_output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== LLM Output ===\n\n")
            f.write(answer)
        print(f"Saved LLM output to {output_file}")

    # 提取所有 Python 代码块
    # 把对应的 python 代码都提取到一个列表中,然后分别在对应的目录下创建动作类py文件
    code_blocks_pattern = r"```python\n(.*?)```"
    code_blocks = re.findall(code_blocks_pattern, answer, re.DOTALL)

    # 解析每个代码块
    class_pattern = r'class\s+(\w+)\s*\((\w+)\):'
    for code in code_blocks:
        # 提取类定义
        matches = re.finditer(class_pattern, code) # 迭代器是一次性的
        for match in matches:
            class_name = match.group(1)
            base_class = match.group(2)
            
            # 确定类的类型
            # 从 behavior_lib_path 中提取到 a_exp1_llm_bt_tasks 的路径
            # 例如: a_exp1_llm_bt_tasks/task2/exec_lib -> a_exp1_llm_bt_tasks
            rel_path = os.path.relpath(behavior_lib_path, DIR)
            # 移除 task 名称和 exec_lib 部分
            path_parts = rel_path.split(os.path.sep)
            if 'a_exp1_llm_bt_tasks' in path_parts:
                idx = path_parts.index('a_exp1_llm_bt_tasks')
                _lib_path = '.'.join(path_parts[:idx+1])  # 只保留到 a_exp1_llm_bt_tasks
            else:
                # 向后兼容：如果路径中没有 a_exp1_llm_bt_tasks，使用原来的逻辑
                _lib_path = rel_path.replace(os.path.sep, '.')
            
            if base_class == 'OGAction':
                class_type = "Action"
                # _base 目录在 a_exp1_llm_bt_tasks 下，不在 exec_lib 下
                import_statement = f"from exps_bt_learning.{_lib_path}._base.OGAction import OGAction\n\n"
            elif base_class == 'OGCondition':
                class_type = "Condition"
                import_statement = f"from exps_bt_learning.{_lib_path}._base.OGCondition import OGCondition\n\n"
            else:
                continue  # 如果基类不匹配，跳过这个类
            
            # 找到类定义的起始位置
            start_index = code.find(f"class {class_name}")
            
            # 找到下一个类定义的起始位置，或者代码块的结束位置
            end_index = code.find("class ", start_index + 1)
            if end_index == -1:
                end_index = len(code)
            
            # 提取单个类的代码
            single_class_code = code[start_index:end_index].strip()
            
            # 删除代码中原有的关于  from 和 import 的语句
            single_class_code = re.sub(r'from\s+\w+\s+import\s+\w+', '', single_class_code)
            
            # 创建目录（如果不存在）
            class_dir = os.path.join(behavior_lib_path, class_type)
            os.makedirs(class_dir, exist_ok=True)
            
            # 写入 .py 文件，确保导入语句在文件开头
            file_path = os.path.join(class_dir, f"{class_name}.py")
            with open(file_path, "w") as f:
                f.write(import_statement + single_class_code)

            # 蓝色打印
            print("\033[94m",f"Written {class_name} to {file_path}","\033[0m")
 


def llm_generate_behavior_lib_need_feedback(bddl_file,goal_str,objects,start_state,behavior_lib_path,llm,messages=None,clear_lib=True):

    # 确保 behavior_lib_path 目录存在
    os.makedirs(behavior_lib_path, exist_ok=True)
    
    # 清空 behavior_lib_path Action 和 Condition 目录下的所有 .py 文件
    action_dir = os.path.join(behavior_lib_path, "Action")
    condition_dir = os.path.join(behavior_lib_path, "Condition")
    
    # 确保目录存在
    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(condition_dir, exist_ok=True)
 
    if clear_lib:
        # 清空 Action 目录
        if os.path.exists(action_dir):
            for file in os.listdir(action_dir):
                file_path = os.path.join(action_dir, file)
                if os.path.isfile(file_path) and file.endswith('.py'):
                    os.remove(file_path)
        
        # 清空 Condition 目录
        if os.path.exists(condition_dir):
            for file in os.listdir(condition_dir):
                file_path = os.path.join(condition_dir, file)
                if os.path.isfile(file_path) and file.endswith('.py'):
                    os.remove(file_path)
    
        print(f"Cleared {action_dir} and {condition_dir}")
    ########################
    # 2. call llm to generate behavior lib 调用大模型生成行为库
    ########################
    if messages is None:
        prompt = build_prompt(goal=goal_str,objects=objects)
        messages = []
        messages.append({"role": "user", "content": prompt}) 
        # 蓝色打印
        print("\033[94m",f"Requesting llm...","\033[0m")
        start_time = time.time()
        answer = llm.request_instruction(prompt)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        # 绿色打印
        print("\033[92m",answer,"\033[0m")
        messages.append({"role": "assistant", "content": answer}) 
    else:
        # 蓝色打印
        print("\033[94m",f"Requesting llm...","\033[0m")
        start_time = time.time()
        answer = llm.request_instruction_with_history(messages)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        # 绿色打印
        print("\033[92m",answer,"\033[0m")
        messages.append({"role": "assistant", "content": answer}) 
    

    # 提取所有 Python 代码块
    # 把对应的 python 代码都提取到一个列表中,然后分别在对应的目录下创建动作类py文件
    code_blocks_pattern = r"```python\n(.*?)```"
    code_blocks = re.findall(code_blocks_pattern, answer, re.DOTALL)

    # 解析每个代码块
    class_pattern = r'class\s+(\w+)\s*\((\w+)\):'
    for code in code_blocks:
        # 提取类定义
        matches = re.finditer(class_pattern, code) # 迭代器是一次性的
        for match in matches:
            class_name = match.group(1)
            base_class = match.group(2)
            
            # 确定类的类型
            # 从 behavior_lib_path 中提取到 a_exp1_llm_bt_tasks 的路径
            # 例如: a_exp1_llm_bt_tasks/task2/exec_lib -> a_exp1_llm_bt_tasks
            rel_path = os.path.relpath(behavior_lib_path, DIR)
            # 移除 task 名称和 exec_lib 部分
            path_parts = rel_path.split(os.path.sep)
            if 'a_exp1_llm_bt_tasks' in path_parts:
                idx = path_parts.index('a_exp1_llm_bt_tasks')
                _lib_path = '.'.join(path_parts[:idx+1])  # 只保留到 a_exp1_llm_bt_tasks
            else:
                # 向后兼容：如果路径中没有 a_exp1_llm_bt_tasks，使用原来的逻辑
                _lib_path = rel_path.replace(os.path.sep, '.')
            
            if base_class == 'OGAction':
                class_type = "Action"
                # _base 目录在 a_exp1_llm_bt_tasks 下，不在 exec_lib 下
                import_statement = f"from exps_bt_learning.{_lib_path}._base.OGAction import OGAction\n\n"
            elif base_class == 'OGCondition':
                class_type = "Condition"
                import_statement = f"from exps_bt_learning.{_lib_path}._base.OGCondition import OGCondition\n\n"
            else:
                continue  # 如果基类不匹配，跳过这个类
            
            # 找到类定义的起始位置
            start_index = code.find(f"class {class_name}")
            
            # 找到下一个类定义的起始位置，或者代码块的结束位置
            end_index = code.find("class ", start_index + 1)
            if end_index == -1:
                end_index = len(code)
            
            # 提取单个类的代码
            single_class_code = code[start_index:end_index].strip()
            
            # 删除代码中原有的关于  from 和 import 的语句
            single_class_code = re.sub(r'from\s+\w+\s+import\s+\w+', '', single_class_code)
            
            # 创建目录（如果不存在）
            # 如果目录存在，给动作类添加一个后缀，避免文件名冲突和覆盖
            class_dir = os.path.join(behavior_lib_path, class_type)
            os.makedirs(class_dir, exist_ok=True)
            
            # 确定文件路径并避免文件名冲突
            file_path = os.path.join(class_dir, f"{class_name}.py")
            counter = 1
            while os.path.exists(file_path):
                file_path = os.path.join(class_dir, f"{class_name}{counter}.py")
                # 同时把 single_class_code 中的 class_name 也替换为 class_name_{counter}
                # class_name 的格式为：class_name(base_class)
                # class_name 通常在   class class_name(OGAction) 中
                if class_type == "Action":
                    single_class_code = re.sub(r'class\s+(\w+)\s*\(\s*OGAction\s*\):', f'class {class_name}{counter}(OGAction):', single_class_code)
                counter += 1
            
            # 写入 .py 文件，确保导入语句在文件开头
            with open(file_path, "w") as f:
                f.write(import_statement + single_class_code)
            # 蓝色打印
            print("\033[94m",f"Written {class_name} to {file_path}","\033[0m")
    
    return messages


# def llm_generate_behavior_lib(bddl_file,goal_str,objects,start_state,behavior_lib_path,model="gpt-4o"):
    
#     ########################
#     # 1. set input BDDL file 设置输入 BDDL 文件
#     ########################
#     # bddl_file = os.path.join(DIR,"../tasks/task1/problem0.bddl")
#     # objects, start_state, goal = parse_bddl(bddl_file)
#     # # 把 goal 转换为字符串
#     # goal_str = ' '.join(goal)
#     # print("objects:",objects)
#     # print("start_state:",start_state)
#     # print("goal_str:",goal_str)

#     # start_state.update({'IsHandEmpty()'})
#     # # goal_str = 'IsHolding(apple)'
#     # goal_str = 'On(apple,coffee_table)'

        

#     # behavior_lib_path = os.path.join(DIR,"../exec_lib")
#     # 先收集所有 lib
#     # behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
    

    
#     # 清空 behavior_lib_path Action 和 Condition 目录下的所有 .py 文件
#     action_dir = os.path.join(behavior_lib_path, "Action")
#     condition_dir = os.path.join(behavior_lib_path, "Condition")
    
#     for file in os.listdir(action_dir):
#         file_path = os.path.join(action_dir, file)
#         if os.path.isfile(file_path) and file.endswith('.py'):
#             os.remove(file_path)
    
#     for file in os.listdir(condition_dir):
#         file_path = os.path.join(condition_dir, file)
#         if os.path.isfile(file_path) and file.endswith('.py'):
#             os.remove(file_path)

#     print(f"Cleared {action_dir} and {condition_dir}")
#     ########################
#     # 2. call llm to generate behavior lib 调用大模型生成行为库
#     ########################
#     llm = LLM(model=model) #gpt-3.5-turbo
#     prompt = build_prompt(goal=goal_str,objects=objects)
#     # 蓝色打印
#     print("\033[94m",f"Requesting llm...","\033[0m")
#     start_time = time.time()
#     answer = llm.request_instruction(prompt)
#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time:.2f} seconds")
#     # 绿色打印
#     print("\033[92m",answer,"\033[0m")

#     # 提取所有 Python 代码块
#     # 把对应的 python 代码都提取到一个列表中,然后分别在对应的目录下创建动作类py文件
#     code_blocks_pattern = r"```python\n(.*?)```"
#     code_blocks = re.findall(code_blocks_pattern, answer, re.DOTALL)

#     # 解析每个代码块
#     class_pattern = r'class\s+(\w+)\s*\((\w+)\):'
#     for code in code_blocks:
#         # 提取类定义
#         matches = re.finditer(class_pattern, code) # 迭代器是一次性的
#         for match in matches:
#             class_name = match.group(1)
#             base_class = match.group(2)
            
#             # 确定类的类型
#             _lib_path = os.path.relpath(behavior_lib_path, DIR).replace(os.path.sep, '.')
#             if base_class == 'OGAction':
#                 class_type = "Action"
#                 # 把 behavior_lib_path 转换为相对路径
#                 import_statement = f"from exps_bt_learning.{_lib_path}._base.OGAction import OGAction\n\n"
#             elif base_class == 'OGCondition':
#                 class_type = "Condition"
#                 import_statement = f"from exps_bt_learning.{_lib_path}._base.OGCondition import OGCondition\n\n"
#             else:
#                 continue  # 如果基类不匹配，跳过这个类
            
#             # 找到类定义的起始位置
#             start_index = code.find(f"class {class_name}")
            
#             # 找到下一个类定义的起始位置，或者代码块的结束位置
#             end_index = code.find("class ", start_index + 1)
#             if end_index == -1:
#                 end_index = len(code)
            
#             # 提取单个类的代码
#             single_class_code = code[start_index:end_index].strip()
            
#             # 删除代码中原有的关于  from 和 import 的语句
#             single_class_code = re.sub(r'from\s+\w+\s+import\s+\w+', '', single_class_code)
            
#             # 创建目录（如果不存在）
#             class_dir = os.path.join(behavior_lib_path, class_type)
#             os.makedirs(class_dir, exist_ok=True)
            
#             # 写入 .py 文件，确保导入语句在文件开头
#             file_path = os.path.join(class_dir, f"{class_name}.py")
#             with open(file_path, "w") as f:
#                 f.write(import_statement + single_class_code)

#             # 蓝色打印
#             print("\033[94m",f"Written {class_name} to {file_path}","\033[0m")
 
 
 
if __name__ == "__main__":
    
    task_name = "task1"

    bddl_file = os.path.join(DIR,f"tasks/{task_name}/problem0.bddl")
    behavior_lib_path = os.path.join(DIR,f"tasks/{task_name}/exec_lib")  # os.path.join(DIR,"../exec_lib")
    
    
    objects, start_state, goal = parse_bddl(bddl_file)
    # 把 goal 转换为字符串
    goal_str = ' '.join(goal)
    print("objects:",objects)
    print("start_state:",start_state)
    print("goal_str:",goal_str)
    
    start_state.update({'IsHandEmpty()'})
    # goal_str = 'IsHolding(apple)'
    goal_str = 'On(apple,coffee_table)'

    

    llm_generate_behavior_lib(bddl_file=bddl_file,goal_str=goal_str,objects=objects,start_state=start_state,behavior_lib_path=behavior_lib_path)
        
    ########################
    # 3. run planning algorithm and validate behavior lib 运行规划算法并验证行为库
    ########################
    output_dir = os.path.join(DIR,"./tree.btml")
    error,bt = validate_bt_fun(behavior_lib_path=behavior_lib_path, goal_str=goal_str,cur_cond_set=start_state,output_dir=output_dir)


