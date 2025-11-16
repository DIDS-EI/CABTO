import re
import os
DIR = os.path.dirname(os.path.abspath(__file__))

def parse_bddl(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # 提取 objects 的完整标识符
    objects_match = re.search(r'\(:objects(.*?)\)', content, re.DOTALL)
    full_objects = []
    first_words = []
    if objects_match:
        # 提取每一行的完整标识符
        full_objects = re.findall(r'\b[\w.]+\b(?= -)', objects_match.group(1))
        
        # 提取每个标识符的第一个单词
        first_words = [obj.split('.')[0] for obj in full_objects]
        # 去除所有的_
        first_words = [word.replace('_', '') for word in first_words]
    objects = first_words

    # 提取 init (start state)
    init_match = re.search(r'\(:init(.*?)\)', content, re.DOTALL)
    start_state = []
    if init_match:
        for state in init_match.group(1).split('\n'):
            state = state.strip()
            if state:
                # 使用正则表达式提取函数名和参数
                match = re.match(r'\((\w+)\s+([\w.]+)\s+([\w.]+)', state)
                if match:
                    func_name = match.group(1)
                    param1 = match.group(2).split('.')[0]  # 提取第一个参数的第一个单词
                    if match.group(3):
                        param2 = match.group(3).split('.')[0]  # 提取第二个参数的第一个单词
                        start_state.append(f"{func_name}({param1},{param2})")
                    else:
                        start_state.append(f"{func_name}({param1})")

    # 提取 goal
    goal_match = re.search(r'\(:goal(.*?)\)', content, re.DOTALL)
    goal = []
    if goal_match:
        for g in goal_match.group(1).split('\n'):
            g = g.strip()
            if g:
                # 使用正则表达式提取函数名和参数
                match = re.match(r'\((\w+)\s+([\w.]+)\s+([\w.]+)', g)
                if match:
                    func_name = match.group(1)
                    param1 = match.group(2).split('.')[0]  # 提取第一个参数的第一个单词
                    if match.group(3):
                        param2 = match.group(3).split('.')[0]  # 提取第二个参数的第一个单词
                        goal.append(f"{func_name}({param1},{param2})")
                    else:
                        goal.append(f"{func_name}({param1})")


    return set(objects), set(start_state), set(goal)


def build_prompt(goal,objects,initial_state):
    prompt_example_path = os.path.join(DIR,"./prompt_generate_libs_examples.txt")
    prompt_template_path = os.path.join(DIR,"./prompt_generate_libs.txt")
    with open(prompt_example_path, 'r') as file:
        example_content = file.read()
    with open(prompt_template_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 移除被 ''' 包围的部分（不解析作为大模型的输入）
    # 使用正则表达式匹配和移除被 ''' 包围的内容
    content = re.sub(r"'''[\s\S]*?'''", "", content)
    
    # 检查 goal 是否已经包含完整格式化的信息（包含 initial_state = 和 objects =）
    if ("initial_state = " in goal or "start_state = " in goal) and "objects = " in goal:
        # goal 已经包含完整信息，直接使用
        # 需要替换模板中的 {objects} 占位符（如果存在）
        if not isinstance(objects, str):
            objects_str = ', '.join(objects)
        else:
            objects_str = objects
        content = content.format(goal=goal, objects=objects_str)
        instruction = goal  # 直接使用 goal，因为它已经包含了所有信息
    else:
        # 向后兼容：使用旧的格式
        # 如果 objects 不是字符串，则转换为字符串
        if not isinstance(objects, str):
            objects_str = ', '.join(objects)
        else:
            objects_str = objects
        content = content.format(goal=goal, objects=objects_str)
        instruction = f"initial_state: {initial_state}\nobjects: {objects_str}\ngoal: {goal}"
    
    return content +"\n"+ example_content +"\n \
        Here are the initial_state, objects and goals, please generate the behavior libraries:\n"+ instruction

def extract_code(answer,file_path):
    # 使用正则表达式提取代码
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, answer, re.DOTALL) #用于匹配以 python 开头和 结尾的代码块
    if match:
        answer = match.group(1).strip()
    else:
        raise ValueError("No code found in the answer")
    return answer
    # with open(file_path, "w") as f:
    #     f.write(answer)
    
    


if __name__ == "__main__":      
    # 使用示例
    bddl_file = os.path.join(DIR, "a_exp1_llm_bt/problem0.bddl")
    objects, start_state, goal = parse_bddl(bddl_file)

    print("Objects:", objects)
    print("Start State:", start_state)
    print("Goal:", goal)