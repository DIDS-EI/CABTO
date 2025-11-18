import os
import base64
import pandas as pd
from datetime import datetime
from btgym.llm.llm_gpt import LLM
from openai import OpenAI

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def encode_image(image_path):
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def build_correct_action_prompt(action_name, current_pre, current_add, current_del):
    """
    构建修正动作的prompt
    
    Args:
        action_name: 动作名称，如 "Stack(obj1, obj2)"
        current_pre: 当前的前置条件集合
        current_add: 当前的添加效果集合
        current_del: 当前的删除效果集合
        error_type: 错误类型，如 "missing" 或 "incorrect"
        error_description: 错误描述，如 "Pre : (clear y) missing"
    """
    # 将集合转换为列表用于显示
    pre_list = list(current_pre) if isinstance(current_pre, set) else current_pre
    add_list = list(current_add) if isinstance(current_add, set) else current_add
    del_list = list(current_del) if isinstance(current_del, set) else current_del
    
    prompt = f"""你是一个动作效果修正专家。一个动作执行失败了，需要你观察图片，分析失败原因，并修正动作的前置条件（Pre）、添加效果（Add）和删除效果（Del）。

**动作信息：**
- 动作名称：{action_name}
- 当前前置条件（Pre）：{pre_list}
- 当前添加效果（Add）：{add_list}
- 当前删除效果（Del）：{del_list}   

**任务要求：**
1. 仔细观察图片中的场景，理解动作执行失败的原因
2. 分析当前Pre、Add、Del中存在的问题
3. 输出修正后的正确的Pre、Add、Del

**输出格式：**
请严格按照以下JSON格式输出，不要添加任何其他内容：

{{
    "correct_pre": ["条件1", "条件2", ...],
    "correct_add": ["效果1", "效果2", ...],
    "correct_del": ["效果1", "效果2", ...],
    "explanation": "简要说明修正的原因"
}}

**注意事项：**
- Pre条件应该包含动作执行所需的所有前置条件
- Add效果应该包含动作执行后新增的所有状态
- Del效果应该包含动作执行后删除的所有状态
- 确保Pre、Add、Del的逻辑正确性和完整性
"""
    return prompt

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

def parse_llm_response(response_text):
    """
    解析LLM的响应，提取correct_pre, correct_add, correct_del
    
    Args:
        response_text: LLM返回的文本
    
    Returns:
        dict: 包含correct_pre, correct_add, correct_del的字典
    """
    import json
    import re
    
    result = {
        "correct_pre": set(),
        "correct_add": set(),
        "correct_del": set(),
        "explanation": ""
    }
    
    # 首先尝试提取代码块中的JSON（如果LLM用```json包装）
    json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
    if json_block_match:
        json_str = json_block_match.group(1)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                # 将列表转换为集合
                if "correct_pre" in parsed and isinstance(parsed["correct_pre"], list):
                    parsed["correct_pre"] = set(parsed["correct_pre"])
                if "correct_add" in parsed and isinstance(parsed["correct_add"], list):
                    parsed["correct_add"] = set(parsed["correct_add"])
                if "correct_del" in parsed and isinstance(parsed["correct_del"], list):
                    parsed["correct_del"] = set(parsed["correct_del"])
                return parsed
        except:
            pass
    
    # 尝试直接解析JSON（查找第一个完整的JSON对象）
    try:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            # 尝试平衡大括号
            brace_count = 0
            end_pos = 0
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            if end_pos > 0:
                json_str = json_str[:end_pos]
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                # 将列表转换为集合
                if "correct_pre" in parsed and isinstance(parsed["correct_pre"], list):
                    parsed["correct_pre"] = set(parsed["correct_pre"])
                if "correct_add" in parsed and isinstance(parsed["correct_add"], list):
                    parsed["correct_add"] = set(parsed["correct_add"])
                if "correct_del" in parsed and isinstance(parsed["correct_del"], list):
                    parsed["correct_del"] = set(parsed["correct_del"])
                return parsed
    except:
        pass
    
    # 如果直接解析失败，尝试手动提取各个字段
    # 提取correct_pre
    pre_match = re.search(r'"correct_pre"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
    if pre_match:
        pre_content = pre_match.group(1)
        # 更智能地解析列表项
        items = re.findall(r'"([^"]+)"', pre_content)
        if not items:
            # 如果没有引号，尝试按逗号分割
            items = [s.strip() for s in pre_content.split(',') if s.strip()]
        result["correct_pre"] = set(items)
    
    # 提取correct_add
    add_match = re.search(r'"correct_add"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
    if add_match:
        add_content = add_match.group(1)
        items = re.findall(r'"([^"]+)"', add_content)
        if not items:
            items = [s.strip() for s in add_content.split(',') if s.strip()]
        result["correct_add"] = set(items)
    
    # 提取correct_del
    del_match = re.search(r'"correct_del"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
    if del_match:
        del_content = del_match.group(1)
        items = re.findall(r'"([^"]+)"', del_content)
        if not items:
            items = [s.strip() for s in del_content.split(',') if s.strip()]
        result["correct_del"] = set(items)
    
    # 提取explanation
    exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response_text, re.DOTALL)
    if exp_match:
        result["explanation"] = exp_match.group(1)
    else:
        # 尝试提取没有引号的explanation
        exp_match = re.search(r'"explanation"\s*:\s*([^,\}]+)', response_text, re.DOTALL)
        if exp_match:
            result["explanation"] = exp_match.group(1).strip().strip('"')
    
    return result


def build_analysis_prompt(action_name, current_pre, current_add, current_del):
    """
    构建analysis类型的prompt（用于前3个案例）
    要求大模型分析pre、add、del是否有错误，并用Python set格式输出
    """
    pre_list = list(current_pre) if isinstance(current_pre, set) else current_pre
    add_list = list(current_add) if isinstance(current_add, set) else current_add
    del_list = list(current_del) if isinstance(current_del, set) else current_del
    
    prompt = f"""动作 {action_name} 执行失败，请你帮忙分析 pre、add 和 del 是否有错误，重新生成 pre、add 和 del，用 set 的 Python 格式给出。

**当前动作定义：**
- 动作名称：{action_name}
- 当前前置条件（pre）：{pre_list}
- 当前添加效果（add）：{add_list}
- 当前删除效果（del）：{del_list}

**任务要求：**
1. 仔细观察图片中的场景，理解动作执行失败的原因
2. 分析当前 pre、add、del 中存在的问题
3. 输出修正后的正确的 pre、add、del

**输出格式：**
请严格按照以下 Python 代码格式输出，不要添加任何其他内容：

```python
pre = {{
    "条件1",
    "条件2",
    ...
}}

add = {{
    "效果1",
    "效果2",
    ...
}}

del_set = {{
    "效果1",
    "效果2",
    ...
}}
```

**示例：**
```python
pre = {{
    "Holding(apple)",
    "IsOpen(cabinet)"
}}

add = {{
    "In(apple,cabinet)"
}}

del_set = {{
    "Holding(apple)"
}}
```

**注意事项：**
- pre 条件应该包含动作执行所需的所有前置条件
- add 效果应该包含动作执行后新增的所有状态
- del_set 效果应该包含动作执行后删除的所有状态
- 确保 pre、add、del_set 的逻辑正确性和完整性
- 必须使用 Python set 格式（用大括号 {{}} 包围，每个元素用引号括起来）
"""
    return prompt

def build_judgment_prompt(action_name, current_pre, current_add, current_del):
    """
    构建判断类型的prompt（用于后2个案例）
    要求大模型分析pre、add、del是否有错误，并用Python set格式输出
    与analysis类型的区别：第一句是"执行完成"而不是"执行失败"
    """
    pre_list = list(current_pre) if isinstance(current_pre, set) else current_pre
    add_list = list(current_add) if isinstance(current_add, set) else current_add
    del_list = list(current_del) if isinstance(current_del, set) else current_del
    
    prompt = f"""动作 {action_name} 执行完成，请你帮忙分析 pre、add 和 del 是否有错误，重新生成 pre、add 和 del，用 set 的 Python 格式给出。

**当前动作定义：**
- 动作名称：{action_name}
- 当前前置条件（pre）：{pre_list}
- 当前添加效果（add）：{add_list}
- 当前删除效果（del）：{del_list}

**任务要求：**
1. 仔细观察图片中的场景，理解动作执行完成的情况
2. 分析当前 pre、add、del 中是否存在问题
3. 输出修正后的正确的 pre、add、del

**输出格式：**
请严格按照以下 Python 代码格式输出，不要添加任何其他内容：

```python
pre = {{
    "条件1",
    "条件2",
    ...
}}

add = {{
    "效果1",
    "效果2",
    ...
}}

del_set = {{
    "效果1",
    "效果2",
    ...
}}
```

**示例：**
```python
pre = {{
    "Holding(apple)",
    "IsOpen(cabinet)"
}}

add = {{
    "In(apple,cabinet)"
}}

del_set = {{
    "Holding(apple)"
}}
```

**注意事项：**
- pre 条件应该包含动作执行所需的所有前置条件
- add 效果应该包含动作执行后新增的所有状态
- del_set 效果应该包含动作执行后删除的所有状态
- 确保 pre、add、del_set 的逻辑正确性和完整性
- 必须使用 Python set 格式（用大括号 {{}} 包围，每个元素用引号括起来）
"""
    return prompt

def parse_python_set_response(response_text):
    """
    解析大模型返回的Python set格式的输出
    
    Args:
        response_text: LLM返回的文本
    
    Returns:
        dict: 包含correct_pre, correct_add, correct_del的字典（都是set类型）
    """
    import re
    import ast
    
    result = {
        "correct_pre": set(),
        "correct_add": set(),
        "correct_del": set()
    }
    
    # 尝试提取代码块中的Python代码
    code_block_match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', response_text)
    if code_block_match:
        code_str = code_block_match.group(1)
    else:
        # 如果没有代码块，尝试直接查找 pre = {...} 的模式
        code_str = response_text
    
    def extract_set(var_name, text):
        """提取指定变量的set值"""
        # 匹配 var_name = {...}，支持多行和嵌套
        pattern = rf'{var_name}\s*=\s*(\{{)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            start_pos = match.end(1)  # 找到 { 的位置
            # 从 { 开始，找到匹配的 }
            brace_count = 1
            end_pos = start_pos
            for i in range(start_pos, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > start_pos:
                set_str = text[match.start():end_pos]
                # 提取 = 后面的部分
                eq_pos = set_str.find('=')
                if eq_pos >= 0:
                    set_str = set_str[eq_pos+1:].strip()
                    try:
                        # 使用ast.literal_eval安全地解析
                        parsed_set = ast.literal_eval(set_str)
                        if isinstance(parsed_set, set):
                            return parsed_set
                    except:
                        # 如果解析失败，尝试手动提取字符串
                        items = re.findall(r'"([^"]+)"', set_str)
                        return set(items)
        return set()
    
    # 提取各个set
    result["correct_pre"] = extract_set("pre", code_str)
    result["correct_add"] = extract_set("add", code_str)
    result["correct_del"] = extract_set("del_set", code_str)
    
    return result

def extract_condition_args(condition_str):
    """
    提取条件的参数部分
    
    Args:
        condition_str: 条件字符串，如 "Clear(green)" 或 "OnTable(green)"
    
    Returns:
        tuple: (predicate, args) 或 None
    """
    import re
    match = re.match(r'(\w+)\((.*?)\)', condition_str)
    if match:
        return match.group(1), match.group(2)
    return None

def build_args_to_predicates(condition_set):
    """
    构建从参数到谓词集合的映射
    自动识别同义词：如果多个条件有相同的参数但不同的谓词，它们被认为是同义词
    
    Args:
        condition_set: 条件集合
    
    Returns:
        dict: {args: set(predicates)} 映射
    """
    args_to_predicates = {}
    for condition in condition_set:
        parsed = extract_condition_args(condition)
        if parsed:
            predicate, args = parsed
            if args not in args_to_predicates:
                args_to_predicates[args] = set()
            args_to_predicates[args].add(predicate)
    return args_to_predicates

def conditions_semantically_equivalent(condition1, condition2):
    """
    检查两个条件是否语义等价
    支持多种等价模式：
    1. 参数完全相同，谓词不同（如 Clear(green) 和 OnTable(green)）
    2. 语义等价（如 Clear(green) 和 On(green,table)）
    
    Args:
        condition1: 第一个条件字符串
        condition2: 第二个条件字符串
    
    Returns:
        bool: 是否语义等价
    """
    parsed1 = extract_condition_args(condition1)
    parsed2 = extract_condition_args(condition2)
    
    # 如果无法解析，使用精确匹配
    if not parsed1 or not parsed2:
        return condition1 == condition2
    
    pred1, args1 = parsed1
    pred2, args2 = parsed2
    
    # 情况1: 参数完全相同，谓词不同（视为同义词）
    if args1 == args2:
        return True
    
    # 情况2: 语义等价模式
    # Clear(X) 等价于 On(X,任何表面) - 只要第一个参数相同就认为等价
    if pred1 == "Clear" and pred2 == "On":
        # Clear(green) 等价于 On(green,table) 或 On(green,任何表面)
        # 检查 args1 是否是 args2 的第一个参数
        args2_list = [arg.strip() for arg in args2.split(',')]
        if len(args2_list) >= 1 and args1 == args2_list[0]:
            return True
    
    if pred2 == "Clear" and pred1 == "On":
        # 同上，但顺序相反
        args1_list = [arg.strip() for arg in args1.split(',')]
        if len(args1_list) >= 1 and args2 == args1_list[0]:
            return True
    
    # 情况2b: IsSurface(X) 等价于 On(X,任何表面)
    if pred1 in ["IsSurface", "OnTable", "IsOnSurface", "OnSurface"] and pred2 == "On":
        args2_list = [arg.strip() for arg in args2.split(',')]
        if len(args2_list) >= 1 and args1 == args2_list[0]:
            return True
    
    if pred2 in ["IsSurface", "OnTable", "IsOnSurface", "OnSurface"] and pred1 == "On":
        args1_list = [arg.strip() for arg in args1.split(',')]
        if len(args1_list) >= 1 and args2 == args1_list[0]:
            return True
    
    # 情况3: OnTable(X) 等价于 On(X,table)
    if pred1 == "OnTable" and pred2 == "On":
        args2_list = [arg.strip() for arg in args2.split(',')]
        if len(args2_list) >= 2 and args1 == args2_list[0] and args2_list[1] == "table":
            return True
    
    if pred2 == "OnTable" and pred1 == "On":
        args1_list = [arg.strip() for arg in args1.split(',')]
        if len(args1_list) >= 2 and args2 == args1_list[0] and args1_list[1] == "table":
            return True
    
    return False

def sets_match_with_synonyms(set1, set2):
    """
    比较两个集合是否匹配，支持自动识别同义词和语义等价
    如果两个条件语义等价，它们被认为是匹配的
    
    Args:
        set1: 第一个条件集合
        set2: 第二个条件集合
    
    Returns:
        bool: 是否匹配
    """
    if not isinstance(set1, set) or not isinstance(set2, set):
        return set1 == set2
    
    # 如果集合大小不同，肯定不匹配
    if len(set1) != len(set2):
        return False
    
    # 转换为列表以便跟踪已匹配的条件
    list1 = list(set1)
    list2 = list(set2)
    set2_matched = set()  # 记录set2中已匹配的条件索引
    
    # 检查每个集合中的条件是否能在另一个集合中找到等价条件
    for condition1 in list1:
        found_match = False
        
        for idx, condition2 in enumerate(list2):
            if idx in set2_matched:
                continue
            
            # 使用语义等价检查
            if conditions_semantically_equivalent(condition1, condition2):
                found_match = True
                set2_matched.add(idx)
                break
        
        if not found_match:
            return False
    
    # 确保set2中的所有条件都被匹配了
    return len(set2_matched) == len(list2)

def compare_results(llm_result, ground_truth):
    """
    比较LLM输出和标准答案是否一致
    自动识别同义词和语义等价：
    1. 参数完全相同，谓词不同（如 Clear(green) 和 OnTable(green)）
    2. 语义等价（如 Clear(green) 和 On(green,table)）
    
    Args:
        llm_result: dict，包含correct_pre, correct_add, correct_del
        ground_truth: dict，包含correct_pre, correct_add, correct_del
    
    Returns:
        tuple: (is_correct, feedback_message)
    """
    # 使用支持同义词的集合比较
    pre_match = sets_match_with_synonyms(
        llm_result.get("correct_pre", set()),
        ground_truth.get("correct_pre", set())
    )
    add_match = sets_match_with_synonyms(
        llm_result.get("correct_add", set()),
        ground_truth.get("correct_add", set())
    )
    del_match = sets_match_with_synonyms(
        llm_result.get("correct_del", set()),
        ground_truth.get("correct_del", set())
    )
    
    is_correct = pre_match and add_match and del_match
    
    feedback_parts = []
    if not pre_match:
        feedback_parts.append(f"pre 不匹配：期望 {ground_truth.get('correct_pre', set())}，得到 {llm_result.get('correct_pre', set())}")
    if not add_match:
        feedback_parts.append(f"add 不匹配：期望 {ground_truth.get('correct_add', set())}，得到 {llm_result.get('correct_add', set())}")
    if not del_match:
        feedback_parts.append(f"del_set 不匹配：期望 {ground_truth.get('correct_del', set())}，得到 {llm_result.get('correct_del', set())}")
    
    feedback_message = "\n".join(feedback_parts) if feedback_parts else "完全正确！"
    
    return is_correct, feedback_message

def build_feedback_prompt(action_name, current_pre, current_add, current_del, llm_result, ground_truth, feedback_message):
    """
    构建反馈prompt，用于告诉大模型哪里错了
    """
    pre_list = list(current_pre) if isinstance(current_pre, set) else current_pre
    add_list = list(current_add) if isinstance(current_add, set) else current_add
    del_list = list(current_del) if isinstance(current_del, set) else current_del
    
    llm_pre_list = list(llm_result.get("correct_pre", set()))
    llm_add_list = list(llm_result.get("correct_add", set()))
    llm_del_list = list(llm_result.get("correct_del", set()))
    
    ground_truth_pre_list = list(ground_truth.get("correct_pre", set()))
    ground_truth_add_list = list(ground_truth.get("correct_add", set()))
    ground_truth_del_list = list(ground_truth.get("correct_del", set()))
    
    prompt = f"""你之前的回答不正确，请重新分析。

**动作信息：**
- 动作名称：{action_name}
- 当前前置条件（pre）：{pre_list}
- 当前添加效果（add）：{add_list}
- 当前删除效果（del）：{del_list}

**你之前的回答：**
```python
pre = {llm_pre_list}
add = {llm_add_list}
del_set = {llm_del_list}
```

**错误信息：**
{feedback_message}

**标准答案参考：**
- 正确的 pre：{ground_truth_pre_list}
- 正确的 add：{ground_truth_add_list}
- 正确的 del_set：{ground_truth_del_list}

请重新分析图片，找出问题所在，并输出修正后的正确的 pre、add、del_set，使用 Python set 格式。
"""
    return prompt

# 定义5个示例案例
example_cases = [
    # 案例1: PutIn(apple,cabinet) - pre缺少IsOpen(cabinet)
    {
        "image_name": "putin_failure.png",
        "action_name": "PutIn(apple,cabinet)",

        "initial_state": {"IsHoliding(apple)"},
        "objects": {"apple", "cabinet"},
        "goal": {"In(apple,cabinet)"},


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
        "goal": {"On(red,green)"},


        "current_pre": {"Holding(red)"},
        "current_add": {"On(red,green)"},
        "current_del": {"Holding(red)"},


        "error_type": "missing",
        "error_description": "Pre : Clear(green) missing",
        "correct_pre": {"Holding(red)", "Clear(green)"},
        "correct_add": {"On(red,green)"},
        "correct_del": {"Holding(red)"}
    },
    
    # 案例3: Lift(big_box,board) - pre缺少IsHolding(leftrobot,big_box)
    {
        "image_name": "lift_failure.png",
        "action_name": "Lift(big_box,board)",

        "initial_state": {"IsHandEmpty(right_robot)","IsHandEmpty(left_robot)", "On(right_robot,right_table)", "On(left_robot,left_table)", "On(big_box,center_table)"},
        "objects": {"big_box", "right_robot", "left_robot", "right_table", "left_table", "center_table"},
        "goal": {"On(big_box,board)"},

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
        "goal": {"On(left_green_tea,right_table)"},


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
        "goal": {"On(apple,table)"},

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

def process_single_case(case, image_dir, llm_client, output_dir):
    """
    处理单个案例
    
    Args:
        case: 案例字典
        image_dir: 图片目录
        llm_client: LLM客户端
        output_dir: 输出目录
    """
    image_path = os.path.join(image_dir, case["image_name"])
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"警告：图片不存在: {image_path}")
        return None
    
    # 构建prompt
    prompt = build_correct_action_prompt(
        action_name=case["action_name"],
        current_pre=case["current_pre"],
        current_add=case["current_add"],
        current_del=case["current_del"],
        error_type=case["error_type"],
        # error_description=case["error_description"]
    )
    
    # 保存prompt
    prompt_file = os.path.join(output_dir, f"prompt_{case['image_name'].replace('.png', '')}.txt")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"\n处理案例: {case['action_name']}")
    print(f"图片: {case['image_name']}")
    
    # 发送请求
    response = request_with_image(llm_client, image_path, prompt)
    
    if response is None:
        print("请求失败")
        return None
    
    # 保存响应
    response_file = os.path.join(output_dir, f"response_{case['image_name'].replace('.png', '')}.txt")
    with open(response_file, 'w', encoding='utf-8') as f:
        f.write(response)
    
    # 解析响应
    parsed_result = parse_llm_response(response)
    
    # 构建结果字典
    result = {
        "image_name": case["image_name"],
        "action_name": case["action_name"],
        "current_pre": str(case["current_pre"]),
        "current_add": str(case["current_add"]),
        "current_del": str(case["current_del"]),
        "error_type": case["error_type"],
        "error_description": case["error_description"],
        "llm_response": response,
        "correct_pre": str(parsed_result.get("correct_pre", set())),
        "correct_add": str(parsed_result.get("correct_add", set())),
        "correct_del": str(parsed_result.get("correct_del", set())),
        "explanation": parsed_result.get("explanation", ""),
        "ground_truth_pre": str(case.get("correct_pre", set())),
        "ground_truth_add": str(case.get("correct_add", set())),
        "ground_truth_del": str(case.get("correct_del", set()))
    }
    
    return result

def process_analysis_case(case, image_path, openai_client, model, output_dir, max_feedback_times=3, try_idx=None):
    """
    处理analysis类型的案例（前3个），实现反馈循环
    
    Args:
        case: 案例字典
        image_path: 图片路径
        openai_client: OpenAI客户端
        model: 模型名称
        output_dir: 输出目录
        max_feedback_times: 最大反馈次数（默认3次）
        try_idx: 运行次数索引（用于区分多次运行）
    
    Returns:
        dict: 结果字典
    """
    case_name = case['image_name'].replace('.png', '').replace('.jpg', '')
    if try_idx is not None:
        case_name = f"{case_name}_try{try_idx}"
    ground_truth = {
        "correct_pre": case.get("correct_pre", set()),
        "correct_add": case.get("correct_add", set()),
        "correct_del": case.get("correct_del", set())
    }
    
    # 构建初始prompt
    prompt = build_analysis_prompt(
        action_name=case["action_name"],
        current_pre=case["current_pre"],
        current_add=case["current_add"],
        current_del=case["current_del"]
    )
    
    messages = []
    feedback_times = 0
    is_correct = False
    final_result = None
    
    # 编码图片
    img_base64 = encode_image(image_path)
    
    while feedback_times <= max_feedback_times and not is_correct:
        attempt_num = feedback_times + 1
        
        # 构建消息
        message_content = [
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
        
        if feedback_times == 0:
            # 第一次请求
            messages.append({
                "role": "user",
                "content": message_content
            })
        else:
            # 反馈请求
            messages.append({
                "role": "user",
                "content": message_content
            })
        
        print(f"\n  尝试 {attempt_num}/{max_feedback_times + 1}...")
        
        # 保存prompt
        prompt_file = os.path.join(output_dir, f"prompt_{case_name}_attempt{attempt_num}.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # 发送请求
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            response_text = response.choices[0].message.content
            
            # 保存响应
            response_file = os.path.join(output_dir, f"response_{case_name}_attempt{attempt_num}.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response_text)
            
            # 解析响应
            parsed_result = parse_python_set_response(response_text)
            
            # 比较结果
            is_correct, feedback_message = compare_results(parsed_result, ground_truth)
            
            if is_correct:
                print(f"  ✓ 回答正确！")
                final_result = parsed_result
            else:
                print(f"  ✗ 回答不正确：{feedback_message}")
                if feedback_times < max_feedback_times:
                    # 构建反馈prompt
                    prompt = build_feedback_prompt(
                        action_name=case["action_name"],
                        current_pre=case["current_pre"],
                        current_add=case["current_add"],
                        current_del=case["current_del"],
                        llm_result=parsed_result,
                        ground_truth=ground_truth,
                        feedback_message=feedback_message
                    )
                    feedback_times += 1
                else:
                    print(f"  已达到最大反馈次数，停止反馈")
                    final_result = parsed_result
                    
        except Exception as e:
            print(f"  请求失败: {e}")
            break
    
    # 构建最终结果字典
    result = {
        "image_name": case["image_name"],
        "action_name": case["action_name"],
        "case_type": "analysis",
        "current_pre": str(case["current_pre"]),
        "current_add": str(case["current_add"]),
        "current_del": str(case["current_del"]),
        "error_type": case.get("error_type", ""),
        "error_description": case.get("error_description", ""),
        "feedback_times": feedback_times,
        "success": is_correct,
        "llm_pre": str(final_result.get("correct_pre", set()) if final_result else set()),
        "llm_add": str(final_result.get("correct_add", set()) if final_result else set()),
        "llm_del": str(final_result.get("correct_del", set()) if final_result else set()),
        "ground_truth_pre": str(ground_truth.get("correct_pre", set())),
        "ground_truth_add": str(ground_truth.get("correct_add", set())),
        "ground_truth_del": str(ground_truth.get("correct_del", set()))
    }
    
    return result

def process_judgment_case(case, image_path, openai_client, model, output_dir, max_feedback_times=3, try_idx=None):
    """
    处理判断类型的案例（后2个），实现反馈循环
    与analysis类型的区别：使用build_judgment_prompt（第一句是"执行完成"）
    
    Args:
        case: 案例字典
        image_path: 图片路径
        openai_client: OpenAI客户端
        model: 模型名称
        output_dir: 输出目录
        max_feedback_times: 最大反馈次数（默认3次）
        try_idx: 运行次数索引（用于区分多次运行）
    
    Returns:
        dict: 结果字典
    """
    case_name = case['image_name'].replace('.png', '').replace('.jpg', '')
    if try_idx is not None:
        case_name = f"{case_name}_try{try_idx}"
    ground_truth = {
        "correct_pre": case.get("correct_pre", set()),
        "correct_add": case.get("correct_add", set()),
        "correct_del": case.get("correct_del", set())
    }
    
    # 构建初始prompt（使用judgment类型的prompt）
    prompt = build_judgment_prompt(
        action_name=case["action_name"],
        current_pre=case["current_pre"],
        current_add=case["current_add"],
        current_del=case["current_del"]
    )
    
    messages = []
    feedback_times = 0
    is_correct = False
    final_result = None
    
    # 编码图片
    img_base64 = encode_image(image_path)
    
    while feedback_times <= max_feedback_times and not is_correct:
        attempt_num = feedback_times + 1
        
        # 构建消息
        message_content = [
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
        
        if feedback_times == 0:
            # 第一次请求
            messages.append({
                "role": "user",
                "content": message_content
            })
        else:
            # 反馈请求
            messages.append({
                "role": "user",
                "content": message_content
            })
        
        print(f"\n  尝试 {attempt_num}/{max_feedback_times + 1}...")
        
        # 保存prompt
        prompt_file = os.path.join(output_dir, f"prompt_{case_name}_attempt{attempt_num}.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # 发送请求
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            response_text = response.choices[0].message.content
            
            # 保存响应
            response_file = os.path.join(output_dir, f"response_{case_name}_attempt{attempt_num}.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response_text)
            
            # 解析响应
            parsed_result = parse_python_set_response(response_text)
            
            # 比较结果
            is_correct, feedback_message = compare_results(parsed_result, ground_truth)
            
            if is_correct:
                print(f"  ✓ 回答正确！")
                final_result = parsed_result
            else:
                print(f"  ✗ 回答不正确：{feedback_message}")
                if feedback_times < max_feedback_times:
                    # 构建反馈prompt
                    prompt = build_feedback_prompt(
                        action_name=case["action_name"],
                        current_pre=case["current_pre"],
                        current_add=case["current_add"],
                        current_del=case["current_del"],
                        llm_result=parsed_result,
                        ground_truth=ground_truth,
                        feedback_message=feedback_message
                    )
                    feedback_times += 1
                else:
                    print(f"  已达到最大反馈次数，停止反馈")
                    final_result = parsed_result
                    
        except Exception as e:
            print(f"  请求失败: {e}")
            break
    
    # 构建最终结果字典
    result = {
        "image_name": case["image_name"],
        "action_name": case["action_name"],
        "case_type": "judgment",
        "current_pre": str(case["current_pre"]),
        "current_add": str(case["current_add"]),
        "current_del": str(case["current_del"]),
        "error_type": case.get("error_type", ""),
        "error_description": case.get("error_description", ""),
        "feedback_times": feedback_times,
        "success": is_correct,
        "llm_pre": str(final_result.get("correct_pre", set()) if final_result else set()),
        "llm_add": str(final_result.get("correct_add", set()) if final_result else set()),
        "llm_del": str(final_result.get("correct_del", set()) if final_result else set()),
        "ground_truth_pre": str(ground_truth.get("correct_pre", set())),
        "ground_truth_add": str(ground_truth.get("correct_add", set())),
        "ground_truth_del": str(ground_truth.get("correct_del", set()))
    }
    
    return result

def main():
    """主函数"""
    # 配置
    model = "gpt-4o-mini"  # 使用支持视觉的模型
    image_dir = os.path.join(DIR, "a_exp3_pic")  # 图片目录，需要用户提供
    output_dir = os.path.join(DIR, "a_exp1_llm_bt_results", "cross_feedback_results")

    
    total_try_times = 2  # 每个案例跑几次，取平均值
    feedback_times = 0
    case_ls = [2]  # 可以选择要处理的案例，例如：[0, 1, 2] 或 ["PutIn(apple,cabinet)", "Stack(red,green)"]
                  # 如果为空列表，则处理所有案例
                  # 可以通过索引（0-4）或案例名称来选择
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # 获取当前时间戳
    current_date = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = os.path.join(output_dir, f"results_{current_date}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建OpenAI客户端用于图片请求
    openai_client = OpenAI(
        base_url="https://api.dwyu.top/v1",
        api_key="sk-Gtk0rmTrrRjEOj8Kru5exXuOKwpwSqiR3intYCMvtIBzLqzN"
    )
    
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
    
    # 分离analysis类型和判断类型的案例（通过action_name匹配）
    analysis_cases = []
    judgment_cases = []
    analysis_action_names = {case["action_name"] for case in example_cases[:3]}
    judgment_action_names = {case["action_name"] for case in example_cases[3:]}
    
    for case in filtered_all_cases:
        if case["action_name"] in analysis_action_names:
            analysis_cases.append(case)
        elif case["action_name"] in judgment_action_names:
            judgment_cases.append(case)
    
    # 打印将要处理的案例
    print("\n" + "="*80)
    print("案例选择")
    print("="*80)
    if case_ls:
        print(f"选择的案例: {case_ls}")
        print(f"\n将处理的案例:")
        if analysis_cases:
            print(f"  Analysis 类型 ({len(analysis_cases)} 个):")
            for case in analysis_cases:
                print(f"    - {case['action_name']} ({case['image_name']})")
        if judgment_cases:
            print(f"  Judgment 类型 ({len(judgment_cases)} 个):")
            for case in judgment_cases:
                print(f"    - {case['action_name']} ({case['image_name']})")
    else:
        print("将处理所有案例")
        print(f"  Analysis 类型: {len(analysis_cases)} 个")
        print(f"  Judgment 类型: {len(judgment_cases)} 个")
    
    all_results = []
    
    # 处理analysis类型的案例（带反馈循环）
    print("="*80)
    print("处理 Analysis 类型案例（带反馈循环）")
    print("="*80)
    print(f"每个案例将运行 {total_try_times} 次")
    
    for case in analysis_cases:
        image_path = os.path.join(image_dir, case["image_name"])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"跳过案例 {case['action_name']}：图片不存在 {image_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"处理案例: {case['action_name']}")
        print(f"图片: {case['image_name']}")
        print(f"{'='*80}")
        
        # 对每个案例运行 total_try_times 次
        for try_idx in range(1, total_try_times + 1):
            print(f"\n--- 第 {try_idx}/{total_try_times} 次运行 ---")
            
            result = process_analysis_case(
                case=case,
                image_path=image_path,
                openai_client=openai_client,
                model=model,
                output_dir=output_dir,
                max_feedback_times=feedback_times,
                try_idx=try_idx
            )
            
            # 添加 try_idx 字段
            result["try_idx"] = try_idx
            all_results.append(result)
            
            print(f"第 {try_idx} 次运行完成: 成功={result['success']}, 反馈次数={result['feedback_times']}")
        
        # 统计这个案例的成功率
        case_results = [r for r in all_results if r.get("action_name") == case["action_name"]]
        success_count = sum(1 for r in case_results if r.get("success", False))
        success_rate = success_count / len(case_results) * 100 if case_results else 0.0
        avg_feedback = sum(r.get("feedback_times", 0) for r in case_results) / len(case_results) if case_results else 0.0
        print(f"\n案例 {case['action_name']} 统计: 成功 {success_count}/{len(case_results)} ({success_rate:.2f}%), 平均反馈次数: {avg_feedback:.2f}")
    
    # 处理判断类型的案例（带反馈循环）
    print("\n" + "="*80)
    print("处理判断类型案例（带反馈循环）")
    print("="*80)
    print(f"每个案例将运行 {total_try_times} 次")
    
    for case in judgment_cases:
        image_path = os.path.join(image_dir, case["image_name"])
        
        if not os.path.exists(image_path):
            print(f"跳过案例 {case['action_name']}：图片不存在 {image_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"处理案例: {case['action_name']}")
        print(f"图片: {case['image_name']}")
        print(f"{'='*80}")
        
        # 对每个案例运行 total_try_times 次
        for try_idx in range(1, total_try_times + 1):
            print(f"\n--- 第 {try_idx}/{total_try_times} 次运行 ---")
            
            result = process_judgment_case(
                case=case,
                image_path=image_path,
                openai_client=openai_client,
                model=model,
                output_dir=output_dir,
                max_feedback_times=feedback_times,
                try_idx=try_idx
            )
            
            # 添加 try_idx 字段
            result["try_idx"] = try_idx
            all_results.append(result)
            
            print(f"第 {try_idx} 次运行完成: 成功={result['success']}, 反馈次数={result['feedback_times']}")
        
        # 统计这个案例的成功率
        case_results = [r for r in all_results if r.get("action_name") == case["action_name"]]
        success_count = sum(1 for r in case_results if r.get("success", False))
        success_rate = success_count / len(case_results) * 100 if case_results else 0.0
        avg_feedback = sum(r.get("feedback_times", 0) for r in case_results) / len(case_results) if case_results else 0.0
        print(f"\n案例 {case['action_name']} 统计: 成功 {success_count}/{len(case_results)} ({success_rate:.2f}%), 平均反馈次数: {avg_feedback:.2f}")
    
    # 保存结果到CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(output_dir, f"results_{current_date}.csv")
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
            case_key = result.get("action_name")
            case_groups[case_key].append(result)
        
        # Analysis 类型统计
        analysis_results = [r for r in all_results if r.get("case_type") == "analysis"]
        if analysis_results:
            # 按案例分组
            analysis_case_groups = {k: v for k, v in case_groups.items() 
                                   if v[0].get("case_type") == "analysis"}
            
            print(f"\nAnalysis 类型案例:")
            
            case_success_rates = []
            for case_name, case_results in analysis_case_groups.items():
                success_count = sum(1 for r in case_results if r.get("success", False))
                total_runs = len(case_results)
                case_success_rate = success_count / total_runs * 100 if total_runs > 0 else 0.0
                case_success_rates.append(case_success_rate)
                avg_feedback = sum(r.get("feedback_times", 0) for r in case_results) / total_runs if total_runs > 0 else 0.0
                print(f"  {case_name}: 成功 {success_count}/{total_runs} ({case_success_rate:.2f}%), 平均反馈次数: {avg_feedback:.2f}")
            
            # 计算平均成功率（按案例平均）
            avg_success_rate = sum(case_success_rates) / len(case_success_rates) if case_success_rates else 0.0
            total_runs = len(analysis_results)
            total_success = sum(1 for r in analysis_results if r.get("success", False))
            avg_feedback = sum(r.get("feedback_times", 0) for r in analysis_results) / total_runs if total_runs > 0 else 0.0
            
            print(f"\n  Analysis 类型汇总:")
            print(f"    案例数: {len(analysis_case_groups)}")
            print(f"    总运行次数: {total_runs}")
            print(f"    总成功次数: {total_success}")
            print(f"    按运行次数计算成功率: {total_success}/{total_runs} ({total_success/total_runs*100:.2f}%)")
            print(f"    按案例平均成功率: {avg_success_rate:.2f}%")
            print(f"    平均反馈次数: {avg_feedback:.2f}")
        
        # Judgment 类型统计
        judgment_results = [r for r in all_results if r.get("case_type") == "judgment"]
        if judgment_results:
            # 按案例分组
            judgment_case_groups = {k: v for k, v in case_groups.items() 
                                  if v[0].get("case_type") == "judgment"}
            
            print(f"\nJudgment 类型案例:")
            
            case_success_rates = []
            for case_name, case_results in judgment_case_groups.items():
                success_count = sum(1 for r in case_results if r.get("success", False))
                total_runs = len(case_results)
                case_success_rate = success_count / total_runs * 100 if total_runs > 0 else 0.0
                case_success_rates.append(case_success_rate)
                avg_feedback = sum(r.get("feedback_times", 0) for r in case_results) / total_runs if total_runs > 0 else 0.0
                print(f"  {case_name}: 成功 {success_count}/{total_runs} ({case_success_rate:.2f}%), 平均反馈次数: {avg_feedback:.2f}")
            
            # 计算平均成功率（按案例平均）
            avg_success_rate = sum(case_success_rates) / len(case_success_rates) if case_success_rates else 0.0
            total_runs = len(judgment_results)
            total_success = sum(1 for r in judgment_results if r.get("success", False))
            avg_feedback = sum(r.get("feedback_times", 0) for r in judgment_results) / total_runs if total_runs > 0 else 0.0
            
            print(f"\n  Judgment 类型汇总:")
            print(f"    案例数: {len(judgment_case_groups)}")
            print(f"    总运行次数: {total_runs}")
            print(f"    总成功次数: {total_success}")
            print(f"    按运行次数计算成功率: {total_success}/{total_runs} ({total_success/total_runs*100:.2f}%)")
            print(f"    按案例平均成功率: {avg_success_rate:.2f}%")
            print(f"    平均反馈次数: {avg_feedback:.2f}")
        
        # 总体统计
        if analysis_results or judgment_results:
            all_typed_results = analysis_results + judgment_results
            all_case_groups = {k: v for k, v in case_groups.items()}
            
            all_case_success_rates = []
            for case_name, case_results in all_case_groups.items():
                success_count = sum(1 for r in case_results if r.get("success", False))
                total_runs = len(case_results)
                case_success_rate = success_count / total_runs * 100 if total_runs > 0 else 0.0
                all_case_success_rates.append(case_success_rate)
            
            avg_success_rate_all = sum(all_case_success_rates) / len(all_case_success_rates) if all_case_success_rates else 0.0
            total_runs_all = len(all_typed_results)
            total_success_all = sum(1 for r in all_typed_results if r.get("success", False))
            avg_feedback_all = sum(r.get("feedback_times", 0) for r in all_typed_results) / total_runs_all if total_runs_all > 0 else 0.0
            
            print(f"\n总体统计:")
            print(f"  案例数: {len(all_case_groups)}")
            print(f"  总运行次数: {total_runs_all}")
            print(f"  总成功次数: {total_success_all}")
            print(f"  按运行次数计算成功率: {total_success_all}/{total_runs_all} ({total_success_all/total_runs_all*100:.2f}%)")
            print(f"  按案例平均成功率: {avg_success_rate_all:.2f}%")
            print(f"  平均反馈次数: {avg_feedback_all:.2f}")
        
        # 打印摘要（按案例分组显示）
        print("\n" + "="*80)
        print("处理摘要")
        print("="*80)
        for case_name, case_results in sorted(case_groups.items()):
            print(f"\n案例: {case_name}")
            print(f"类型: {case_results[0].get('case_type', 'unknown')}")
            for result in sorted(case_results, key=lambda x: x.get('try_idx', 0)):
                status = "✓" if result.get('success', False) else "✗"
                print(f"  运行 {result.get('try_idx', 0)}: {status} 成功={result.get('success', False)}, 反馈次数={result.get('feedback_times', 0)}")
            success_count = sum(1 for r in case_results if r.get('success', False))
            print(f"  总计: {success_count}/{len(case_results)} 成功")
    else:
        print("\n没有成功处理的案例")

if __name__ == "__main__":
    main()

