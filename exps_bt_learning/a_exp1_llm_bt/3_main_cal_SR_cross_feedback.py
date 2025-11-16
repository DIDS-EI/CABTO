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

def build_correct_action_prompt(action_name, current_pre, current_add, current_del, error_type, error_description):
    """
    构建修正动作的prompt
    
    Args:
        action_name: 动作名称，如 "Stack(obj1, obj2)"
        current_pre: 当前的前置条件列表
        current_add: 当前的添加效果列表
        current_del: 当前的删除效果列表
        error_type: 错误类型，如 "missing" 或 "incorrect"
        error_description: 错误描述，如 "Pre : (clear y) missing"
    """
    prompt = f"""你是一个动作效果修正专家。一个动作执行失败了，需要你观察图片，分析失败原因，并修正动作的前置条件（Pre）、添加效果（Add）和删除效果（Del）。

**动作信息：**
- 动作名称：{action_name}
- 当前前置条件（Pre）：{current_pre}
- 当前添加效果（Add）：{current_add}
- 当前删除效果（Del）：{current_del}

**错误信息：**
- 错误类型：{error_type}
- 错误描述：{error_description}

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
        "correct_pre": [],
        "correct_add": [],
        "correct_del": [],
        "explanation": ""
    }
    
    # 首先尝试提取代码块中的JSON（如果LLM用```json包装）
    json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
    if json_block_match:
        json_str = json_block_match.group(1)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
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
        result["correct_pre"] = items
    
    # 提取correct_add
    add_match = re.search(r'"correct_add"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
    if add_match:
        add_content = add_match.group(1)
        items = re.findall(r'"([^"]+)"', add_content)
        if not items:
            items = [s.strip() for s in add_content.split(',') if s.strip()]
        result["correct_add"] = items
    
    # 提取correct_del
    del_match = re.search(r'"correct_del"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
    if del_match:
        del_content = del_match.group(1)
        items = re.findall(r'"([^"]+)"', del_content)
        if not items:
            items = [s.strip() for s in del_content.split(',') if s.strip()]
        result["correct_del"] = items
    
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

# 定义5个示例案例
example_cases = [
    {
        "image_name": "stack_failure.png",  # 需要用户提供实际图片
        "action_name": "Stack(obj1,obj2)",
        "current_pre": ["On(obj1,table)", "On(obj2,table)", "IsHandEmpty()"],
        "current_add": ["On(obj1,obj2)"],
        "current_del": ["On(obj1,table)"],
        "error_type": "missing",
        "error_description": "Pre : (clear y) missing",
        "correct_pre": ["On(obj1,table)", "On(obj2,table)", "IsHandEmpty()", "clear(obj2)"],
        "correct_add": ["On(obj1,obj2)"],
        "correct_del": ["On(obj1,table)"]
    },
    {
        "image_name": "putin_failure.png",
        "action_name": "PutIn(item,container)",
        "current_pre": ["IsHolding(item)", "IsContainer(container)"],
        "current_add": ["In(item,container)"],
        "current_del": ["IsHolding(item)"],
        "error_type": "missing",
        "error_description": "Pre : (is-open container) missing",
        "correct_pre": ["IsHolding(item)", "IsContainer(container)", "is-open(container)"],
        "correct_add": ["In(item,container)"],
        "correct_del": ["IsHolding(item)"]
    },
    {
        "image_name": "pour_failure.png",
        "action_name": "Pour(item,c1,c2)",
        "current_pre": ["Contains(c1,item)", "IsContainer(c2)"],
        "current_add": ["Contains(c2,item)"],
        "current_del": [],
        "error_type": "missing",
        "error_description": "Del : (contains c1 liq) missing",
        "correct_pre": ["Contains(c1,item)", "IsContainer(c2)"],
        "correct_add": ["Contains(c2,item)"],
        "correct_del": ["Contains(c1,liq)"]
    },
    {
        "image_name": "lift_failure.png",
        "action_name": "Lift(heavy obj)",
        "current_pre": ["On(heavy_obj,table)", "IsHandEmpty()"],
        "current_add": ["IsHolding(heavy_obj)"],
        "current_del": ["On(heavy_obj,table)"],
        "error_type": "missing",
        "error_description": "Pre : (two-hands-required) missing",
        "correct_pre": ["On(heavy_obj,table)", "IsHandEmpty()", "two-hands-required(heavy_obj)"],
        "correct_add": ["IsHolding(heavy_obj)"],
        "correct_del": ["On(heavy_obj,table)"]
    },
    {
        "image_name": "rightpickup_failure.png",
        "action_name": "RightPickUp(obj,loc)",
        "current_pre": ["On(obj,loc)", "IsHandEmpty(RightHand)"],
        "current_add": ["IsHolding(RightHand,obj)"],
        "current_del": ["On(obj,loc)"],
        "error_type": "incorrect",
        "error_description": "Add : (holding LeftHand obj) incorrect",
        "correct_pre": ["On(obj,loc)", "IsHandEmpty(RightHand)"],
        "correct_add": ["IsHolding(RightHand,obj)"],
        "correct_del": ["On(obj,loc)"]
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
        error_description=case["error_description"]
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
        "correct_pre": str(parsed_result.get("correct_pre", [])),
        "correct_add": str(parsed_result.get("correct_add", [])),
        "correct_del": str(parsed_result.get("correct_del", [])),
        "explanation": parsed_result.get("explanation", ""),
        "ground_truth_pre": str(case.get("correct_pre", [])),
        "ground_truth_add": str(case.get("correct_add", [])),
        "ground_truth_del": str(case.get("correct_del", []))
    }
    
    return result

def main():
    """主函数"""
    # 配置
    model = "gpt-4o"  # 使用支持视觉的模型
    image_dir = os.path.join(DIR, "a_exp1_llm_bt_results", "images")  # 图片目录，需要用户提供
    output_dir = os.path.join(DIR, "a_exp1_llm_bt_results", "cross_feedback_results")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # 获取当前时间戳
    current_date = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = os.path.join(output_dir, f"results_{current_date}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化LLM（需要支持视觉的模型）
    print("初始化LLM...")
    # 注意：需要修改LLM类以支持图片输入，或者直接使用OpenAI客户端
    # 这里我们创建一个支持图片的LLM客户端
    try:
        llm_client = LLM(request_model=model)
        # 为了支持图片，我们需要直接使用OpenAI客户端
        # 检查是否有OPENAI_API_KEY环境变量
        if not os.getenv("OPENAI_API_KEY"):
            print("警告：未设置OPENAI_API_KEY环境变量，将使用LLM类中的默认配置")
    except Exception as e:
        print(f"初始化LLM失败: {e}")
        return
    
    # 创建OpenAI客户端用于图片请求
    openai_client = OpenAI(
        base_url="https://api.dwyu.top/v1",
        api_key="sk-Gtk0rmTrrRjEOj8Kru5exXuOKwpwSqiR3intYCMvtIBzLqzN"
    )
    
    # 处理所有案例
    all_results = []
    
    for case in example_cases:
        image_path = os.path.join(image_dir, case["image_name"])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"跳过案例 {case['action_name']}：图片不存在 {image_path}")
            continue
        
        # 构建prompt
        prompt = build_correct_action_prompt(
            action_name=case["action_name"],
            current_pre=case["current_pre"],
            current_add=case["current_add"],
            current_del=case["current_del"],
            error_type=case["error_type"],
            error_description=case["error_description"]
        )
        
        # 保存prompt
        prompt_file = os.path.join(output_dir, f"prompt_{case['image_name'].replace('.png', '').replace('.jpg', '')}.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"\n处理案例: {case['action_name']}")
        print(f"图片: {case['image_name']}")
        
        # 编码图片并发送请求
        try:
            img_base64 = encode_image(image_path)
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
            
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            response_text = response.choices[0].message.content
            
        except Exception as e:
            print(f"请求失败: {e}")
            response_text = None
        
        if response_text is None:
            print("请求失败，跳过此案例")
            continue
        
        # 保存响应
        response_file = os.path.join(output_dir, f"response_{case['image_name'].replace('.png', '').replace('.jpg', '')}.txt")
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(response_text)
        
        # 解析响应
        parsed_result = parse_llm_response(response_text)
        
        # 构建结果字典
        result = {
            "image_name": case["image_name"],
            "action_name": case["action_name"],
            "current_pre": str(case["current_pre"]),
            "current_add": str(case["current_add"]),
            "current_del": str(case["current_del"]),
            "error_type": case["error_type"],
            "error_description": case["error_description"],
            "llm_response": response_text,
            "correct_pre": str(parsed_result.get("correct_pre", [])),
            "correct_add": str(parsed_result.get("correct_add", [])),
            "correct_del": str(parsed_result.get("correct_del", [])),
            "explanation": parsed_result.get("explanation", ""),
            "ground_truth_pre": str(case.get("correct_pre", [])),
            "ground_truth_add": str(case.get("correct_add", [])),
            "ground_truth_del": str(case.get("correct_del", []))
        }
        
        all_results.append(result)
        print(f"完成案例: {case['action_name']}")
    
    # 保存结果到CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(output_dir, f"results_{current_date}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {csv_file}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("处理摘要")
        print("="*80)
        for result in all_results:
            print(f"\n动作: {result['action_name']}")
            print(f"图片: {result['image_name']}")
            print(f"错误: {result['error_description']}")
            print(f"修正后的Pre: {result['correct_pre']}")
            print(f"修正后的Add: {result['correct_add']}")
            print(f"修正后的Del: {result['correct_del']}")
    else:
        print("\n没有成功处理的案例")

if __name__ == "__main__":
    main()

