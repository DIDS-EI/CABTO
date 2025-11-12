def format_array(array):
    result = "["
    if isinstance(array, float):
        return f"{array:.2f}"
    for i in range(len(array)):
        result += f"{array[i]:.2f},"
    return result[:-1] + "]"

# 彩色打印函数
def print_green(x):
    """打印绿色文本，带✅图标"""
    print(f"\033[92m✅ {x}\033[0m")

def print_red(x):
    """打印红色文本，带❌图标"""
    print(f"\033[91m❌ {x}\033[0m")

def print_yellow(x):
    """打印黄色文本，带⚠️图标"""
    print(f"\033[93m⚠️  {x}\033[0m")

def print_just_yellow(x):
    """打印黄色文本，不带图标"""
    print(f"\033[93m{x}\033[0m")

def print_blue(x):
    """打印蓝色文本，带🔵图标"""
    print(f"\033[94m🔵 {x}\033[0m")

def print_cyan(x):
    """打印青色文本，带💡图标"""
    print(f"\033[96m💡 {x}\033[0m")

def print_magenta(x):
    """打印洋红色文本，带🎯图标"""
    print(f"\033[95m🎯 {x}\033[0m")

def print_orange(x):
    """打印橘色文本"""
    print(f"\033[38;5;208m {x}\033[0m")

def print_mix_pose(x):
    """打印混合位姿，前三维黄色，后三位橘色"""
    print(f"\033[93m{'mixed_pose:':<10} {x[:3]}\033[0m \033[38;5;208m{x[3:6]}\033[0m")

def print_separator(title=""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print(f"\n{'='*50}")

def print_section(title):
    """打印章节标题"""
    print(f"\n📋 {title}")
    print("-" * 40)

