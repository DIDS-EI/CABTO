def print_red(text):
    """Print text in red color.
    
    Args:
        text: Text to print
    """
    print(f"\033[91m{text}\033[0m")  # 91m is the ANSI color code for red 