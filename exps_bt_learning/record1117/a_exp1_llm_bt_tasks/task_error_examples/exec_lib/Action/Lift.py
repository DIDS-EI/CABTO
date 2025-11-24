from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Lift(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["big_box"], ["board"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # 错误：pre缺少 IsHolding(leftrobot,big_box)
        # 正确的pre应该包含 IsHolding(leftrobot,big_box)，但这里缺少了
        info["pre"] = {f"On({arg[0]},{arg[1]})"}
        info["add"] = {f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"On({arg[0]},{arg[1]})"}
        info["cost"] = 1
        return info

