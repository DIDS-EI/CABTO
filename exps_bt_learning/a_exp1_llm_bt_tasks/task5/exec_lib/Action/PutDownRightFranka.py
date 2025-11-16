from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["right_milk"], ["right_table", "right_cup"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj, loc = arg
        info = {}
        info["pre"] = {f"Holding(right_franka,{obj})"}
        if loc == "right_cup":
            info["pre"].add(f"IsEmpty(right_cup)")
        elif loc == "right_table":
            info["pre"].add(f"On(right_cup,right_table)")
        if loc == "right_cup":
            info["add"] = {f"IsHandEmpty(right_franka)", f"IsHalfFull(right_cup)"}
            info["del_set"] = {f"Holding(right_franka,{obj})", f"IsEmpty(right_cup)"}
        else:
            info["add"] = {f"IsHandEmpty(right_franka)", f"On({obj},{loc})"}
            info["del_set"] = {f"Holding(right_franka,{obj})"}
        info["cost"] = 1
        return info