from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [("right_franka", "right_lego")]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {f"IsHandEmpty(right_franka)", f"On({args[1]},{args[0]})"}
        info["add"] = {f"Holding(right_franka,{args[1]})"}
        info["del_set"] = {f"IsHandEmpty(right_franka)", f"On({args[1]},{args[0]})"}
        info["cost"] = 1
        return info