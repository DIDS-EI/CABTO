from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_box1", "left_box2"], ["left_table", "right_table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj, table = arg
        info = {}
        info["pre"] = {
            f"IsHolding(left_franka,{obj})",
            f"On(left_franka,{table})"
        }
        info["add"] = {f"IsHandEmpty(left_franka)", f"On({obj},{table})"}
        info["del_set"] = {f"IsHolding(left_franka,{obj})"}
        info["cost"] = 1
        return info