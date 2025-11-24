from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["right_box"], ["left_table", "right_table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj, table = arg
        info = {}
        info["pre"] = {
            f"IsHolding(right_franka,{obj})",
            f"On(right_franka,{table})"
        }
        info["add"] = {f"IsHandEmpty(right_franka)", f"On({obj},{table})"}
        info["del_set"] = {f"IsHolding(right_franka,{obj})"}
        info["cost"] = 1
        return info