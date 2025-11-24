from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_box"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj = arg[0]
        info = {}
        info["pre"] = {
            f"IsHandEmpty(right_franka)",
            f"On({obj},right_table)",
            f"On(right_franka,right_table)"
        }
        info["add"] = {f"IsHolding(right_franka,{obj})"}
        info["del_set"] = {f"IsHandEmpty(right_franka)", f"On({obj},right_table)"}
        info["cost"] = 1
        return info