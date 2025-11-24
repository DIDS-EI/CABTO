from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_box1", "left_box2"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {
            f"On({arg[0]},left_table)",
            "IsHandEmpty(left_franka)",
            "On(left_franka,left_table)"
        }
        info["add"] = {f"IsHolding(left_franka,{arg[0]})"}
        info["del_set"] = {
            f"On({arg[0]},left_table)",
            "IsHandEmpty(left_franka)"
        }
        info["cost"] = 1
        return info