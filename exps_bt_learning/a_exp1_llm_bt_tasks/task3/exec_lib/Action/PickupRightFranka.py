from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickupRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_lego", "center_big_box"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsHandEmpty(right_franka)", f"On(right_franka,right_table)", f"On({arg[0]},right_table)"}
        info["add"] = {f"Holding(right_franka,{arg[0]})"}
        info["del_set"] = {f"IsHandEmpty(right_franka)", f"On({arg[0]},right_table)"}
        info["cost"] = 1
        return info