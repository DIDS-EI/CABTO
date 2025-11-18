from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_lego"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsHandEmpty(left_franka)", f"On(left_franka,left_table)", f"On({arg[0]},left_table)"}
        info["add"] = {f"Holding(left_franka,{arg[0]})"}
        info["del_set"] = {f"IsHandEmpty(left_franka)", f"On({arg[0]},left_table)"}
        info["cost"] = 1
        return info