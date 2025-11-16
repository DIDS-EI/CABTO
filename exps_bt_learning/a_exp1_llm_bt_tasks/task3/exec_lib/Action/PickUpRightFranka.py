from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_lego"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        obj = args[0]
        info = dict()
        info["pre"] = {f"IsHandEmpty(right_franka)", f"On({obj},right_table)", f"On(right_franka,right_table)"}
        info["add"] = {f"Holding(right_franka,{obj})"}
        info["del_set"] = {f"IsHandEmpty(right_franka)", f"On({obj},right_table)"}
        info["cost"] = 1
        return info