from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_lego"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {
            "IsHandEmpty(right_franka)", 
            f"On({arg[0]},right_table)", 
            "On(right_franka,right_table)", 
            "RightFrankaCanOnlyDoRightAction()"
        }
        info["add"] = {f"Holding(right_franka,{arg[0]})"}
        info["del_set"] = {"IsHandEmpty(right_franka)", f"On({arg[0]},right_table)"}
        info["cost"] = 1
        return info