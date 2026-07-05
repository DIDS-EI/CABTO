from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class DropRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_table", "right_table"]

    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(right_franka,right_box)"}
        info["add"] = {f"On(right_box,{arg[0]})", f"IsHandEmpty(right_franka)"}
        info["del_set"] = {f"Holding(right_franka,right_box)"}
        info["cost"] = 1
        return info