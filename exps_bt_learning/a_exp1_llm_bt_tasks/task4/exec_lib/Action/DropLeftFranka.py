from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class DropLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_table", "left_table"]

    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(left_franka,left_box1)"}
        info["add"] = {f"On(left_box1,{arg[0]})", f"IsHandEmpty(left_franka)"}
        info["del_set"] = {f"Holding(left_franka,left_box1)"}
        info["cost"] = 1
        return info