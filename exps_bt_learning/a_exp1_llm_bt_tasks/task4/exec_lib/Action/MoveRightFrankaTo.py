from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveRightFrankaTo(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_table", "left_table"]

    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"On(right_franka,{arg[0]})"}
        info["add"] = {f"On(right_franka,{arg[0]})"}
        info["del_set"] = {f"On(right_franka,right_table)"} if arg[0] == "left_table" else {f"On(right_franka,left_table)"}
        info["cost"] = 1
        return info