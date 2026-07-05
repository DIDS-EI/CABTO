from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveLeftRobot(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_table", "center_table"], ["center_table", "left_table"]))
    
    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"On(left_robot,{arg[0]})"}
        info["add"] = {f"On(left_robot,{arg[1]})"}
        info["del_set"] = {f"On(left_robot,{arg[0]})"}
        info["cost"] = 1
        return info