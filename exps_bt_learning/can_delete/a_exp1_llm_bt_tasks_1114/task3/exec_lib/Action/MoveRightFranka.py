from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["center_big_box"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {"On(right_franka,right_table)"}
        info["add"] = {"On(right_franka,center_big_box)"}  
        info["del_set"] = {"On(right_franka,right_table)"}
        info["cost"] = 1
        return info