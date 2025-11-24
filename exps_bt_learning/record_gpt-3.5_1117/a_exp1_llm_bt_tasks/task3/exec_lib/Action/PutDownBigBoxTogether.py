from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownBigBoxTogether(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["box_board", "center_table"]

    def __init__(self, *args):
        super().__init__()
        
    @classmethod
    def get_info(cls, *arg):
        dest = arg[0]
        info = {}
        info["pre"] = {
            "Holding(left_franka,center_big_box)",
            "Holding(right_franka,center_big_box)",
            "On(left_franka,left_table)",
            "On(right_franka,right_table)"
        }
        info["add"] = {
            "IsHandEmpty(left_franka)",
            "IsHandEmpty(right_franka)",
            f"On(center_big_box,{dest})"
        }
        info["del_set"] = {
            "Holding(left_franka,center_big_box)",
            "Holding(right_franka,center_big_box)"
        }
        info["cost"] = 1
        return info