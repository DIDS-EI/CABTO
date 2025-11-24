from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(
        ["right_lego", "center_big_box"],
        ["right_table", "center_table", "center_big_box", "box_board"]
    ))

    def __init__(self, *args):
        super().__init__()

    @classmethod
    def get_info(cls, *arg):
        obj, dest = arg
        info = {}
        info["pre"] = {
            f"Holding(right_franka,{obj})",
            f"On(right_franka,right_table)"
        }
        info["add"] = {
            f"IsHandEmpty(right_franka)",
            f"On({obj},{dest})"
        }
        info["del_set"] = {f"Holding(right_franka,{obj})"}
        info["cost"] = 1
        return info