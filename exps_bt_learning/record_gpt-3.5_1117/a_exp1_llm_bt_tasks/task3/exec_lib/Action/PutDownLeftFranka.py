from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(
        ["left_lego", "center_big_box"],
        ["left_table", "center_table", "center_big_box", "box_board"]
    ))

    def __init__(self, *args):
        super().__init__()

    @classmethod
    def get_info(cls, *arg):
        obj, dest = arg
        info = {}
        info["pre"] = {
            f"Holding(left_franka,{obj})",
            f"On(left_franka,left_table)"
        }
        info["add"] = {
            f"IsHandEmpty(left_franka)",
            f"On({obj},{dest})"
        }
        info["del_set"] = {f"Holding(left_franka,{obj})"}
        info["cost"] = 1
        return info