from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(
        ["left_table", "center_table", "right_table", "box_board"],
        ["left_table", "center_table", "right_table", "box_board"]
    ))

    def __init__(self, *args):
        super().__init__()

    @classmethod
    def get_info(cls, *arg):
        frm, to = arg
        if frm == to:
            return None
        info = {}
        info["pre"] = {f"On(right_franka,{frm})"}
        info["add"] = {f"On(right_franka,{to})"}
        info["del_set"] = {f"On(right_franka,{frm})"}
        info["cost"] = 1
        return info