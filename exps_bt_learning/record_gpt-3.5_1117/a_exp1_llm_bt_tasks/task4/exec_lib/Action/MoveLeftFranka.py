from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [pair for pair in itertools.product(["left_table", "right_table"], repeat=2) if pair[0] != pair[1]]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        frm, to = arg
        info = {}
        info["pre"] = {f"On(left_franka,{frm})"}
        info["add"] = {f"On(left_franka,{to})"}
        info["del_set"] = {f"On(left_franka,{frm})"}
        info["cost"] = 1
        return info