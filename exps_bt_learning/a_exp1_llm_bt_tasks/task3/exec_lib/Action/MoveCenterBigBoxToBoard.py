from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveCenterBigBoxToBoard(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [("center_big_box", "box_board")]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"On(center_big_box,center_table)", "BigBoxNeedTwoFrankaHoldTogether()"}
        info["add"] = {f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"On({arg[0]},center_table)"}
        info["cost"] = 1
        return info