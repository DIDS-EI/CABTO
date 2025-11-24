from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Lift(OGAction):
    can_be_expanded = False
    num_args = 2
    valid_args = [("big_box", "board")]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {"IsHandEmpty(right_robot)", "IsHandEmpty(left_robot)", "On(big_box,center_table)", "On(left_robot,center_table)", "On(right_robot,right_table)"}
        info["add"] = {f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"On({arg[0]},center_table)"}
        info["cost"] = 1
        return info