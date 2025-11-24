from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Place(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = ["big_box", "board"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsHolding(right_robot,{arg[0]})"}
        info["add"] = {f"On({arg[0]},board)"}
        info["del_set"] = {f"IsHolding(right_robot,{arg[0]})", f"On({arg[0]},center_table)"}
        info["cost"] = 1
        return info