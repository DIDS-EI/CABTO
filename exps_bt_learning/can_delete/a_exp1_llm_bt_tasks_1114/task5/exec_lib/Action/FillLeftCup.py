from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class FillLeftCup(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_cup"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {
            f"Holding(left_franka, {arg[0]})",
            "IsFull(left_milk)",
            "IsEmpty(left_cup)"
        }
        info["add"] = {"IsHalfFull(left_cup)"}
        info["del_set"] = {"IsEmpty(left_cup)"}
        info["cost"] = 1
        return info