from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PourRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["right_milk"], ["right_cup"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        milk, cup = arg
        info = {}
        info["pre"] = {
            f"On({milk},right_table)",
            f"IsFull({milk})",
            f"IsEmpty({cup})",
        }
        info["add"] = {f"IsHalfFull({cup})", f"IsHalfFull({milk})"}
        info["del_set"] = {f"IsFull({milk})", f"IsEmpty({cup})"}
        info["cost"] = 1
        return info