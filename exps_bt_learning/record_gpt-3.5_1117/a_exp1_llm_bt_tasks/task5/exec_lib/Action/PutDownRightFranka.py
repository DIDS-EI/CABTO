from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["right_milk"], ["right_table", "right_cup"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        milk, dest = arg
        info = {}
        info["pre"] = {
            f"Holding(right_franka,{milk})"
        }
        info["add"] = {f"IsHandEmpty(right_franka)", f"On({milk},{dest})"}
        info["del_set"] = {f"Holding(right_franka,{milk})"}
        info["cost"] = 1
        return info