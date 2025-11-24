from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        milk = arg[0]
        info = {}
        info["pre"] = {
            f"IsHandEmpty(right_franka)",
            f"On({milk},right_table)",
            f"CanGrasp(right_franka,{milk})"
        }
        info["add"] = {f"Holding(right_franka,{milk})"}
        info["del_set"] = {f"IsHandEmpty(right_franka)", f"On({milk},right_table)"}
        info["cost"] = 1
        return info