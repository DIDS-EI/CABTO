from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        milk = arg[0]
        info = {}
        info["pre"] = {
            f"IsHandEmpty(left_franka)",
            f"On({milk},left_table)",
            f"CanGrasp(left_franka,{milk})"
        }
        info["add"] = {f"Holding(left_franka,{milk})"}
        info["del_set"] = {f"IsHandEmpty(left_franka)", f"On({milk},left_table)"}
        info["cost"] = 1
        return info