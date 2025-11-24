from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["right_cup", "right_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {
            "IsHandEmpty(right_franka)",
            f"On({arg[0]}, right_table)" if arg[0] == "right_cup" else 
            f"On({arg[0]}, right_table) & CanGrasp(right_franka, right_milk) & IsFull(right_milk)"
        }
        info["add"] = {f"Holding(right_franka, {arg[0]})"}
        info["del_set"] = {"IsHandEmpty(right_franka)", f"On({arg[0]}, right_table)"}
        info["cost"] = 1
        return info