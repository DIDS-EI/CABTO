from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_cup", "left_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {
            "IsHandEmpty(left_franka)",
            f"On({arg[0]}, left_table)" if arg[0] == "left_cup" else 
            f"On({arg[0]}, left_table) & CanGrasp(left_franka, left_milk) & IsFull(left_milk)"
        }
        info["add"] = {f"Holding(left_franka, {arg[0]})"}
        info["del_set"] = {"IsHandEmpty(left_franka)", f"On({arg[0]}, left_table)"}
        info["cost"] = 1
        return info