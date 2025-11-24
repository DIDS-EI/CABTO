from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_milk"], ["left_table", "left_cup"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        milk, dest = arg
        info = {}
        info["pre"] = {
            f"Holding(left_franka,{milk})"
        }
        # When put down on left_table, milk is placed, hand emptied
        # When placing in left_cup, milk is half poured into left_cup, milk remains in hand? No, it is placed (so milk no longer held), hand empty
        # We model pour as a separate action for clarity, here putdown only places the milk container down
        info["add"] = {f"IsHandEmpty(left_franka)", f"On({milk},{dest})"}
        info["del_set"] = {f"Holding(left_franka,{milk})"}
        info["cost"] = 1
        return info