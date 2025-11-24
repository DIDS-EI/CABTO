from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PourLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_milk"], ["left_cup"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        milk, cup = arg
        info = {}
        # Preconditions:
        # Milk container is On(left_table), IsFull or partially full (here represented as IsFull only in initial)
        # Cup is empty or possibly partially filled (only empty cups in init)
        # Let's require milk container to be On left_table (to pour, must be stable)
        # Hand must be empty (or not holding anything) for pour? Typically yes, pouring is done by placing milk container on table or holding it to pour
        # For simplicity, assume must be On(left_table) and IsFull milk container, and cup empty.
        info["pre"] = {
            f"On({milk},left_table)",
            f"IsFull({milk})",
            f"IsEmpty({cup})"
        }
        # After pouring: cup is half full, milk is half full (milk lost half)
        # We remove IsFull(milk), add IsHalfFull(milk)
        # Remove IsEmpty(cup), add IsHalfFull(cup)
        info["add"] = {f"IsHalfFull({cup})", f"IsHalfFull({milk})"}
        info["del_set"] = {f"IsFull({milk})", f"IsEmpty({cup})"}
        info["cost"] = 1
        return info