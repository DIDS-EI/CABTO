from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Unstack(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["red", "yellow", "green"], ["red", "yellow", "green", "table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        # Preconditions: The first object is on the second object, hand is empty, first object is clear
        info["pre"] = {f"On({args[0]},{args[1]})", "IsHandEmpty()", f"Clear({args[0]})"}
        # Add: Holding the first object, clear the second object
        info["add"] = {f"Holding({args[0]})", f"Clear({args[1]})"}
        # Delete: The first object is on the second object, hand is empty
        info["del_set"] = {f"On({args[0]},{args[1]})", "IsHandEmpty()"}
        info["cost"] = 1
        return info