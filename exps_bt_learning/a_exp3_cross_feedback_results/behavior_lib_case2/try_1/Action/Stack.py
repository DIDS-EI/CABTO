from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Stack(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["red", "yellow", "green"], ["red", "yellow", "green", "table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        # Preconditions: Hand is holding the first object, and the second object is clear
        info["pre"] = {f"Holding({args[0]})", f"Clear({args[1]})"}
        # Add: The first object is on the second object, hand becomes empty
        info["add"] = {f"On({args[0]},{args[1]})", "IsHandEmpty()"}
        # Delete: The holding status of the first object, and the clear status of the second object
        info["del_set"] = {f"Holding({args[0]})", f"Clear({args[1]})"}
        info["cost"] = 1
        return info