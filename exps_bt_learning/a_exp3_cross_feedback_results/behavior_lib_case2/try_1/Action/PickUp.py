from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["red", "yellow", "green"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        # Preconditions: The object is on the table, it is clear, and hand is empty
        info["pre"] = {f"On({args[0]},table)", f"Clear({args[0]})", "IsHandEmpty()"}
        # Add: Holding the object
        info["add"] = {f"Holding({args[0]})"}
        # Delete: The object is on the table, hand is empty
        info["del_set"] = {f"On({args[0]},table)", "IsHandEmpty()"}
        info["cost"] = 1
        return info