from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDown(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["red", "yellow", "green"], ["table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        # Preconditions: Holding the object
        info["pre"] = {f"Holding({args[0]})"}
        # Add: The object is on the table and hand is empty
        info["add"] = {f"On({args[0]},{args[1]})", "IsHandEmpty()"}
        # Delete: The holding status of the object
        info["del_set"] = {f"Holding({args[0]})"}
        info["cost"] = 1
        return info