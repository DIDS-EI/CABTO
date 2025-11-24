from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["red", "yellow"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Hand is empty, object is on the table
        info["pre"] = {f"IsHandEmpty()", f"On({arg[0]},table)"}
        # Added: Holding the object
        info["add"] = {f"Holding({arg[0]})"}
        # Deleted: Hand empty, object on the table
        info["del_set"] = {f"IsHandEmpty()", f"On({arg[0]},table)"}
        info["cost"] = 1
        return info