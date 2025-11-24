from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["a", "b", "c"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj = arg[0]
        info = {}
        # Preconditions: hand empty, object on table
        info["pre"] = {f"IsHandEmpty()", f"On({obj},table)"}
        # Add: holding the object
        info["add"] = {f"Holding({obj})"}
        # Delete: hand empty, object on table
        info["del_set"] = {f"IsHandEmpty()", f"On({obj},table)"}
        info["cost"] = 1
        return info