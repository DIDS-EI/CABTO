from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceBackOnTable(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["a", "b", "c"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj = arg[0]
        info = {}
        # Preconditions: holding object
        info["pre"] = {f"Holding({obj})"}
        # Add: hand empty, object on table
        info["add"] = {f"IsHandEmpty()", f"On({obj},table)"}
        # Delete: holding object
        info["del_set"] = {f"Holding({obj})"}
        info["cost"] = 1
        return info