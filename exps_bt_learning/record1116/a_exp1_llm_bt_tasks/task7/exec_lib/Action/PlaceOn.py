from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceOn(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["pie"], ["breakfast_table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Holding the object
        info["pre"] = {f"Holding({arg[0]})"}
        # Added: Object is on the destination, hand is empty
        info["add"] = {f"On({arg[0]},{arg[1]})", f"IsHandEmpty()"}
        # Deleted: Holding the object
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info