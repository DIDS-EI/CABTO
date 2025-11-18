from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutInPlace(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["a", "b", "c"], ["place_a", "place_b", "place_c"]))
    
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Holding the object and the place is empty
        info["pre"] = {f"Holding({arg[0]})", f"IsEmpty({arg[1]})"}
        # Added: Object is in the place and hand is empty
        info["add"] = {f"On({arg[0]},{arg[1]})", "IsHandEmpty()"}
        # Deleted: The object is no longer held and the place is no longer empty
        info["del_set"] = {f"Holding({arg[0]})", f"IsEmpty({arg[1]})"}
        info["cost"] = 1
        return info