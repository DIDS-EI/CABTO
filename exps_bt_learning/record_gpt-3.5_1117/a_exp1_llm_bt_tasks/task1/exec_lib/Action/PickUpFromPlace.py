from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpFromPlace(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["a", "b", "c"], ["place_a", "place_b", "place_c"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj = arg[0]
        place = arg[1]
        info = {}
        # Preconditions: hand empty, object on place
        info["pre"] = {f"IsHandEmpty()", f"On({obj},{place})"}
        # Add: holding object, place empty
        info["add"] = {f"Holding({obj})", f"IsEmpty({place})"}
        # Delete: hand empty, object on place
        info["del_set"] = {f"IsHandEmpty()", f"On({obj},{place})"}
        info["cost"] = 1
        return info