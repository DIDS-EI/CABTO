from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownOnPlace(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["a", "b", "c"], ["place_a", "place_b", "place_c"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj, place = arg
        info = {}
        # Preconditions: holding object, place empty
        info["pre"] = {f"Holding({obj})", f"IsEmpty({place})"}
        # Add: hand empty, object on place
        info["add"] = {"IsHandEmpty()", f"On({obj},{place})"}
        # Delete: holding object, place empty
        info["del_set"] = {f"Holding({obj})", f"IsEmpty({place})"}
        info["cost"] = 1
        return info