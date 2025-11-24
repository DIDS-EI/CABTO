from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceInMicrowave(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["soup"]

    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Holding the object, microwave is closed
        info["pre"] = {f"Holding({arg[0]})", f"IsClosed(microwave)"}
        # Added: Object is in the microwave, hand is empty
        info["add"] = {f"In({arg[0]},microwave)", f"IsHandEmpty()"}
        # Deleted: Holding the object
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info