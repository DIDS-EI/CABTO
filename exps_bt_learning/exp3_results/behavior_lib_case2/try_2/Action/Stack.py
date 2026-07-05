from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Stack(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [("red", "green")]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Corrected Preconditions: Holding the object, target location is clear
        info["pre"] = {f"Holding({arg[0]})", f"IsHandEmpty()"}
        # Correct Effects: Object is on the new location, hand becomes empty
        info["add"] = {f"On({arg[0]},{arg[1]})", f"IsHandEmpty()"}
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info