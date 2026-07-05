from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutOn(OGAction):
    can_be_expanded = True
    num_args = 2
    # Can put any object onto another object
    valid_args = list(itertools.product(["a", "c", "d", "b"], ["a", "c", "d", "b", "table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Holding an object, target surface is clear
        info["pre"] = {f"Holding({arg[0]})", f"Clear({arg[1]})"}
        # Added: Object is on the target surface, hand is empty
        info["add"] = {f"On({arg[0]},{arg[1]})", "IsHandEmpty()"}
        # Deleted: Hand is holding the object, target surface is clear
        info["del_set"] = {f"Holding({arg[0]})", f"Clear({arg[1]})"}
        info["cost"] = 1
        return info