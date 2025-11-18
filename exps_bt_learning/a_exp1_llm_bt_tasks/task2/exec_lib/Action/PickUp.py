from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    # Can pick up any clear object a, c, d, b from the table if hand is empty
    valid_args = ["a", "c", "d", "b"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Hand is empty, object is on the table, object is clear
        info["pre"] = {f"IsHandEmpty()", f"On({arg[0]},table)", f"Clear({arg[0]})"}
        # Postconditions: Holding the object, table is clear
        info["add"] = {f"Holding({arg[0]})", "Clear(table)"}
        # Deleted: IsHandEmpty, On Object, Object clear
        info["del_set"] = {f"IsHandEmpty()", f"On({arg[0]},table)", f"Clear({arg[0]})"}
        info["cost"] = 1
        return info