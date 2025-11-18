from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_milk", "right_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        actor = "left_franka" if arg[0] == "left_milk" else "right_franka"
        # Preconditions: Hand is empty, Milk is on the corresponding table
        info["pre"] = {f"IsHandEmpty({actor})", f"On({arg[0]},{actor.split('_')[0]}_table)"}
        # Added: Holding milk
        info["add"] = {f"Holding({actor},{arg[0]})"}
        # Deleted: Hand is no longer empty, Milk is moved from table
        info["del_set"] = {f"IsHandEmpty({actor})", f"On({arg[0]},{actor.split('_')[0]}_table)"}
        info["cost"] = 1
        return info