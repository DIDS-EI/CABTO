from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDown(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_milk", "right_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        actor = "left_franka" if arg[0] == "left_milk" else "right_franka"
        # Preconditions: Holding milk
        info["pre"] = {f"Holding({actor},{arg[0]})"}
        # Added: Milk is on the table
        info["add"] = {f"On({arg[0]},{actor.split('_')[0]}_table)"}
        # Deleted: No longer holding milk
        info["del_set"] = {f"Holding({actor},{arg[0]})"}
        info["cost"] = 1
        return info