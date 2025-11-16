from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class SwitchOn(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["oven", "microwave", "light", "radio"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: The device is switched off
        info["pre"] = {f"IsSwitchedOff({arg[0]})"}
        # Added: The device is switched on
        info["add"] = {f"IsSwitchedOn({arg[0]})"}
        # Deleted: The device is switched off
        info["del_set"] = {f"IsSwitchedOff({arg[0]})"}
        info["cost"] = 1
        return info