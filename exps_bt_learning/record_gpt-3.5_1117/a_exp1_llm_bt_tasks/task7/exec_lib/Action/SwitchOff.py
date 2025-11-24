from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class SwitchOff(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["oven", "radio", "microwave", "light"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        device = args[0]
        info = {}
        info["pre"] = {f"IsSwitchedOn({device})", "IsHandEmpty()"}
        info["add"] = {f"IsSwitchedOff({device})"}
        info["del_set"] = {f"IsSwitchedOn({device})"}
        info["cost"] = 1
        return info