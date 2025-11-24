from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["chickenleg", "pie", "soup"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        item = args[0]
        info = {}
        info["pre"] = {"IsHandEmpty()", f"On({item},table)"}
        info["add"] = {f"Holding({item})"}
        info["del_set"] = {"IsHandEmpty()", f"On({item},table)"}
        info["cost"] = 1
        return info