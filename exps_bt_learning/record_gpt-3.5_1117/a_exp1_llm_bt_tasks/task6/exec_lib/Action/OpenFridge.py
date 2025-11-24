from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class OpenFridge(OGAction):
    can_be_expanded = True
    num_args = 0
    valid_args = [()]
    actor = "robot"

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {f"IsClosed(fridge)", f"IsHandEmpty({cls.actor})"}
        info["add"] = {f"IsOpened(fridge)"}
        info["del_set"] = {f"IsClosed(fridge)"}
        info["cost"] = 1
        return info