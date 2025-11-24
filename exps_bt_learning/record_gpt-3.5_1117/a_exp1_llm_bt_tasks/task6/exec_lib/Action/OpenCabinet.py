from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class OpenCabinet(OGAction):
    can_be_expanded = True
    num_args = 0
    valid_args = [()]
    actor = "robot"

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {f"IsClosed(cabinet)", f"IsHandEmpty({cls.actor})"}
        info["add"] = {f"IsOpened(cabinet)"}
        info["del_set"] = {f"IsClosed(cabinet)"}
        info["cost"] = 1
        return info