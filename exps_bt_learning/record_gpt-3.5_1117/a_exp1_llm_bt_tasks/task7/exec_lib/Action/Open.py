from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Open(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["oven", "microwave"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        device = args[0]
        info = {}
        info["pre"] = {f"IsClosed({device})"}
        info["add"] = {f"IsOpened({device})"}
        info["del_set"] = {f"IsClosed({device})"}
        info["cost"] = 1
        return info