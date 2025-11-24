from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class DropToCabinet(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["pen"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding({arg[0]})", "IsClosed(cabinet)"}
        info["add"] = {f"In({arg[0]},cabinet)", "IsContainerEmpty(cabinet)"}
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info