from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class InsertInDevice(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["chickenleg", "pie", "soup"], ["oven", "microwave"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        item, device = args
        info = {}
        info["pre"] = {f"Holding({item})", f"IsOpened({device})"}
        info["add"] = {f"In({item},{device})"}
        info["del_set"] = {f"Holding({item})", f"On({item},table)"}
        info["cost"] = 1
        return info