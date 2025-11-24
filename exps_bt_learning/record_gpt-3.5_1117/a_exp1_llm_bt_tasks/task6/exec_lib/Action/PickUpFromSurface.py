from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpFromSurface(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["pen", "banana", "book"], surfaces))
    actor = "robot"

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        obj, surface = args
        info = {}
        info["pre"] = {f"IsHandEmpty({cls.actor})", f"On({obj},{surface})"}
        info["add"] = {f"Holding({cls.actor},{obj})"}
        info["del_set"] = {f"IsHandEmpty({cls.actor})", f"On({obj},{surface})"}
        info["cost"] = 1
        return info