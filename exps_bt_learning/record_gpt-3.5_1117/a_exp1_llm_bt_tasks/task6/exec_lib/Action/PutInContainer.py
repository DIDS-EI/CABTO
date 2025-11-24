from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutInContainer(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["pen", "banana", "book"], containers))
    actor = "robot"

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        obj, container = args
        info = {}
        info["pre"] = {f"Holding({cls.actor},{obj})", f"IsOpened({container})"}
        info["add"] = {f"In({obj},{container})", f"IsHandEmpty({cls.actor})"}
        info["del_set"] = {f"Holding({cls.actor},{obj})"}
        info["cost"] = 1
        return info