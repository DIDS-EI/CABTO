from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["a", "b", "c", "d"]  # objects can be picked up from table

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj = arg[0]
        info = {}
        info["pre"] = {f"On({obj},table)", "IsHandEmpty()"}
        info["add"] = {f"Holding({obj})"}
        info["del_set"] = {f"On({obj},table)", "IsHandEmpty()"}
        info["cost"] = 1
        return info