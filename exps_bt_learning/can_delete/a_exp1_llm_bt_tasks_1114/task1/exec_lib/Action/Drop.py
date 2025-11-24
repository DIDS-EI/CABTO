from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Drop(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["a", "b", "c"], ["place_a", "place_b", "place_c"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding({arg[0]})"}
        info["add"] = {f"IsHandEmpty()", f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info