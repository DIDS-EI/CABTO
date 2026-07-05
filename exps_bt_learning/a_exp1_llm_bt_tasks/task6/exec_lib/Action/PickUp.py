from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["banana", "book", "pen"]
    
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsHandEmpty()", f"On({arg[0]},table)", f"On({arg[0]},breakfast_table)"}
        info["add"] = {f"Holding({arg[0]})"}
        info["del_set"] = {f"IsHandEmpty()", f"On({arg[0]},table)", f"On({arg[0]},breakfast_table)"}
        info["cost"] = 1
        return info