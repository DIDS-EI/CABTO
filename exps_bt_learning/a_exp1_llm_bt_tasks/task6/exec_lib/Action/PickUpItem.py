from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpItem(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["robot"], ["banana", "book", "pen"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsHandEmpty({arg[0]})", f"On({arg[1]},table)", f"On({arg[1]},breakfast_table)"}
        info["add"] = {f"Holding({arg[0]},{arg[1]})"}
        info["del_set"] = {f"IsHandEmpty({arg[0]})", f"On({arg[1]},table)", f"On({arg[1]},breakfast_table)"}
        return info