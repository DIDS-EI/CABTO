from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceOnBoard(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_robot", "right_robot"], ["big_box"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding({arg[0]},{arg[1]})", f"On({arg[0]},center_table)"}
        info["add"] = {f"On({arg[1]},board)", f"IsHandEmpty({arg[0]})"}
        info["del_set"] = {f"Holding({arg[0]},{arg[1]})"}
        info["cost"] = 1
        return info