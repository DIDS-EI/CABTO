from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceLeftRobot(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["big_box"], ["board", "center_table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(left_robot,{arg[0]})", "On(left_robot,left_table)"}
        info["add"] = {f"IsHandEmpty(left_robot)", f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"Holding(left_robot,{arg[0]})"}
        info["cost"] = 1
        return info