from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceRightRobot(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["big_box"], ["board", "center_table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(right_robot,{arg[0]})", "On(right_robot,right_table)"}
        info["add"] = {f"IsHandEmpty(right_robot)", f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"Holding(right_robot,{arg[0]})"}
        info["cost"] = 1
        return info