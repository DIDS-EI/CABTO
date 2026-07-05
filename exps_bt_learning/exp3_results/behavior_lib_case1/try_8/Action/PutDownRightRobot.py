from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownRightRobot(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["big_box"], ["center_table", "board"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(right_robot,{arg[0]})"}
        info["add"] = {f"On({arg[0]},{arg[1]})", f"IsHandEmpty(right_robot)"}
        info["del_set"] = {f"Holding(right_robot,{arg[0]})"}
        info["cost"] = 1
        return info