from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpLeftRobot(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["big_box"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsHandEmpty(left_robot)", f"On({arg[0]},center_table)", "On(left_robot,left_table)"}
        info["add"] = {f"Holding(left_robot,{arg[0]})"}
        info["del_set"] = {f"IsHandEmpty(left_robot)", f"On({arg[0]},center_table)"}
        info["cost"] = 1
        return info