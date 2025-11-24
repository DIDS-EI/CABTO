from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveRobotToCenterTable(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_robot", "right_robot"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"On({arg[0]},left_table)", f"On({arg[0]},right_table)"}
        info["add"] = {f"On({arg[0]},center_table)"}
        info["del_set"] = {f"On({arg[0]},left_table)", f"On({arg[0]},right_table)"}
        info["cost"] = 1
        return info