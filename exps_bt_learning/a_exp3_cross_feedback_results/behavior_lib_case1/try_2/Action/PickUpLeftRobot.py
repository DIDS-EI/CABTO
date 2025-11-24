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
        # Preconditions: left robot's hand is empty and big box is on the center table
        info["pre"] = {f"IsHandEmpty(left_robot)", f"On({arg[0]},center_table)"}
        # Added: left robot is now holding the big box
        info["add"] = {f"Holding(left_robot,{arg[0]})"}
        # Deleted: left robot's hand is no longer empty, and big box is no longer on the center table
        info["del_set"] = {f"IsHandEmpty(left_robot)", f"On({arg[0]},center_table)"}
        info["cost"] = 1
        return info