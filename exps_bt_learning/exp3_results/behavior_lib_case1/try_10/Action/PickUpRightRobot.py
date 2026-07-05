from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpRightRobot(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["big_box"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Robot right hand is empty and object is on center_table
        info["pre"] = {f"IsHandEmpty(right_robot)", f"On({arg[0]},center_table)", f"On(right_robot,right_table)"}
        # Added: Robot right hand is holding the object
        info["add"] = {f"Holding(right_robot,{arg[0]})"}
        # Deleted: Robot right hand is empty
        info["del_set"] = {f"IsHandEmpty(right_robot)"}
        info["cost"] = 1
        return info