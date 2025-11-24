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
        # Preconditions: Right robot's hand is empty and the big box is on center table
        info["pre"] = {f"IsHandEmpty(right_robot)", f"On(big_box,center_table)"}
        # Add: Right robot is holding the big box
        info["add"] = {f"Holding(right_robot,{arg[0]})"}
        # Delete: Right robot's hand is empty, big box is on center table
        info["del_set"] = {f"IsHandEmpty(right_robot)", f"On(big_box,center_table)"}
        info["cost"] = 1
        return info