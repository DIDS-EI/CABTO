from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class LiftLeftRobot(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["big_box"], ["board"])) 

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Left robot is holding the big box
        info["pre"] = {f"Holding(left_robot,{arg[0]})"}
        # Added: big box is now on the board
        info["add"] = {f"On({arg[0]},{arg[1]})"}
        # Deleted: big box is no longer on the initial location (e.g., center_table)
        info["del_set"] = {f"On({arg[0]},center_table)"}
        info["cost"] = 1
        return info