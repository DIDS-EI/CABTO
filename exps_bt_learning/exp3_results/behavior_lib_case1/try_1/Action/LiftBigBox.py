from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class LiftBigBox(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [("big_box", "board")]

    def __init__(self, *args):
        super().__init__(*args)
        
    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Both robots must be free to work together on lifting the box and the box should be on the center_table, ready to be lifted to the board
        info["pre"] = {f"IsHandEmpty(left_robot)", f"IsHandEmpty(right_robot)", f"On(big_box,center_table)"}
        # Add: Box is on the board
        info["add"] = {f"On(big_box,{arg[1]})"}
        # Delete: Box is on the center table
        info["del_set"] = {f"On(big_box,center_table)"}
        info["cost"] = 1
        return info