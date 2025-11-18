from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Pour(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [("right_milk", "right_cup"), ("left_milk", "left_cup")]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Milk is full, CupNeedGraspAndSupport, Cup is empty
        info["pre"] = {f"IsFull({arg[0]})", "CupNeedGraspAndSupport()", f"IsEmpty({arg[1]})"}
        # Added: Cup is half full
        info["add"] = {f"IsHalfFull({arg[1]})"}
        # Deleted: Cup is no longer empty
        info["del_set"] = {f"IsEmpty({arg[1]})"}
        info["cost"] = 1
        return info