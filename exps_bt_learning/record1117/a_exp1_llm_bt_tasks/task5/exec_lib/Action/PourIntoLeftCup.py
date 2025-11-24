from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PourIntoLeftCup(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Holding milk, Cup needs grasp and support
        info["pre"] = {f"Holding(left_franka,{arg[0]})", "CupNeedGraspAndSupport()"}
        # Added: Cup is half full
        info["add"] = {"IsHalfFull(left_cup)"}
        # Deleted: Cup is empty
        info["del_set"] = {"IsEmpty(left_cup)"}
        info["cost"] = 1
        return info