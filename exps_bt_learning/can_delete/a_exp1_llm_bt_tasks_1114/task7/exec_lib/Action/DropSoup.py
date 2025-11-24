from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class DropSoup(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["soup"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(robot,{arg[0]})"}
        info["add"] = {f"In({arg[0]},microwave)"}
        info["del_set"] = {f"Holding(robot,{arg[0]})"}
        info["cost"] = 1
        return info