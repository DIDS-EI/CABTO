from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Lift(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["big_box"], ["board"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Robot1 is holding the object, and it is on center_table
        info["pre"] = {f"Holding(left_robot,{arg[0]})", f"On({arg[0]},center_table)", f"On(left_robot,left_table)"}
        # Added: Object is on the destination location
        info["add"] = {f"On({arg[0]},{arg[1]})"}
        # Deleted: Object is on center_table, also indicating the lift action
        info["del_set"] = {f"On({arg[0]},center_table)", f"Holding(left_robot,{arg[0]})"}
        info["cost"] = 1
        return info