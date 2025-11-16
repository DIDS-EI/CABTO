from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDownLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_milk"], ["left_table", "left_cup"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj, loc = arg
        info = {}
        # Preconditions: holding obj, loc available or empty (for cup), and appropriate place
        info["pre"] = {
            f"Holding(left_franka,{obj})",
        }
        if loc == "left_cup":
            info["pre"].add(f"IsEmpty(left_cup)")
        elif loc == "left_table":
            info["pre"].add(f"On(left_cup,left_table)")  # cup must be on table to pour into it
        # Add: object on loc, hand empty, cup no longer empty if putting milk in it
        if loc == "left_cup":
            info["add"] = {f"IsHandEmpty(left_franka)", f"IsHalfFull(left_cup)"}
            info["del_set"] = {f"Holding(left_franka,{obj})", f"IsEmpty(left_cup)"}
        else:
            info["add"] = {f"IsHandEmpty(left_franka)", f"On({obj},{loc})"}
            info["del_set"] = {f"Holding(left_franka,{obj})"}
        info["cost"] = 1
        return info