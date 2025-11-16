from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUpLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_milk"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: left_franka hand empty, can grasp object, object on some location
        info["pre"] = {
            f"IsHandEmpty(left_franka)",
            f"CanGrasp(left_franka,{arg[0]})",
            f"On({arg[0]},left_table)"
        }
        # Add: left_franka holding object
        info["add"] = {f"Holding(left_franka,{arg[0]})"}
        # Delete: hand empty, object on table
        info["del_set"] = {f"IsHandEmpty(left_franka)", f"On({arg[0]},left_table)"}
        info["cost"] = 1
        return info