from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceRightFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["right_box"], ["right_table", "left_table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        obj, place = arg[0], arg[1]
        info = {}
        info["pre"] = {
            f"IsHolding(right_franka,{obj})",
            # Assume right_franka can place object on any table
        }
        info["add"] = {
            f"IsHandEmpty(right_franka)",
            f"On({obj},{place})"
        }
        info["del_set"] = {
            f"IsHolding(right_franka,{obj})"
        }
        info["cost"] = 1
        return info