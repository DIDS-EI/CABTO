from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PourLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_milk"], ["left_cup"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        milk, cup = arg
        info = {}
        # Preconditions: left_franka holding milk, cup empty, cup and milk at left_table or left_franka holding milk
        info["pre"] = {
            f"Holding(left_franka,{milk})",
            f"IsEmpty({cup})",
            f"CupNeedGraspAndSupport()"
        }
        # After pour: cup half full, milk no longer full (assumed)
        info["add"] = {
            f"IsHalfFull({cup})",
            f"IsHandEmpty(left_franka)",
            f"On({milk},left_table)"
        }
        # Deletes: holding milk, cup empty, milk full (assumed poured partial)
        info["del_set"] = {
            f"Holding(left_franka,{milk})",
            f"IsEmpty({cup})",
            f"IsFull({milk})",
        }
        info["cost"] = 1
        return info