from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveRightFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_table", "right_table"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        target = arg[0]
        tables = {"left_table", "right_table"}
        info = {}
        info["pre"] = {
            f"On(right_franka,{{t}})".format(t=table) for table in tables
        }
        info["add"] = {f"On(right_franka,{target})"}
        info["del_set"] = {f"On(right_franka,{table})" for table in tables if table != target}
        info["cost"] = 1
        return info


# Conditions