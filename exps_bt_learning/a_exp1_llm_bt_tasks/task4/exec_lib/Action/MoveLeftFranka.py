from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["left_table", "right_table"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        target = arg[0]
        # source is the current table the left_franka is on. To represent all possible moves, del_set removes left_franka from all tables except target.
        tables = {"left_table", "right_table"}
        info = {}
        info["pre"] = {
            f"On(left_franka,{{t}})".format(t=table) for table in tables
        }
        # Actually in initial state left_franka is on left_table, so only that case is valid to start.
        # After moving left_franka to target table:
        info["add"] = {f"On(left_franka,{target})"}
        info["del_set"] = {f"On(left_franka,{table})" for table in tables if table != target}
        info["cost"] = 1
        return info