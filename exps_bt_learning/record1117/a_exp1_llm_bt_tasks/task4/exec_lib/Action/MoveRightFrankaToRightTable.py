from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveRightFrankaToRightTable(OGAction):
    can_be_expanded = True
    num_args = 0
    valid_args = []

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls):
        info = {}
        info["pre"] = {"On(right_franka,right_table)"}
        info["add"] = {"On(right_franka,right_table)"}
        info["del_set"] = set()
        info["cost"] = 1
        return info