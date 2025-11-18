from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveRightFrankaToCenter(OGAction):
    can_be_expanded = True
    num_args = 0
    valid_args = []

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls):
        info = {}
        info["pre"] = {f"On(right_franka,right_table)"}
        info["add"] = {f"On(right_franka,center_table)"}
        info["del_set"] = {f"On(right_franka,right_table)"}
        info["cost"] = 1
        return info