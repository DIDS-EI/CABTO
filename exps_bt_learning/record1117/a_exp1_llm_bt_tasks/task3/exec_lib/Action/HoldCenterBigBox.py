from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class HoldCenterBigBox(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = [("center_big_box",)]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {f"On({args[0]},center_table)", "IsHandEmpty(left_franka)", "IsHandEmpty(right_franka)",
                       "BigBoxNeedTwoFrankaHoldTogether()"}
        info["add"] = {f"Holding(left_franka,{args[0]})", f"Holding(right_franka,{args[0]})"}
        info["del_set"] = {f"IsHandEmpty(left_franka)", f"IsHandEmpty(right_franka)", f"On({args[0]},center_table)"}
        info["cost"] = 1
        return info