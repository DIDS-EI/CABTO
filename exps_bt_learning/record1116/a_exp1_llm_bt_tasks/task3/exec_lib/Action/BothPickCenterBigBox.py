from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class BothPickCenterBigBox(OGAction):
    can_be_expanded = True
    num_args = 0
    valid_args = []

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = dict()
        info["pre"] = {
            "IsHandEmpty(left_franka)", "IsHandEmpty(right_franka)",
            "On(left_franka,left_table)", "On(right_franka,right_table)",
            "On(center_big_box,center_table)", "BigBoxNeedTwoFrankaHoldTogether()"
        }
        info["add"] = {"HoldingBoth(left_franka,right_franka,center_big_box)"}
        info["del_set"] = {
            "IsHandEmpty(left_franka)", "IsHandEmpty(right_franka)",
            "On(center_big_box,center_table)", "BigBoxNeedTwoFrankaHoldTogether()"
        }
        info["cost"] = 1
        return info