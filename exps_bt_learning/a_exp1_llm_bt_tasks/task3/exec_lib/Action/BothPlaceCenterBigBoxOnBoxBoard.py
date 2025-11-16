from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class BothPlaceCenterBigBoxOnBoxBoard(OGAction):
    can_be_expanded = True
    num_args = 0
    valid_args = []

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = dict()
        info["pre"] = {
            "HoldingBoth(left_franka,right_franka,center_big_box)",
            "On(left_franka,left_table)", "On(right_franka,right_table)",
            "On(box_board,center_table)"
        }
        info["add"] = {
            "IsHandEmpty(left_franka)", "IsHandEmpty(right_franka)",
            "On(center_big_box,box_board)"
        }
        info["del_set"] = {
            "HoldingBoth(left_franka,right_franka,center_big_box)",
            "On(box_board,center_table)"
        }
        info["cost"] = 1
        return info


### Condition Nodes ###