from exps_bt_learning.a_exp1_llm_bt_results.cross_feedback_results.results_202511162234.behavior_lib_lift_failure.try_2._base.OGAction import OGAction
import itertools

class PickUpBigBoxRightRobot(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = [["big_box"]]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {"IsHandEmpty(right_robot)", "On(big_box,center_table)", "On(right_robot,right_table)"}
        info["add"] = {f"Holding(right_robot,{arg[0]})"}
        info["del_set"] = {"IsHandEmpty(right_robot)", "On(big_box,center_table)"}
        info["cost"] = 1
        return info