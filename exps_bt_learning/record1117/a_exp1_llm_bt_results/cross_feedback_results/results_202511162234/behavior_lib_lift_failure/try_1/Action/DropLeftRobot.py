from exps_bt_learning.a_exp1_llm_bt_results.cross_feedback_results.results_202511162234.behavior_lib_lift_failure.try_1._base.OGAction import OGAction
import itertools

class DropLeftRobot(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["big_box"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(left_robot,{arg[0]})"}
        info["add"] = {f"On({arg[0]},board)", f"IsHandEmpty(left_robot)"}
        info["del_set"] = {f"Holding(left_robot,{arg[0]})"}
        info["cost"] = 1
        return info