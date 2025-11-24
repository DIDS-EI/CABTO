from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutInLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [("left_lego", "center_big_box")]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        info = {}
        info["pre"] = {f"Holding(left_franka,{args[0]})"}
        info["add"] = {f"In({args[0]},{args[1]})", f"IsHandEmpty(left_franka)"}
        info["del_set"] = {f"Holding(left_franka,{args[0]})"}
        info["cost"] = 1
        return info