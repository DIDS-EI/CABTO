from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceInLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [("left_lego", "center_big_box")]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding(left_franka,{arg[0]})", f"On(left_franka,center_table)", f"On({arg[1]},center_table)"}
        info["add"] = {f"In({arg[0]},{arg[1]})", f"IsHandEmpty(left_franka)"}
        info["del_set"] = {f"Holding(left_franka,{arg[0]})"}
        info["cost"] = 1
        return info