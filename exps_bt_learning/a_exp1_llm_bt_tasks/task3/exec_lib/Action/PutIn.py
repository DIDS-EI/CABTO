from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutIn(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_lego", "right_lego"], ["center_big_box"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding({arg[0]})", f"On({arg[0].split('_')[0]}_franka,{arg[1]})"}
        info["add"] = {f"In({arg[0]},{arg[1]})", f"IsHandEmpty({arg[0].split('_')[0]}_franka)"}
        info["del_set"] = {f"Holding({arg[0]}, {arg[1]})"}
        info["cost"] = 1
        return info