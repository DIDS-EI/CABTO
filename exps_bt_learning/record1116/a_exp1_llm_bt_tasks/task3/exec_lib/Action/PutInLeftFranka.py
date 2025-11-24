from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutInLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_lego"], ["center_big_box"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        lego, container = args
        info = dict()
        info["pre"] = {f"Holding(left_franka,{lego})", f"On(left_franka,left_table)"}
        info["add"] = {f"In({lego},{container})", "IsHandEmpty(left_franka)"}
        info["del_set"] = {f"Holding(left_franka,{lego})"}
        info["cost"] = 1
        return info