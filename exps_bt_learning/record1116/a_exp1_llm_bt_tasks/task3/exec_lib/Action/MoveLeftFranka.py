from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class MoveLeftFranka(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(
        ["left_table", "center_table", "box_board", "center_big_box"],
        ["left_table", "center_table", "box_board", "center_big_box"]
    ))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        franka_from, franka_to = args
        if franka_from == franka_to:  # invalid move
            return {"pre": set(), "add": set(), "del_set": set(), "cost": 0}
        info = dict()
        info["pre"] = {f"On(left_franka,{franka_from})", f"IsHandEmpty(left_franka)"}
        info["add"] = {f"On(left_franka,{franka_to})"}
        info["del_set"] = {f"On(left_franka,{franka_from})"}
        info["cost"] = 1
        return info