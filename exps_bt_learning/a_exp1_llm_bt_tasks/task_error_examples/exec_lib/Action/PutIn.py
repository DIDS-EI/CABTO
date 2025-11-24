from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutIn(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["apple"], ["cabinet"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # 错误：pre缺少 IsOpened(cabinet)
        info["pre"] = {f"Holding({arg[0]})"}
        info["add"] = {f"In({arg[0]},{arg[1]})"}
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info

