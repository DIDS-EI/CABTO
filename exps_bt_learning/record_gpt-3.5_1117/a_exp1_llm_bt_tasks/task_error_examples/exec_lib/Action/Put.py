from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Put(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["plate"], ["table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # 错误：del里没有删除plate在其它所有位置
        # 应该删除所有 On(plate,*) 的位置，但这里只删除了 Holding
        info["pre"] = {f"Holding({arg[0]})"}
        info["add"] = {f"On({arg[0]},{arg[1]})"}
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info

