from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Pick(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["left_robot"], ["right_lego"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # 错误：add和del里物体被抓起来了，但实际上图片里是够不着的
        # 所以不应该有这些效果，但这里错误地添加了
        info["pre"] = {f"IsHandEmpty()", f"On({arg[1]},table)"}
        info["add"] = {f"Holding({arg[0]},{arg[1]})"}
        info["del_set"] = {f"IsHandEmpty()", f"On({arg[1]},table)"}
        info["cost"] = 1
        return info

