from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class OpenContainer(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["robot"], ["cabinet", "fridge"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"IsClosed({arg[1]})"}
        info["add"] = {f"IsOpened({arg[1]})"}
        info["del_set"] = {f"IsClosed({arg[1]})"}
        return info