from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceItemIn(OGAction):
    can_be_expanded = True
    num_args = 3
    valid_args = list(itertools.product(["robot"], ["book", "banana"], ["cabinet", "fridge"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {f"Holding({arg[0]},{arg[1]})", f"IsOpened({arg[2]})"}
        info["add"] = {f"In({arg[1]},{arg[2]})", f"IsHandEmpty({arg[0]})"}
        info["del_set"] = {f"Holding({arg[0]},{arg[1]})"}
        return info