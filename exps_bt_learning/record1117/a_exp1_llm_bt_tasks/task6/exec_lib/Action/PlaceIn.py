from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PlaceIn(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["book", "banana"], ["cabinet", "fridge"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: Holding the object and the container is open
        info["pre"] = {f"Holding({arg[0]})", f"IsOpened({arg[1]})"}
        # Add: Object is in the container, Hand is empty
        info["add"] = {f"In({arg[0]}, {arg[1]})", f"IsHandEmpty()"}
        # Delete: Holding the object
        info["del_set"] = {f"Holding({arg[0]})"}
        info["cost"] = 1
        return info