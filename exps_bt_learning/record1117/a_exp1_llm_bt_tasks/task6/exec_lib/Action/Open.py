from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class Open(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["cabinet", "fridge"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        # Preconditions: The container is closed
        info["pre"] = {f"IsClosed({arg[0]})"}
        # Add: The container is open
        info["add"] = {f"IsOpened({arg[0]})"}
        # Delete: The container is closed
        info["del_set"] = {f"IsClosed({arg[0]})"}
        info["cost"] = 1
        return info