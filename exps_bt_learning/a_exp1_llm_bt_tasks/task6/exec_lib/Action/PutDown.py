from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutDown(OGAction):
    can_be_expanded = True
    num_args = 3  # actor, object, destination (location)
    valid_args = list(itertools.product(ACTORS, OBJECTS, LOCATIONS))

    def __init__(self, actor, obj, dest):
        super().__init__(actor, obj, dest)

    @classmethod
    def get_info(cls, actor, obj, dest):
        info = {}
        info["pre"] = {f"Holding({actor},{obj})"}
        info["add"] = {f"IsHandEmpty({actor})", f"On({obj},{dest})"}
        info["del_set"] = {f"Holding({actor},{obj})"}
        info["cost"] = 1
        return info