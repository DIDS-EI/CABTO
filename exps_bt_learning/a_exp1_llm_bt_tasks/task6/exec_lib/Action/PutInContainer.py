from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PutInContainer(OGAction):
    can_be_expanded = True
    num_args = 3  # actor, object, container
    valid_args = list(itertools.product(ACTORS, OBJECTS, CONTAINERS))

    def __init__(self, actor, obj, container):
        super().__init__(actor, obj, container)

    @classmethod
    def get_info(cls, actor, obj, container):
        info = {}
        info["pre"] = {f"Holding({actor},{obj})", f"IsOpened({container})"}
        info["add"] = {f"IsHandEmpty({actor})", f"In({obj},{container})"}
        info["del_set"] = {f"Holding({actor},{obj})"}
        info["cost"] = 1
        return info