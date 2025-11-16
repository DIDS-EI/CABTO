from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 3  # actor, object, source (location or container)
    valid_args = list(itertools.product(ACTORS, OBJECTS, CONTAINERS + LOCATIONS))

    def __init__(self, actor, obj, source):
        super().__init__(actor, obj, source)

    @classmethod
    def get_info(cls, actor, obj, source):
        info = {}
        pre = {f"IsHandEmpty({actor})"}
        if source in CONTAINERS:
            pre.add(f"IsOpened({source})")
            pre.add(f"In({obj},{source})")
        else:
            pre.add(f"On({obj},{source})")
        info["pre"] = pre
        info["add"] = {f"Holding({actor},{obj})"}
        dele = {f"IsHandEmpty({actor})"}
        if source in CONTAINERS:
            dele.add(f"In({obj},{source})")
        else:
            dele.add(f"On({obj},{source})")
        info["del_set"] = dele
        info["cost"] = 1
        return info