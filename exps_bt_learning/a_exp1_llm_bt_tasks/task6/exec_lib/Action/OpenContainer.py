from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class OpenContainer(OGAction):
    can_be_expanded = True
    num_args = 2  # actor, container
    valid_args = list(itertools.product(ACTORS, CONTAINERS))

    def __init__(self, actor, container):
        super().__init__(actor, container)

    @classmethod
    def get_info(cls, actor, container):
        info = {}
        info["pre"] = {f"IsHandEmpty({actor})", f"IsClosed({container})"}
        info["add"] = {f"IsOpened({container})"}
        info["del_set"] = {f"IsClosed({container})"}
        info["cost"] = 1
        return info