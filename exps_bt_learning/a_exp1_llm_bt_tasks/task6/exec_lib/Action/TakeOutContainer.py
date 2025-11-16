from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class TakeOutContainer(OGAction):
    can_be_expanded = True
    num_args = 4  # actor, object, container, destination location
    valid_args = list(itertools.product(ACTORS, OBJECTS, CONTAINERS, LOCATIONS))

    def __init__(self, actor, obj, container, dest):
        super().__init__(actor, obj, container, dest)

    @classmethod
    def get_info(cls, actor, obj, container, dest):
        info = {}
        info["pre"] = {f"IsHandEmpty({actor})", f"In({obj},{container})", f"IsOpened({container})"}
        info["add"] = {f"Holding({actor},{obj})"}
        info["del_set"] = {f"IsHandEmpty({actor})", f"In({obj},{container})"}
        info["cost"] = 1
        return info

##########
# CONDITIONS
##########