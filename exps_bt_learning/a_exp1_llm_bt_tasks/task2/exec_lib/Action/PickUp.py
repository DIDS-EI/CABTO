from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction

class PickUp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["a", "b", "c", "d"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        x = arg[0]
        info["pre"] = {"IsHandEmpty()", f"On({x},table)", f"Clear({x})", f"IsNear({x})"}
        info["add"] = {f"IsHolding({x})"}
        info["del_set"] = {"IsHandEmpty()", f"On({x},table)", f"Clear({x})"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]