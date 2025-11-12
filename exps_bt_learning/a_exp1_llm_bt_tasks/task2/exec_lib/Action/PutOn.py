from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction

class PutOn(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = ["a", "b", "c", "d", "table"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        x, y = arg[0], arg[1]
        info["pre"] = {f"IsHolding({x})", f"IsNear({y})", f"Clear({y})"}
        info["add"] = {f"On({x},{y})", "IsHandEmpty()", f"Clear({x})"}
        info["del_set"] = {f"IsHolding({x})", f"Clear({y})"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= self.info["add"]
        self.agent.condition_set -= self.info["del_set"]