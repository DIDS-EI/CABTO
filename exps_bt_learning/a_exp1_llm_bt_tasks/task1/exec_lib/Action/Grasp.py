from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction

class Grasp(OGAction):
    can_be_expanded = True
    num_args = 1
    valid_args = ["a", "b", "c"]

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {"IsHandEmpty()", f"IsNear({arg[0]})"}
        info["add"] = {f"IsHolding({arg[0]})"}
        info["del_set"] = {"IsHandEmpty()"}
        info["cost"] = 1
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]