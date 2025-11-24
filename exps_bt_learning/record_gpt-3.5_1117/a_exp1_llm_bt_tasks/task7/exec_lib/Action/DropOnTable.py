from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGAction import OGAction
import itertools

class DropOnTable(OGAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["chickenleg", "pie", "soup"], ["table", "breakfast_table"]))

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls, *args):
        item, table = args
        # Only pie allowed on breakfast_table
        if table == "breakfast_table" and item != "pie":
            raise ValueError(f"Cannot drop {item} on breakfast_table")
        info = {}
        info["pre"] = {f"Holding({item})"}
        info["add"] = {f"On({item},{table})"}
        info["del_set"] = {
            f"Holding({item})",
            f"In({item},oven)",
            f"In({item},microwave)"
        }
        info["cost"] = 1
        return info