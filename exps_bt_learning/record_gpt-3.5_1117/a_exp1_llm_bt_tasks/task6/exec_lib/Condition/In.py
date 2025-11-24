from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGCondition import OGCondition
import itertools

class In(OGCondition):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(["pen", "banana", "book"], containers))