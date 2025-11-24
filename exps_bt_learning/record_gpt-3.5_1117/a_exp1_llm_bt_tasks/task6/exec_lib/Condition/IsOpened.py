from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGCondition import OGCondition
import itertools

class IsOpened(OGCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = list(containers)