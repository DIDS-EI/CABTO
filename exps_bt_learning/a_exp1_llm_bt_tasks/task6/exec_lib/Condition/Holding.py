from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGCondition import OGCondition

class Holding(OGCondition):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(ACTORS, OBJECTS))