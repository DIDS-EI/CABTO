from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGCondition import OGCondition

class IsHolding(OGCondition):
    can_be_expanded = True
    num_args = 2  # actor, object