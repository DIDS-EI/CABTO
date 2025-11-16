from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGCondition import OGCondition

class HoldingBoth(OGCondition):
    can_be_expanded = True
    num_args = 3
    # HoldingBoth(left_franka,right_franka,center_big_box)