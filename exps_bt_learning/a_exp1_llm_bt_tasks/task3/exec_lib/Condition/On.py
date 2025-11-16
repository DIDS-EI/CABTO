from exps_bt_learning.a_exp1_llm_bt_tasks._base.OGCondition import OGCondition

class On(OGCondition):
    can_be_expanded = True
    num_args = 2
    # Examples: On(left_lego,left_table), On(center_big_box,box_board), On(right_franka,right_table)