# Goal definitions for error examples

# Example 1: PutIn(apple,cabinet)
# Goal: In(apple,cabinet)
goal_putin = "In(apple,cabinet)"

# Example 2: Stack(red,green)
# Goal: On(red,green)
goal_stack = "On(red,green)"

# Example 3: Lift(big_box,board)
# Goal: On(big_box,board) (assuming lift means placing on board)
goal_lift = "On(big_box,board)"

# Example 4: Pick(left_robot,right_lego)
# Goal: Holding(left_robot,right_lego)
goal_pick = "Holding(left_robot,right_lego)"

# Example 5: Put(plate,table)
# Goal: On(plate,table)
goal_put = "On(plate,table)"

# All goals in a dictionary
all_goals = {
    "putin": goal_putin,
    "stack": goal_stack,
    "lift": goal_lift,
    "pick": goal_pick,
    "put": goal_put
}

