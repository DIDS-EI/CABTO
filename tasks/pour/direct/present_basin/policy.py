def policy(api):
    pose = api.target_pose()
    api.move_basin(pose)
    api.settle()