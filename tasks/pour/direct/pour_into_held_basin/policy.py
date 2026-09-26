def policy(api):
    source_pose = api.source_pose()
    move_source = api.move_source(source_pose)
    receiver_pose = api.receiver_pose()
    align_receiver = api.align_receiver(receiver_pose)
    tip_source = api.tip_source()
    settle = api.settle()