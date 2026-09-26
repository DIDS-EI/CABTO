def policy(api):
    pose = api.target_pose()
    top_z = pose[2]
    xyz = pose[:2]
    tcp_xy = api.move_above(xyz)
    if api.refine_above(tcp_xy) is not None:
        tcp_xy = api.refine_above(tcp_xy)
    api.descend_place(top_z, tcp_xy)
    api.place_release()
    api.retreat()
    api.settle()