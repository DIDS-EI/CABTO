def policy(api):
    # Approach the target pose for the lid handle
    target_pose = api.target_pose()
    
    # Align the hand horizontally above the target pose
    api.align_xy(target_pose)
    
    # Descend to the lid handle and lock the hand horizontally
    api.descend_to(target_pose[2], (target_pose[0], target_pose[1]))
    
    # Grasp the lid handle
    api.grasp(target_pose)
    
    # Record the offset for the held lid
    api.record_grab_offset()
    
    # Lift the lid to transport height
    api.lift()