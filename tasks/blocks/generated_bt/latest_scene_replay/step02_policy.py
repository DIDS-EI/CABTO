def policy(api):
    api.locate("green_block")
    api.approach(api.locate("green_block")[0:3])
    api.align_xy(api.locate("green_block")[0:3])
    api.descend_to(api.locate("green_block")[2], api.locate("green_block")[0:2])
    api.grasp(api.locate("green_block")[0:2])
    api.record_grab_offset("green_block")
    api.lift()