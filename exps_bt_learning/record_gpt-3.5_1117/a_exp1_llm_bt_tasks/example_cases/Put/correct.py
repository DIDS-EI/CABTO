# Put(plate,table) - 正确版本
# 正确：del应该删除plate在其它所有位置
# 需要删除所有 On(plate,*) 的位置（除了目标位置table）

pre = {
    "Holding(plate)"
}

add = {
    "On(plate,table)"
}

# 应该删除所有 On(plate,*) 的位置（除了table）
# 例如：On(plate,other_surface), On(plate,cabinet) 等
# 注意：实际实现时需要删除所有 On(plate,location) 的位置（location != table）
del_set = {
    "Holding(plate)",
    # 这里应该包含所有可能的 On(plate,location) 位置（location != table）
    # 例如：On(plate,other_surface), On(plate,cabinet), On(plate,board) 等
    # 实际使用时需要根据当前状态动态添加所有 On(plate,*) 的位置
}

