# Put(plate,table) - 错误版本
# 错误：del里没有删除plate在其它所有位置
# 应该删除所有 On(plate,*) 的位置，但这里只删除了 Holding

pre = {
    "Holding(plate)"
}

add = {
    "On(plate,table)"
}

del_set = {
    "Holding(plate)"
}

