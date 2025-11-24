# Pick(left_robot,right_lego) - 错误版本
# 错误：add和del里物体被抓起来了，但实际上图片里是够不着的
# 所以不应该有这些效果，但这里错误地添加了

pre = {
    "IsHandEmpty()",
    "On(right_lego,table)"
}

add = {
    "Holding(left_robot,right_lego)"
}

del_set = {
    "IsHandEmpty()",
    "On(right_lego,table)"
}

