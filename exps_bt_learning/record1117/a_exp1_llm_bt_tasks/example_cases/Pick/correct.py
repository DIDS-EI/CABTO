# Pick(left_robot,right_lego) - 正确版本
# 正确：因为实际上图片里是够不着的，所以不应该有抓取效果
# 应该保持原状，或者返回失败

# 因为够不着，所以不应该添加 Holding
pre = {
    "IsHandEmpty()",
    "On(right_lego,table)"
}

# 因为够不着，所以不应该添加 Holding
add = set()

# 因为够不着，所以不应该删除任何状态
del_set = set()

