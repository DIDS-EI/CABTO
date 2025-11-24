# Error Examples Task

这个文件夹包含了5个带有错误动作定义的任务示例，用于测试和验证动作修正功能。

## 错误示例列表

### 1. PutIn(apple,cabinet)
- **错误类型**: pre缺少条件
- **错误描述**: pre缺少 `IsOpened(cabinet)`
- **Goal**: `In(apple,cabinet)`
- **文件**: `exec_lib/Action/PutIn.py`

### 2. Stack(red,green)
- **错误类型**: pre缺少条件
- **错误描述**: pre缺少 `Clear(green)`
- **Goal**: `On(red,green)`
- **文件**: `exec_lib/Action/Stack.py`

### 3. Lift(big_box,board)
- **错误类型**: pre缺少条件
- **错误描述**: pre缺少 `IsHolding(leftrobot,big_box)`
- **Goal**: `On(big_box,board)`
- **文件**: `exec_lib/Action/Lift.py`

### 4. Pick(left_robot,right_lego)
- **错误类型**: add/del错误
- **错误描述**: add和del里物体被抓起来了，但实际上图片里是够不着的
- **Goal**: `Holding(left_robot,right_lego)`
- **文件**: `exec_lib/Action/Pick.py`

### 5. Put(plate,table)
- **错误类型**: del不完整
- **错误描述**: del里没有删除plate在其它所有位置
- **Goal**: `On(plate,table)`
- **文件**: `exec_lib/Action/Put.py`

## 条件类

所有需要的条件类都在 `exec_lib/Condition/` 目录下：
- `In.py` - 物体在容器内
- `IsOpened.py` - 容器是否打开
- `Holding.py` - 机器人是否持有物体
- `IsHandEmpty.py` - 手是否为空
- `On.py` - 物体在表面上
- `Clear.py` - 是否为表面
- `IsHolding.py` - 是否持有（用于机器人）

## Goal定义

所有goal定义在 `goals.py` 文件中。

