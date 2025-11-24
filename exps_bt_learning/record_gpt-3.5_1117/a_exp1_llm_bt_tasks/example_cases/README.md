# Example Cases - 动作定义错误和正确示例

这个文件夹包含了5个动作的错误和正确版本的pre、add和del定义，用于对比学习和错误修正。

## 示例列表

### 1. PutIn(apple,cabinet)
- **错误类型**: pre缺少条件
- **错误描述**: pre缺少 `IsOpen(cabinet)`
- **文件**: 
  - `PutIn/wrong.py` - 错误版本（缺少IsOpen条件）
  - `PutIn/correct.py` - 正确版本（包含IsOpen条件）

### 2. Stack(red,green)
- **错误类型**: pre缺少条件
- **错误描述**: pre缺少 `Clear(green)`
- **文件**: 
  - `Stack/wrong.py` - 错误版本（缺少Clear条件）
  - `Stack/correct.py` - 正确版本（包含Clear条件）

### 3. Lift(big_box,board)
- **错误类型**: pre缺少条件
- **错误描述**: pre缺少 `IsHolding(leftrobot,big_box)`
- **文件**: 
  - `Lift/wrong.py` - 错误版本（缺少IsHolding条件）
  - `Lift/correct.py` - 正确版本（包含IsHolding条件）

### 4. Pick(left_robot,right_lego)
- **错误类型**: add/del错误
- **错误描述**: add和del里物体被抓起来了，但实际上图片里是够不着的
- **文件**: 
  - `Pick/wrong.py` - 错误版本（错误地添加了抓取效果）
  - `Pick/correct.py` - 正确版本（因为够不着，所以不应该有抓取效果）

### 5. Put(plate,table)
- **错误类型**: del不完整
- **错误描述**: del里没有删除plate在其它所有位置
- **文件**: 
  - `Put/wrong.py` - 错误版本（只删除了Holding，没有删除其他位置）
  - `Put/correct.py` - 正确版本（应该删除所有On(plate,*)的位置，除了目标位置）

## 文件格式

每个文件包含三个集合：
- `pre`: 前置条件集合
- `add`: 添加的效果集合
- `del_set`: 删除的效果集合

## 使用方法

这些示例可以用于：
1. 对比学习错误和正确的动作定义
2. 训练模型识别和修正动作定义错误
3. 验证动作修正算法的正确性

