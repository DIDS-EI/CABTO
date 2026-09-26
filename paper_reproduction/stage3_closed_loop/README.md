# CABTO Stage 3 闭环证据

- `oracle_replay/`：在本复现包内重新运行 `test_loop_oracle.py oracle` 的成功结果和 MP4。
- `qwen_result.json`：原工程保留的真实本地 Qwen 闭环结果；该次结果失败，记录了VLM定位瓶颈。
- `step*_before/after.png`：原闭环逐步视觉证据。
- 活跃源码位于 `../../CABTO/exp4_bt_tasks/stage3_cabto/`。

运行：

```bash
cd ../../CABTO/exp4_bt_tasks/stage3_cabto
PYTHONPATH=. python test_loop_oracle.py oracle
# 安装 mlx-vlm 且本地有模型后，可将 oracle 改为 qwen。
```
