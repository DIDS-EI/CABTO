# EDPS-CABTO 五任务最小复现包

本目录是从 `edps_cabto`、CABTO stage2/stage3 和 BTExpansion-demo 中筛出的自包含复现工程。源工程保持不动；这里只保留五个任务的当前推荐直接执行版本、最新可用生成程序与正式 BT 证据、测试、Panda 资产，以及 CABTO 论文实验中的 Oracle/Qwen-VL 打点与闭环结果。

## 目录

- `project/`：五任务源码、5份当前配置、程序缓存和测试。
- `CABTO/`：最小 stage2 机器人控制、stage3 正式 BT/代码生成/闭环运行时和 Panda 定义。
- `BTExpansion-demo/src/`：正式 BT Expansion 内核。
- `tasks/{cover,blocks,pour,handover,storage}/`
  - `direct/`：直接执行结果、MP4、场景与 JSON。
  - `generated_bt/`：模型生成程序、正式 BT JSON/DOT/SVG、重放视频与结果。
  - `config.json`、`metadata.json`、`report.html`（如原任务已有报告）。
- `paper_reproduction/perception_pointing/`：实验2 Oracle/Qwen 打点代码、评估 JSON、叠加图、MP4 和 HTML。
- `paper_reproduction/stage3_closed_loop/`：oracle/qwen 闭环结果与逐步图片。
- `tests/results/`：164项单元测试日志和五任务生成BT烟测摘要。

## 环境

验证环境：Python 3.11.9、MuJoCo 3.10.0、NumPy 2.4.6、SciPy 1.17.1、Pillow 12.2.0、imageio 2.37.3。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

如需重新运行真实 Qwen 打点，再安装 `mlx-vlm`。模型权重没有复制进最小包。

## 直接执行五任务

```bash
PY=.venv/bin/python
$PY run_task.py cover   --output runs/direct_cover
$PY run_task.py blocks  --output runs/direct_blocks
$PY run_task.py pour    --output runs/direct_pour
$PY run_task.py handover --output runs/direct_handover
$PY run_task.py storage --output runs/direct_storage
```

Cover、Blocks、Handover 可附加 `--no-render` 做快速检查。输出目录需为新目录，避免覆盖证据。

## 重放生成程序与正式 BT

```bash
$PY run_generated_bt.py cover   --output runs/generated_cover
$PY run_generated_bt.py blocks  --output runs/generated_blocks
$PY run_generated_bt.py pour    --output runs/generated_pour
$PY run_generated_bt.py handover --output runs/generated_handover
$PY run_generated_bt.py storage --output runs/generated_storage
```

五个入口已在本目录实际重放成功。Handover 的原子包装程序来自先前真实本地模型输出，并在无把手 v3 上重新执行每一步。Cover/Blocks 也保留此前模型生成源码原文，通过已披露的任务局部角色/原子适配器，在各自最新活动场景上重新构建正式BT并完成物理重放；适配边界见各任务 `metadata.json` 与重放 `result.json`。

## 测试

```bash
.venv/bin/python run_tests.py
```

当前结果：29 + 18 + 53 + 39 + 25 = **164项全部通过**；详见 `tests/results/`。

## Oracle 与 Qwen-VL 打点

保留并核验了三组视觉打点证据：Oracle原始明细2/2；历史 **Qwen2.5-VL-3B-Instruct-4bit** 为12/20（median 18.0 px）；本次在新工程中补跑的 **Qwen3.5-0.8B-4bit** 为2/20（median 158.5 px）。旧报告中的Oracle 20/20缺少对应的20条原始记录，已按可核验JSON修正。

0.8B完整明细、叠加图和成功样例视频见 `paper_reproduction/perception_pointing/qwen3_5_0_8b_report.html`。模型权重不随最小包复制；详见 `paper_reproduction/perception_pointing/MODEL_FACTS.md`。

## 实验边界

- 五任务并非都属于 strict 纯物理抓取：Pour、Handover、Storage及历史生成档案包含显式、已记录的 grasp-weld 辅助。
- Oracle 位姿、机器人重力补偿和手写低层控制不等于真机自主系统。
- 模型生成的是受限 API 程序；形式化 BT 与物理效果由独立状态检查验证。
- 历史 JSON 中的原绝对路径保留以维护溯源；活跃代码已改为当前包内相对路径。
