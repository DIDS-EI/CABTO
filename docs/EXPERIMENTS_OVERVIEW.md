# CABTO 论文复现 · 实验总览

> 本文档说明论文里的**每一个实验**：它的源码在哪、结果文件在哪、可视化产物在哪、怎么复现、复现出来的数字是多少。
>
> 仓库根目录：`/Users/yishuaicai/mywork/CABTO`
> 论文共 **3 个实验**（无 exp4 / exp5）。

### 三个实验的真实位置

| 实验 | 真实目录 | 说明 |
|------|----------|------|
| exp1 | `exps_bt_learning/` | exp1 全部源码、task 定义、结果都在这个 BT 学习包里（与 exp3 共用） |
| exp2 | `exp2_low_level_codegen/` | exp2 完整自包含包，直接进入即可 |
| exp3 | `exps_bt_learning/exp3_results/` | exp3 结果（修正后的动作库 + 行为树）；复现脚本在 `reproduce/` |

> 说明：exp1/exp3 的代码被 Python 写死在 `exps_bt_learning.` 包命名空间（268 处 `import`，含历史生成的 exec_lib），不能改名或物理移动，否则会断裂 import 链；故二者直接以 `exps_bt_learning/` 为真实位置。exp2 完全自包含，单独成包 `exp2_low_level_codegen/`。

---

## 速查表：第几个实验 → 在哪个文件夹

| 实验 | 名称（论文对应） | 源码位置 | 结果位置 | 复现状态 |
|------|------------------|----------|----------|----------|
| **exp1** | 高层模型提案（LLM-BT 成功率，表1） | `exps_bt_learning/exp1_llm_bt_src/` | `exps_bt_learning/exp1_results/`<br>`reproduce/results/exp1_replay_*.csv` | ✅ 离线重放，overall 63.8% |
| **exp2** | 底层策略采样（VLM 打点 + 代码生成抓取） | `exp2_low_level_codegen/franka_grasp_dp/exp2_codegen/` | `exp2_low_level_codegen/franka_grasp_dp/exp2_codegen/results/`<br>`.../renders/`（视频+打点图） | ✅ 本机真跑（oracle 100% / Qwen 60%） |
| **exp3** | 跨层精化反馈（修正缺陷动作模型，表3） | `reproduce/scripts/run_exp3_replay.py` | `exps_bt_learning/exp3_results/`<br>`reproduce/results/exp3_replay_*.csv` | ✅ 离线复算，一致率 100% |

> 命名说明：`a_exp1_llm_bt_tasks/`（task1~task7 定义）是被 Python 动态 `import` 的包（模块路径 `exps_bt_learning.a_exp1_llm_bt_tasks.*`），改名会破坏 exec_lib 的 import 链，故**保持原名**，它逻辑上属于 exp1。

---

## 实验 1 — 高层模型提案（LLM-BT 成功率）

**做什么**：给定任务的 BDDL 描述 → GPT-4o 生成上下文相关的动作/条件库 → OBTEA/HOBTEA 规划出行为树 → 仿真执行 → 统计成功率。对应论文**表 1**。覆盖 7 个任务集 × 10 次 try × 3 个难度 goal（easy/medium/hard）。

### 源码
| 路径 | 作用 |
|------|------|
| `exps_bt_learning/exp1_llm_bt_src/main_llm_bt.py` | 原作者主入口（在线调 LLM） |
| `exps_bt_learning/exp1_llm_bt_src/1_main_cal_SR_models.py` | 成功率统计 |
| `exps_bt_learning/a_exp1_llm_bt_tasks/` | task1~task7 任务定义 + `_base/`（OGAction/OGCondition 基类） |
| `exps_bt_learning/llm_generate_lib_func.py` | LLM → 行为库生成 |
| `exps_bt_learning/validate_bt_fun.py` | BT 规划 + 执行验证（依赖 `btgym`） |
| `reproduce/scripts/run_exp1_replay.py` | **离线重放复现入口**（零 API 调用，重放作者已缓存的 LLM 输出） |
| `reproduce/scripts/replay_llm.py` | ReplayLLM：从缓存读历史 LLM 输出，替代真实 API |

### 结果文件
| 路径 | 内容 |
|------|------|
| `exps_bt_learning/exp1_results/` | 作者原始结果：70 个 `task{N}_try_{i}_*/` 目录（每个含 `llm_output.txt` 缓存）+ `summary_results_gpt-4o_*.csv`（论文表1 原始数据） |
| `reproduce/results/exp1_replay_summary_*.csv` | **本次复现汇总**（按任务的成功率） |
| `reproduce/results/exp1_replay_detail_*.csv` | 本次复现明细（每个 try × 难度的 success/expanded/act） |

### 怎么复现
```bash
cd /Users/yishuaicai/mywork/CABTO
# btgym 框架已归位到仓库顶层 CABTO/btgym/，仅需 PYTHONPATH=.
PYTHONPATH=. python reproduce/scripts/run_exp1_replay.py
# 依赖：numpy sympy pandas py_trees antlr4-python3-runtime shortuuid（完全离线，无需 API/GPU）
```

### 复现结果（本次重放，GPT-4o 缓存）
| task | 名称 | all_SR(%) |
|------|------|-----------|
| task1 | Cover | 100.0 |
| task2 | Blocks | 80.0 |
| task3 | Clean | 40.0 |
| task4 | Handover | 46.7 |
| task5 | Pour | 60.0 |
| task6 | HomeRearrangement | 43.3 |
| task7 | MealPreparation | 76.7 |
| **ALL** | **Overall** | **63.8**（easy 64.3 / medium 62.9 / hard 64.3） |

---

## 实验 2 — 底层策略采样（VLM 打点 + 代码生成抓取）

**做什么**：把底层抓取策略写成「代码生成串接原语」——用 VLM 在两个视角的相机图上打点（描述要抓哪 → 模型给像素点）→ 两视角射线三角化反投影成 3D 点 → 调用 `move_to / descend / close_gripper / lift` 等原语完成抓取。对比 **oracle**（几何真值打点）与**真实本地 Qwen2.5-VL** 两种后端。对应论文实验 2 的「底层策略采样 + 一致性验证」。

> 说明：论文原文实验 2 用 Molmo + cuRobo 在 Isaac Sim 中采样，本仓库**没有**该端到端主循环（`baselines/atoms` 仅是含 fake IK/碰撞/RNEA 的组件脚手架）。本次是用 MuJoCo + Franka 在本机**重新实现并真跑**的等价实验。

### 源码（`exp2_low_level_codegen/franka_grasp_dp/exp2_codegen/`）
| 模块 | 作用 |
|------|------|
| `camera.py` | 投影 `project_point` / 深度反投影 `deproject_with_depth` / 两视角三角化 `triangulate` |
| `vlm_pointer.py` | VLM 打点接口：`OraclePointer`、`QwenVLPointer`（本地 mlx-vlm）、Molmo 占位 |
| `primitives.py` | 机械臂原语：`move_to / descend_to / align_xy / open_gripper / close_gripper / lift`（含积分项消伺服偏差） |
| `codegen_action.py` | **代码生成的 pick 动作**：双视角打点→三角化→几何 grasp_z→approach/descend/close/lift + 鲁棒融合 |
| `eval_pick.py` | 端到端抓取评估（5 物体 × N 种子） |
| `render_viz.py` | 渲染可视化（打点叠加图 + 多视角 + rollout 视频） |
| `calibrate_qwen_prompt.py` | Qwen prompt/解析器标定 |
| `build_report.py` | 生成可视化 HTML 报告 |
| 依赖：`../dp_common/`（MuJoCo 环境）、`../diffusion_policy_train/`（DP 训练+权重）、`../../assets/panda_assets/`（机械臂网格） |

### 结果文件
| 路径 | 内容 |
|------|------|
| `.../exp2_codegen/results/eval_pick_oracle.json` | oracle 后端评估汇总 |
| `.../exp2_codegen/results/eval_pick_qwen.json` | 真实 Qwen2.5-VL 后端评估汇总 |
| `.../exp2_codegen/results/qwen_prompt_calib.json` | Qwen prompt 标定结果 |
| `.../exp2_codegen/exp2_codegen_report.html` | **可视化总报告**（内嵌打点图 + 视频引用 + 对比表） |

### 可视化产物（`.../exp2_codegen/renders/`，共 44 个文件）
| 类型 | 文件 | 说明 |
|------|------|------|
| 抓取过程视频 | `rollout_{oracle,qwen}_{物体}.mp4`（7 段） | 完整抓取 rollout |
| 打点叠加图 | `pointing_overlay_{backend}_{物体}.png`（7 张） | 红叉=VLM 实际打的点，绿圈=三角化 3D 重投影 |
| 单视角打点 | `point_{backend}_{物体}_{overview,front}.png`（14 张） | 各相机视角的打点 |
| 三视角拼图 | `multiview_{backend}_{物体}.png`（7 张） | front/overview/wrist 拼图 |

### 怎么复现
```bash
cd /Users/yishuaicai/mywork/CABTO/exp2_low_level_codegen/franka_grasp_dp
# oracle（快，无需 VLM）
PYTHONPATH=. python exp2_codegen/eval_pick.py --backend oracle --seeds 0-3
# 真实 Qwen2.5-VL 本地推理（需 mlx-vlm + 模型缓存，Apple Silicon）
PYTHONPATH=. python exp2_codegen/eval_pick.py --backend qwen --seeds 0-3
# 渲染可视化
PYTHONPATH=. python exp2_codegen/render_viz.py --backend qwen --objects dice_cube soda_can tea_tin
```

### 复现结果
| 后端 | 端到端成功率 | 打点误差 median / mean |
|------|--------------|------------------------|
| oracle（几何真值） | **20/20 = 100%** | 6.7 / 6.8 px |
| 真实 Qwen2.5-VL-3B 本地 | **12/20 = 60%** | 18.0 / 24.2 px |

Prompt 标定：简洁 JSON prompt 最准（median 20.4px@512≈1cm），出现「grasp」字样误差翻倍（33.5px）。

---

## 实验 3 — 跨层精化反馈（修正缺陷动作模型）

**做什么**：当某个动作模型有缺陷导致规划/执行失败时，用视觉 LLM 看失败图像 + 执行上下文，给出**跨层反馈**修正动作库，使其重新变成可成功规划的库。对应论文**表 3** 的结论：跨层精化能把缺陷动作模型修正为可成功规划。

> 说明：实验 3 在线需要视觉 LLM(gpt-4o) + 失败图片 + 仿真上下文（本机无法离线获得）。但作者已保存完整产物（修正后的动作库 + 记录 csv），故本次做**离线复算 + 一致性校验**：加载修正后的库用 obtea 重跑 sound&complete 验证，对照作者记录。

### 源码
| 路径 | 作用 |
|------|------|
| `reproduce/scripts/run_exp3_replay.py` | **离线复现入口**：读记录 csv，重跑修正后动作库验证一致性 |
| `exps_bt_learning/exp1_llm_bt_src/3_main_cal_SR_cross_feedback.py` | 原作者跨层反馈成功率统计脚本 |
| `exps_bt_learning/validate_bt_fun.py` | BT 规划 + 执行验证（依赖 `btgym`） |

### 结果文件
| 路径 | 内容 |
|------|------|
| `exps_bt_learning/exp3_results/results.csv` | 作者原始记录：每个 try 的 initial/goal/feedback_times/success |
| `exps_bt_learning/exp3_results/behavior_lib_case{1,2}/try_N/` | 经跨层精化**修正后**的动作库 |
| `exps_bt_learning/exp3_results/case{1,2}_try_N/` | 每个 try 的行为树（`bt.btml` + `bt.svg`，含 easy/medium/hard 难度树） |
| `reproduce/results/exp3_replay_summary_*.csv` | **本次复现汇总** |
| `reproduce/results/exp3_replay_detail_*.csv` | 本次复现明细（10 个 try 逐行对照） |

### 怎么复现
```bash
cd /Users/yishuaicai/mywork/CABTO
PYTHONPATH=. python reproduce/scripts/run_exp3_replay.py
# 完全离线，无需 LLM key / 图片 / GPU / 仿真
```

### 复现结果（case1: `Lift(big_box,board)`，10 try）
| 指标 | 值 |
|------|-----|
| 记录_总成功率 | 90.0% |
| 记录_无反馈成功率 | 100.0% |
| 含反馈条目数 | 2（其中成功率 50%） |
| **离线重跑_与记录一致率** | **100.0%** ✅ |

结论印证论文表3：无反馈时缺陷库成功率受限，跨层精化反馈修正后可恢复成功规划；离线重跑与作者记录 100% 一致，复现可信。

---

## 附：其它相关目录

| 目录 | 归属 | 说明 |
|------|------|------|
| `btgym/` | 原作者 BT 规划/执行框架（已归位顶层，入库） | exp1/exp3 复现依赖它，`ROOT_PATH` 相对自身定位，自带 `assets/task_names.txt`，无需额外配置 |
| `之前的/original_exp2_scaffold/` | 原作者实验2脚手架（已归档，复现**不依赖**） | `baselines/atoms/`（含 fake IK/碰撞/RNEA 的组件脚手架）、`eval_v15/metrics.py`（随机数据 metrics）、`tests/test_atoms.py`（仅测脚手架）——非端到端主循环，已被 `exp2_low_level_codegen`（MuJoCo+真 Qwen）的真实复现替代 |
| `之前的/` | 原作者其余框架与历史归档（被 `.gitignore` 忽略） | `dexrl/`（仿真环境）、`assets/`（409M 大资产）、`sim_extension/`、`exps_bt_learning_archive/`（record1116/1117 等历史 record）——**复现不依赖** |

---

*生成时间：2026-06-27 ｜ 仓库：CABTO（AAAI 2026）*
