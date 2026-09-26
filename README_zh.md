<div align="center">

# CABTO

### Context-Aware Behavior Tree Grounding for Robot Manipulation

### 面向机器人操作的情境感知行为树落地

**AAAI 2026** &nbsp;·&nbsp; [📄 论文 (PDF)](https://arxiv.org/abs/2603.16809) &nbsp;·&nbsp; [🌐 项目主页](https://dids-ei.github.io/Project/CABTO/)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.10-000000?logo=mujoco&logoColor=white)
![MLX--VLM](https://img.shields.io/badge/VLM-MLX%20(Apple%20Silicon)-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/experiments-reproducible-success)

*从自然语言任务描述，到可落地、可自纠错的机器人行为——
LLM 符号规划 · 代码生成原语 · 闭环效果验证。*

</div>

---

<p align="center">
  <a href="README.md">English</a> &nbsp;·&nbsp; <b>中文</b>
</p>

---

## 📋 目录

- [概览](#概览)
- [核心亮点](#核心亮点)
- [仓库结构](#仓库结构)
- [五个任务](#五个任务)
- [快速开始](#快速开始)
- [运行实验](#运行实验)
- [引用](#引用)
- [许可证](#许可证)

---

## 📌 概览

**CABTO** 是一个面向**长时程机器人操作落地（grounding）**的框架，它将
**基于 LLM 的符号推理**、**形式化行为树（Behavior Tree）规划**与
**闭环、自纠错的执行引擎**紧密结合在一起。

给定一个以自然语言（或 BDDL）描述的任务，CABTO 会依次：

1. 使用 **LLM** 生成*情境感知*的符号计划 / 行为库，其中每个技能都建模为 `⟨pre, add, del⟩`（前置条件、增加效应、删除效应）。
2. 通过 **HOBTEA / OBTEA** 规划算法合成一棵**成本最小、可执行的行为树**。
3. 通过*代码生成的可组合原语*，将每个抽象动作**落地**为底层电机指令。
4. 每一步都与真值对照**验证**（可选 VLM 交叉验证），一旦失败就把**违规信息反馈给规划器并重新规划**——自纠错循环以 `max_rounds = 3` 为上限。

<div align="center">
<a href="images/framework.pdf"><img src="images/framework.png" width="760"/></a>
</div>

---

## ✨ 核心亮点

| | |
|:-|:-|
| 🧠 **LLM 符号规划** | 情境感知的行为库，每个技能建模为 `⟨pre, add, del⟩` |
| 🌳 **形式化 BT 规划** | 通过 **HOBTEA / OBTEA** 生成成本最小、可执行的行为树 |
| 🛠 **代码生成原语** | 用代码生成的可组合底层原语替代手写 FSM |
| 🔁 **自纠错闭环** | 逐步真值验证 + 重新规划，`max_rounds = 3` |
| 🖥 **全设备端运行** | MuJoCo 3.10 物理引擎 + MLX-VLM 视觉落地，Apple Silicon 本地运行，无需云端 / Isaac Sim |

---

## 🗂 仓库结构

> 本仓库在 `main` 分支上维护一个**自包含的五任务最小复现包**，
> 而**原始完整 CABTO 代码库**保留在 [`last`](https://github.com/DIDS-EI/CABTO/tree/last) 分支。

| 分支 | 内容 |
|:------:|---------|
| [`main`](https://github.com/DIDS-EI/CABTO) | 五任务最小复现包——活动任务代码、正式 BT 证据、Panda 资产、Oracle/Qwen-VL 打点与闭环结果 |
| [`last`](https://github.com/DIDS-EI/CABTO/tree/last) | 原始完整代码库——`btgym/`、`exps_bt_learning/`、`exp2_low_level_codegen/`、`exp4_bt_tasks/`、`reproduce/`、`docs/`、`images/`（参见[原始 README](https://github.com/DIDS-EI/CABTO/blob/last/README.md)） |

最小复现包的详细说明见 [`MINIMAL_REPRO.md`](MINIMAL_REPRO.md)；可视化报告入口见 [`index.html`](index.html)。

### 目录一览

```
CABTO/
├── project/                      # ── 五任务源码、当前配置与程序缓存
│   ├── core/                     #   任务 API / 桥接 / 落地 / 代码生成契约
│   ├── scenes/                   #   五个任务的当前场景 JSON
│   ├── data/programs/            #   缓存的模型生成程序
│   ├── assets/                   #   任务贴图（如茶盒标签）
│   ├── mujoco_workspace/         #   关节式 Franka MuJoCo 资产
│   ├── probe_*.py  run_*.py      #   任务探针与运行器（5 个任务）
│   ├── test_*.py   verify_*.py   #   各任务的测试与验证脚本
│   └── test_fixtures/  tests/    #   回归固件与测试资产
│
├── CABTO/exp4_bt_tasks/          # ── 实验 4 运行时（最小版）
│   ├── stage2_scripted/          #   手写脚本专家（双臂引擎基础）
│   ├── stage3_cabto/             #   单臂完整 CABTO 闭环（规划/代码生成/校验/循环）
│   ├── scenes/  assets/          #   Panda 机体定义与网格资产
│
├── BTExpansion-demo/src/         # ── 形式化 BT 扩展内核
│   ├── behavior_lib/             #   序列 / 选择 / 取反节点库
│   ├── behavior_tree/            #   行为树数据结构 + PTML
│   ├── bt_expansion/             #   BT 扩展引擎
│   ├── pddl_adapter/             #   PDDL ↔ BT 适配器
│   └── utils/
│
├── reports/                      # ── 精简报告页（已上传 GitHub）
│   ├── remaining_cabto/          #   五场景 CABTO 统一报告 + 相机打点（主入口）
│   ├── cover_cabto_fix/          #   Cover 生成与视觉放置修复报告
│   └── cover_cabto_method/       #   早期 Cover 方法实验（历史）
│
├── tasks/                        # ── 各任务证据（cover / blocks / pour / handover / storage）
│   ├── <task>/direct/            #   直接执行：rollout.mp4、result.json、场景、阶段图
│   ├── <task>/generated_bt/      #   生成程序、正式 BT（JSON/DOT/SVG）、重放
│   └── <task>/config.json · metadata.json · report.html
│
├── paper_reproduction/           # ── 论文实验证据
│   ├── perception_pointing/      #   实验 2：Oracle/Qwen-VL 打点代码、评估 JSON、叠加图、MP4、HTML
│   └── stage3_closed_loop/       #   Stage 3 oracle/qwen 闭环结果 + 逐步图片
│
├── tests/results/                #   164 项单元测试日志 + 五任务生成 BT 烟测摘要
│
├── outputs/                      # ── 完整原始实验产出（仅本地，已被 git 忽略）
│   ├── remaining_cabto/          #   五任务 CABTO 迁移实验完整运行（模型提案 / 采样 / 修订）
│   ├── cover_cabto_fix/          #   Cover 修复完整运行
│   ├── cover_cabto_method/       #   Cover 方法完整运行
│   ├── unified_cameras/          #   统一相机快照
│   └── chinese_paper/            #   中文论文源（ctexart）
│
├── index.html                    #  可视化门面（聚合 reports/ + tasks/ + paper_reproduction/）
├── run_task.py                   #  五任务直接执行入口
├── run_generated_bt.py           #  重放生成程序 + 正式 BT
├── run_tests.py                  #  单元测试运行器
├── verify_package.py             #  包完整性校验
├── requirements.txt              #  Python 依赖
└── package_manifest.json · PROVENANCE.json   #  清单（SHA256）与溯源
```

### 实验产出：`reports/` 与 `outputs/`

- **`reports/`**（已上传 GitHub）：从 `outputs/` 中抽取的精简、自包含报告页——只保留报告 HTML 及其直接引用的资源（mp4 / png / svg / json），用于支撑线上的 [`index.html`](index.html) 门面。
- **`outputs/`**（仅本地，已被 git 忽略）：完整原始实验产出（约 1.3 GB），包含每次模型提案、策略采样尝试、效果修订以及每次运行的碰撞审计。为完整复现而保留在本地，不纳入版本控制（含超过 100 MB 的文件和大量中间产物）。

| 产出目录 | 记录内容 |
|------------------|-----------------|
| `outputs/remaining_cabto/` | 五任务 CABTO 迁移实验完整运行——各任务的 `proposal/` → `sampling/` → `refinement/` → `final_seed0/1/`，以及 `summary.json` 和 `reproduce.txt` |
| `outputs/cover_cabto_fix/` | Cover 生成与视觉放置修复运行 |
| `outputs/cover_cabto_method/` | 早期 Cover 方法实验（历史） |
| `outputs/unified_cameras/` | 统一相机快照 |
| `outputs/chinese_paper/` | 中文论文源（ctexart + xelatex + biber） |

---

## 🎯 五个任务

`main` 分支端到端复现五个长时程操作任务：

| 任务 | 场景 | 指令 | 证据 |
|:----:|:------|:------------|:---------|
| 🍤 **Cover** | `cover_kitchen_sort_v2` | 三物体配对放置——虾入碗、苹果上砧板、土豆入锅 | `tasks/cover/` |
| 🧱 **Blocks** | `blocks_rebuilt_v1` | 积木堆叠并回位 | `tasks/blocks/` |
| 🥣 **Pour** | `held_basin_pour_smooth_v2` | 双臂：端盆接球并归位 | `tasks/pour/` |
| 🤝 **Handover** | `handover_tea_box_v3` | 无把手三盒空中交接 | `tasks/handover/` |
| 📦 **Storage** | `packing_dual_lift_shelf_storage_v3` | 装箱、双臂抬箱上架 | `tasks/storage/` |

---

## ⚡ 快速开始

**环境**——Python 3.11.9 · MuJoCo 3.10.0 · NumPy 2.4.6 · SciPy 1.17.1 · Pillow 12.2.0 · imageio 2.37.3

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> 若需在 Apple Silicon 上重跑本地 Qwen 视觉落地实验，请额外安装 `mlx-vlm`。
> 模型权重**不包含**在本包内。

---

## ▶️ 运行实验

**1. 直接执行五个任务**

```bash
PY=.venv/bin/python
$PY run_task.py cover    --output runs/direct_cover
$PY run_task.py blocks   --output runs/direct_blocks
$PY run_task.py pour     --output runs/direct_pour
$PY run_task.py handover --output runs/direct_handover
$PY run_task.py storage  --output runs/direct_storage
```

**2. 重放生成程序与正式行为树**

```bash
$PY run_generated_bt.py cover    --output runs/generated_cover
$PY run_generated_bt.py blocks   --output runs/generated_blocks
$PY run_generated_bt.py pour     --output runs/generated_pour
$PY run_generated_bt.py handover --output runs/generated_handover
$PY run_generated_bt.py storage  --output runs/generated_storage
```

**3. 运行测试套件**

```bash
.venv/bin/python run_tests.py     # 164 项测试全部通过
```

> 逐任务细节、证据路径与实验边界见 [`MINIMAL_REPRO.md`](MINIMAL_REPRO.md)。

---

## 🔖 引用

```bibtex
@inproceedings{cabto2026,
  title     = {Context-Aware Behavior Tree Grounding for Robot Manipulation},
  author    = {Cai, Yishuai and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

---

## 📄 许可证

基于 **MIT 许可证**发布。原始完整 CABTO 代码库位于
[`last`](https://github.com/DIDS-EI/CABTO/tree/last) 分支。

---

<div align="center">

*情境感知符号规划 · 形式化行为树 · 可落地、自纠错的执行。*

</div>
