# CABTO 项目总览 README（docs 版）

> 本文件整理 **CABTO 仓库的代码结构**，并汇总 **docs 下 v15 的整体下一步需求**。
> 配套阅读：`CABTO_ONBOARDING.md`（师弟师妹一页纸）、`CABTO_README.md`（4.5 周执行 checklist）、4 份 `0X_*_v15.html`（调研 / 扩展方案 / 论文构思 / 答辩提纲）。
>
> 维护人：蔡以帅 ｜ 分支：`feature/v15_highlight` ｜ 目标会议：AAAI 2026 / 博士论文 chap05

---

## 0. 一句话定位

CABTO 有 **两层语义**，读代码时务必先分清：

| 层 | 名称 | 含义 | 对应代码 |
|---|---|---|---|
| **底座层** | *Context-Aware Behavior Tree grOunding* | LLM 生成上下文相关行为库 → HOBTEA/OBTEA 规划出最小代价、可执行的行为树 → Isaac Sim / 真机执行 | `btgym/`、`dexrl/`、`exps_bt_learning/`、`sim_extension/` |
| **v15 高亮层** | *结构化上行反馈接口 $\mathcal{I}_\uparrow$* | 在底座之上，研究"具身大模型需要什么样的反馈接口才能跨 episode **单调改进**"，提出 **4-原子结构化反馈**（pose/contact/dynamics/semantic） | `baselines/`、`eval_v15/`、`tests/` |

**科学问题**：现有 SOTA（Inner Monologue / DoReMi / REFLECT）的反馈是自由文本或 scalar，反馈带宽 $H(F\mid\mathcal{I}_\downarrow)$ 不足，跨 episode 学习斜率 $k\approx 0$。CABTO 用 4-原子结构化反馈，由定理 1 保证 $k\ge\alpha\epsilon>0$（单调改进）。

---

## 1. 顶层目录速览

```
CABTO/
├── btgym/                 # 【底座】BT 规划 + 执行框架核心
├── dexrl/                 # 【底座】仿真环境 & Isaac Sim 扩展（Franka/双臂/IK）
├── exps_bt_learning/      # 【底座】实验脚本 + task1-7 任务定义 + LLM 生成行为库
├── sim_extension/         # 【底座】Isaac Sim Omni 扩展（UI + 场景管理）
├── assets/                # USD/场景/物体资源
├── baselines/             # 【v15】SOTA baseline + 4-原子约束检测器  ★新增
│   ├── inner_monologue_agent.py   # Inner Monologue baseline（自由文本反馈）
│   └── atoms/                     # 4-原子物理约束 violation 检测器
├── eval_v15/              # 【v15】统一评测指标（SR / CR / slope_k + 学习曲线）★新增
│   └── metrics.py
├── tests/                 # 【v15】4-原子检测器单元测试  ★新增
│   └── test_atoms.py
├── docs/                  # 【v15】方案 / onboarding / 调研 / 论文构思（本文件所在）
├── global_config.ini      # 系统配置（路径 / 硬件 IP / API key）
├── global_config.py       # 配置加载器
└── launch_sim.sh          # Isaac Sim 启动脚本
```

---

## 2. 底座层代码结构（已有，直接复用 ★★★★★）

### 2.1 `btgym/` — BT 规划与执行核心

```
btgym/
├── algos/
│   ├── bt_planning/          # 规划算法
│   │   ├── HOBTEA.py         # 分层最优 BT 扩展（论文主算法）
│   │   ├── OBTEA.py          # 最优 BT 扩展（非分层）
│   │   ├── BTPAlgo.py        # 所有规划算法的抽象基类
│   │   ├── behaviour_tree.py # 规划期 BT 中间表示
│   │   ├── Action.py         # 规划级 action 表示
│   │   └── main_interface.py # BTExpInterface：统一规划器入口
│   └── llm_client/           # LLM API 客户端（GPT-4 / GPT-3.5 / ERNIE）
├── behavior_tree/
│   ├── base_nodes/           # BT 节点基类（Action/Condition/Sequence/Selector）
│   ├── behavior_libs/        # ExecBehaviorLibrary：动态加载 LLM 生成的行为库
│   ├── behavior_trees/       # 基于 py_trees 的 BT 执行器
│   └── btml/                 # BTML 语言编译器（ANTLR4 语法）
├── core/                     # OmniGibson 仿真接口 & Tiago 控制器
└── assets/                   # 900+ BDDL 活动定义、场景清单
```

可用规划算法：`hobtea`（主）、`obtea`、`bfs`、`weak`（反应式）、`hbtp`。

### 2.2 `exps_bt_learning/` — 实验入口与任务

| 文件 | 作用 |
|---|---|
| `a_exp1_llm_bt/main_llm_bt.py` | 实验 1 主入口：LLM-BT 流水线 SR 评估 |
| `llm_generate_lib_func.py` | LLM → 行为库生成 |
| `validate_bt_fun.py` | BT 规划 + 执行验证 |
| `tools.py` | BDDL 解析 / prompt 构造 / 代码抽取 |
| `prompt_generate_libs.txt` | LLM 行为库生成 prompt 模板 |
| `a_exp1_llm_bt_tasks/task1-7` | 任务定义（含 `_base/` 的 OGAction/OGCondition 基类） |
| `a_exp3_cross_feedback_results/`、`3_main_cal_SR_cross_feedback.py` | **跨反馈实验雏形（v15 的起点，非从零开始）** |

已有任务集：

| ID | 名称 | 目标 |
|----|------|------|
| 1 | PlaceApple | `On(apple, coffeetable)` |
| 2 | ActivateLights | `ToggledOn(light1) & ToggledOff(light2)` |
| 3 | PutInDrawer | `In(pen, cabinet)` |
| 4 | HomeRearrangement | `On(pen,coffeetable) & IsClose(cabinet) & On(apple,coffeetable)` |
| 5 | MealPreparation | `IsClose(oven) & ToggledOn(oven) & On(apple,coffeetable) & In(chickenleg,oven)` |

### 2.3 核心概念

- **行为库（Behavior Library）**：LLM 针对具体任务上下文生成的 Python 类，落盘到 `exec_lib/Action/`、`exec_lib/Condition/`，由 `ExecBehaviorLibrary` 动态加载。Action 继承 `OGAction`（定义 `pre`/`add`/`del_set`），Condition 继承 `OGCondition`。
- **BTML**：行为树序列化语言，ANTLR4 语法编译。

---

## 3. v15 高亮层代码结构（新增 ★，本轮工作核心）

### 3.1 `baselines/atoms/` — 4-原子物理约束检测器

把论文 §5.2.2 的 4 个约束 $C_*$ 实现成统一接口、零依赖、可单测的检测器。**师弟 B 只需把每个文件里的 `_fake_*` 占位实现替换为真实仿真调用**，上层接口与定理 1 所需的反馈结构不变。

| 文件 | 约束 | 检测内容 | 三态判据 |
|---|---|---|---|
| `pose.py` → `check_reach` | `C_reach` | 逆运动学可达性（残差距离） | `<0.02m` PASS / `<0.08m` MARGINAL / 否则 FAIL |
| `contact.py` → `check_collide` | `C_collide` | 轨迹碰撞（点-AABB 有符号距离 + 语义标签） | `>0.05m` PASS / `≥0` MARGINAL / 穿透 FAIL |
| `dynamics.py` → `check_torque` | `C_torque` | 关节扭矩 / 力学包络超限比例 | `<0.85` PASS / `≤1.0` MARGINAL / `>1.0` FAIL |
| `semantic.py` → `check_skill` | `C_skill` | 技能历史成功率（Beta-Binomial 后验 + critic） | `ρ≥0.70` PASS / `≥0.35` MARGINAL / 否则 FAIL |
| `common.py` | — | `AtomLevel` / `AtomReport` / `PriorUpdateHint` / `compose_feedback` | — |

**统一数据契约**（`common.py`）：

```python
AtomReport(name, level: AtomLevel, score: float∈[0,1], detail: dict)
# compose_feedback 把 4 个 AtomReport 拼成论文公式 (5.2) 的 5 字段 JSON：
{
  "C_reach":   {...}, "C_collide": {...},
  "C_torque":  {...}, "C_skill":   {...},
  "prior_update_hint": {failed_atom, failure_mode, suggested_repair, payload} | None
}
```

> 关键设计：`prior_update_hint` 必须是**结构化** JSON 而非自然语言，否则定理 1 的带宽下界条件 (5.4) 不成立（onboarding 坑 #1）。

### 3.2 `baselines/inner_monologue_agent.py` — baseline #1

复现 Inner Monologue（Huang 2022 CoRL）的最小可跑骨架。Agent 接口（与 `eval_v15` 对齐）：

```python
agent.reset(task: dict)      # episode 开始
agent.act(obs) -> {"skill", "args", "rationale"}
agent.update(feedback)       # 只能消费自由文本；传入结构化 dict 会被强行降级为 str（与 CABTO 的本质差异）
agent.dump_trace() -> List[dict]
```

直接 `python baselines/inner_monologue_agent.py` 用 stub LLM 跑通烟囱测试。

### 3.3 `eval_v15/metrics.py` — 统一评测指标

三个核心指标（对齐 §5.4 理论分析）：

| 指标 | 含义 |
|---|---|
| **SR** (Success Rate) | episode 成功率，跨 task/seed 平均 |
| **CR** (Constraint Recovery) | 4-原子约束被触发并成功修正的比例（仅 CABTO 类方法，baseline 为 `nan`） |
| **slope_k** | 跨 episode 学习曲线的线性回归斜率（**核心卖点**：CABTO `>0`，baseline `≈0`） |

数据契约 `EpisodeResult(agent, task_id, episode_idx, success, steps, trace, constraint_events)`。
产出：`plot_learning_curve()` 出 PNG，`write_html_report()` 出 HTML 报告。

```bash
python -m eval_v15.metrics --runs "runs/*.json" --out report.html   # 真实数据
python eval_v15/metrics.py                                          # 无参跑假数据自检
```

### 3.4 `tests/test_atoms.py` — 4-原子单元测试

师弟 B 的验收脚本，覆盖每个原子的 PASS/MARGINAL/FAIL 三态 + `compose_feedback` 行为 + 容错。

```bash
python tests/test_atoms.py   # 期望末行：ALL ATOMS OK
```

---

## 4. 快速上手（一周路径）

```bash
# Day 1 环境
git checkout -b feature/v15_${YOUR_NAME}
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install numpy matplotlib openai

# Day 2 跑通烟囱测试
python tests/test_atoms.py                  # 4-原子检测器 → ALL ATOMS OK
python baselines/inner_monologue_agent.py   # baseline mini episode
python eval_v15/metrics.py                  # 假数据学习曲线 + HTML 报告

# 底座（需 Ubuntu + NVIDIA GPU + Isaac Sim 4.2）
cp global_config.ini.example global_config.ini   # 配 isaacsim_path / data_path / api_key
./launch_sim.sh                                   # 标准启动；-e 带扩展 GUI
cd exps_bt_learning/a_exp1_llm_bt && python main_llm_bt.py
```

> 运行环境：Ubuntu 20.04/22.04 + NVIDIA GPU(CUDA) + Isaac Sim 4.2 + OmniGibson；Python 3.10+。
> v15 上层（`baselines`/`eval_v15`/`tests`）零仿真依赖，在 macOS 上即可跑通自检。

---

## 5. 下一步需求（docs 整体规划）

v15 的目标：**4.5 周把 CABTO 从"普通章节"升级为博士论文 highlight chapter（chap05，40-50 页）**。
详见 `02_CABTO扩展方案_v15.html` + `CABTO_README.md`。三条 P0 实验路线：

| # | 实验 | 周数 | 杀手锏产出 | 期望结果 |
|---|------|------|-----------|---------|
| ① | **vs Inner Monologue / DoReMi / REFLECT** | 2 | 跨 episode 学习曲线（仅 CABTO 上升） | CABTO slope ≥ 0.02，baseline ≈ 0；SR ~75% vs ~55-60% |
| ② | **4-原子反馈源消融**（8 组合 A0-A7） | 1 | 瀑布图 + 雷达图 | A4（skill 可行域）单项贡献 ≥ 10%，full(A7) +22% |
| ③ | **跨 5 VLM 通用性** | 1.5 | 柱状图 + SR std 横线 | ≥3 VLM，SR std < 8% |

### 5.1 实验① — 三大 baseline（最关键）

- `inner_monologue.py`（已有骨架 ✅）：每 episode 把"成功/失败 + 自然语言场景描述"追加进下次 prompt。
- `doremi.py`（待做）：用 LLM 当 checker 检测 BDDL 谓词违反 → 触发 OBTEA 重规划，不更新 prior。
- `reflect.py`（待做）：三段失败摘要（视觉/力/状态）→ LLM 解释 → 修正 plan。
- `cabto_full.py`（待做）：把现有 CABTO 包装成 agent。
- 主入口 `run_exp1_main_comparison.py` + 出图 `plot_learning_curve.py`。
- 任务集：CABTO task1-5 + LIBERO-LONG(10) + RoboCasa Kitchen(5)。

### 5.2 实验② — 反馈源消融

把 4 原子做成开关 `FeedbackConfig(enable_reach/collide/torque/skill)`，跑 8 组合 `0000…1111` × task1-5 × 3 seeds，出瀑布图（验证各原子边际贡献）+ 雷达图。
> 4-原子检测器骨架（`baselines/atoms/`）已就绪，消融的"开关化"接口可直接基于 `ALL_CHECKERS` 搭建。

### 5.3 实验③ — 跨 VLM 通用性

统一抽象类 `BaseLLMClient.generate_behavior_lib(prompt) -> {文件路径: 代码}`，接 5 个 client：Qwen2.5-VL-72B / 混元 hunyuan-t1 / GPT-4o / Gemini-1.5-Pro / Claude-3.5-Sonnet。
v15 标准化 prompt：反馈块用 **JSON 结构化**（含 `prior_update_hint`），非自由文本。
关键指标：SR std across VLM、反馈吸收率、反馈块 token 成本（<1000）。

### 5.4 理论与论文落点（chap05）

- §5.4 Theoretical Analysis：**定理 1（单调收敛 $k\ge\alpha\epsilon$）** + **定理 2（反馈带宽下界）**。
- §5.5.5：t-SNE 三联图（反馈前 / 后 / KL 热图）。
- §5.5.6：≥3 个失败 case 截图 + 文字分析。
- LaTeX：`latex/data/chap05.tex`（+ `chap05_method.tex` / `chap05_exp.tex`），图存 `latex/figures/v15_fig5_*.pdf`。

### 5.5 验收 checklist（硬性标准）

- [ ] 实验① learning curve：CABTO slope ≥ 0.02 且显著高于 baseline
- [ ] 实验② 瀑布图：A4（skill 可行域）贡献 ≥ 10%；雷达图 4 维全展开
- [ ] 实验③ 跨 VLM 柱状图：≥3 VLM，SR std < 8%；反馈 prompt 模板归档
- [ ] chap05 总页数 ≥ 35 页；§5.4 ≥1 定理 + 半页证明 sketch
- [ ] §5.5.5 t-SNE 三联图；§5.5.6 ≥3 失败 case

### 5.6 风险与 fallback

| 风险 | fallback |
|------|---------|
| LIBERO-LONG 适配超 1 周 | 砍 LIBERO，仅用 task1-7 + RoboCasa 3 任务 |
| Gemini / Claude API 不通 | 砍到 3 VLM（Qwen + 混元 + GPT-4o） |
| baseline 复现 SR 异常 | 用原论文报告值 + 注脚说明 |
| 定理 1 严格证明推不出 | 改为 empirical observation + 30 次实验数据 |

---

## 6. docs 文件索引

| 文件 | 作用 |
|---|---|
| `README.md` / `README.html` | 本总览（代码结构 + 下一步需求） |
| `CABTO_ONBOARDING.md` | 师弟师妹一页纸：科学问题 / 分工 / 一周路径 / 常见坑 |
| `CABTO_README.md` | 4.5 周执行 checklist + 文件级实现清单 + 时间表 |
| `01_调研报告_v15.html` | 文献调研报告 |
| `02_CABTO扩展方案_v15.html` | **最详细方案**：§1 现状盘点 → §10 验收 checklist |
| `03_博士论文构思书_v15.html` | 博士论文构思书 |
| `04_答辩PPT提纲_v15.html` | 答辩 PPT 提纲 |

---

## 7. 红线（不要做的事）

1. 不要把 4 原子约束写成自然语言再喂 LLM（退化成 REFLECT，定理 1 不成立）——必须 JSON。
2. `episode_idx` 必须按**持续学习顺序**编号，不是 task_id，否则学习斜率是噪声。
3. VLM caption 一定要 cache 落盘（省钱）。
4. 跑 50 任务设 5 个 seed（0/1/2/3/4）。
5. 不要直接改 `main` 分支 / 改 v15 LaTeX / 删 `baselines/` 任何东西。

---

🚀 *CABTO — Context-Aware Behavior Tree Grounding + 结构化上行反馈接口 ｜ 蔡以帅 2026-06-13*
