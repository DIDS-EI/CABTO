# CABTO v15 Highlight Extension · 动手实施 README

> 4.5 周把 CABTO 从"普通章节"升级为博士论文 highlight chapter（chap05, 40-50 页）
>
> 配套：`02_CABTO扩展方案_v15.html`（全详细方案）+ 本 README（执行 checklist）

## 0. 仓库准备

```bash
cd /Users/yishuaicai/mywork/CABTO
git status                                  # 确认干净
git checkout -b feature/v15_highlight       # 新建分支
mkdir -p exps_bt_learning/v15_highlight/{baselines,ablation,cross_vlm,tasks,metrics,results,figures}
```

## 1. 三大扩展实验路线图

| # | 实验 | 周数 | 优先级 | 杀手锏产出 |
|---|------|------|--------|-----------|
| ① | vs Inner Monologue / DoReMi / REFLECT | 2 | P0 必做 | 跨 episode 学习曲线（CABTO 唯一上升） |
| ② | 反馈源消融（4 原子组件 × 8 组合） | 1 | P0 必做 | 瀑布图 + 雷达图 |
| ③ | 跨 5 VLM 通用性 | 1.5 | P0 必做 | 柱状图 + SR std 横线 |

## 2. 实验① · 三大 baseline 实现细节

### 2.1 Inner Monologue (Huang 2022 CoRL)
- 文件：`exps_bt_learning/v15_highlight/baselines/inner_monologue.py`
- 核心：每 episode 结束把"成功/失败 + 自然语言场景描述"附加到下次 prompt
- 复用：`btgym/algos/llm_client/llm_gpt.py` 的 ChatCompletion 接口
- 不改：BT 规划走原 OBTEA
- 关键代码骨架：

```python
class InnerMonologueAgent:
    def __init__(self, llm_client, bt_planner):
        self.llm = llm_client
        self.planner = bt_planner
        self.history = []                              # 自然语言历史

    def step(self, task_desc, init_state):
        prompt = self._build_prompt(task_desc, init_state, self.history)
        lib = self.llm.generate_behavior_lib(prompt)
        bt = self.planner.plan(lib, init_state, task_desc.goal)
        success, trace = self.execute(bt)
        # 关键差异：仅追加自然语言文本，不结构化、不更新 prior
        nl_feedback = f"Episode result: {'success' if success else 'failed at ' + trace[-1]}"
        self.history.append(nl_feedback)
        return success
```

### 2.2 DoReMi (Guo 2024 ICRA)
- 文件：`baselines/doremi.py`
- 核心：每个 action 执行前后调 LLM 检查 BDDL 谓词违反
- 关键差异 vs CABTO：用 LLM 当 checker（对物理约束不敏感），不更新 prior，只触发重规划

### 2.3 REFLECT (Liu 2023 CoRL)
- 文件：`baselines/reflect.py`
- 核心：三段失败摘要（视觉 / 力 / 状态）→ LLM 解释 → 修正 plan
- 数据源：OmniGibson 的 RGB 图 + Tiago 力反馈 + state dict
- 复用：CABTO 现有 `dexrl/sim/` 接口

### 2.4 主实验入口
- 文件：`run_exp1_main_comparison.py`
- 命令：
```bash
python exps_bt_learning/v15_highlight/run_exp1_main_comparison.py \
    --agents cabto,inner_monologue,doremi,reflect \
    --tasks 1,2,3,4,5,libero1,libero2,libero3 \
    --episodes 20 \
    --seeds 3 \
    --llm gpt-4o \
    --output exps_bt_learning/v15_highlight/results/exp1/
```

### 2.5 出图
- 文件：`plot_learning_curve.py`
- 关键图：横轴 episode (1-20)，纵轴 SR，4 条曲线 + 误差棒
- 预期：CABTO slope ≥ 0.02，其余 ≈ 0

## 3. 实验② · 反馈源消融

### 3.1 4 原子组件的开关化
原 CABTO 的反馈接口在 `btgym/algos/bt_planning/HOBTEA.py` 的 `expand()` 函数里散落，需要重构为：

```python
@dataclass
class FeedbackConfig:
    enable_reach: bool = True
    enable_collide: bool = True
    enable_torque: bool = True
    enable_skill: bool = True

class CABTOWithFeedbackSwitches:
    def __init__(self, cfg: FeedbackConfig):
        self.cfg = cfg

    def collect_feedback(self, trace):
        fb = {}
        if self.cfg.enable_reach:   fb['reach']   = check_reach(trace)
        if self.cfg.enable_collide: fb['collide'] = check_collide(trace)
        if self.cfg.enable_torque:  fb['torque']  = check_torque(trace)
        if self.cfg.enable_skill:   fb['skill']   = compute_skill_success_rate(trace)
        return fb
```

### 3.2 8 组合
| ID | reach | collide | torque | skill | 预期 SR |
|----|-------|---------|--------|-------|---------|
| A0 | 0 | 0 | 0 | 0 | 50% (baseline=LLM-BT) |
| A1 | 1 | 0 | 0 | 0 | +5% |
| A2 | 0 | 1 | 0 | 0 | +3% |
| A3 | 0 | 0 | 1 | 0 | +2% |
| A4 | 0 | 0 | 0 | 1 | +12% (技能可行域最强) |
| A5 | 1 | 1 | 0 | 0 | +7% |
| A6 | 1 | 1 | 1 | 0 | +8% |
| A7 | 1 | 1 | 1 | 1 | +22% (full CABTO) |

```bash
python exps_bt_learning/v15_highlight/ablation/run_ablation.py \
    --config_grid 0000,1000,0100,0010,0001,1100,1110,1111 \
    --tasks 1,2,3,4,5 --seeds 3
```

## 4. 实验③ · 跨 VLM 通用性

### 4.1 统一接口
- 文件：`cross_vlm/unified_llm_interface.py`
- 5 个 client 实现同一抽象类：

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def generate_behavior_lib(self, prompt: str) -> dict[str, str]:
        """返回 {'Action/PickUp.py': '...', 'Condition/Holding.py': '...'}"""
```

| Client | 文件 | 接入方式 |
|--------|------|---------|
| Qwen2.5-VL-72B | `qwen_client.py` | 本地 vllm / OpenAI 兼容 |
| 混元 hunyuan-t1 | `hunyuan_client.py` | 沿用 planning_llm_server feature/game_claw_v0.1 经验 |
| GPT-4o | 复用 `llm_gpt.py` | OpenAI 标准 |
| Gemini 1.5 Pro | `gemini_client.py` | google-generativeai |
| Claude 3.5 Sonnet | `claude_client.py` | anthropic |

### 4.2 v15 标准化 prompt
- 文件：`cross_vlm/prompt_v15.py`
- 关键：反馈块用 JSON 结构化（不是自由文本），含 `prior_update_hint` 字段
- 模板见 `02_CABTO扩展方案_v15.html` §4.2

### 4.3 跑实验
```bash
python exps_bt_learning/v15_highlight/cross_vlm/run_cross_vlm.py \
    --vlms qwen2.5-vl,hunyuan-t1,gpt-4o,gemini-1.5-pro,claude-3.5-sonnet \
    --tasks all --seeds 3
```

### 4.4 关键指标
- **SR std across VLM**（越小说明越通用）
- 反馈吸收率：diff(BT_前, BT_后) 中 feedback 标的的回避率
- token 成本：反馈块平均 token 数（控制 <1000）

## 5. 章节 chap05 的 LaTeX 落点

- 主章：`latex/data/chap05.tex`（沿用现有，重写为 highlight 结构）
- 配套：`chap05_method.tex` + `chap05_exp.tex` 已存在，可以分别承载方法与实验
- 新增节：5.4 Theoretical Analysis（定理 1 单调收敛 + 定理 2 反馈带宽下界）
- 图表路径：`latex/figures/v15_fig5_*.pdf`

## 6. 验收 checklist

- [ ] 实验① learning curve：CABTO slope ≥ 0.02 且显著高于 baseline
- [ ] 实验② 瀑布图：A4 (skill 可行域) 贡献 ≥ 10%
- [ ] 实验② 雷达图：4 维全展开
- [ ] 实验③ 跨 VLM 柱状图：≥ 3 VLM，SR std < 8%
- [ ] 实验③ 反馈块 prompt 模板归档
- [ ] chap05 总页数 ≥ 35 页
- [ ] 5.4 节至少 1 定理 + 半页证明 sketch
- [ ] 5.5.5 节 t-SNE 三联图（前/后/KL 热图）
- [ ] 5.5.6 节 ≥ 3 个失败 case 截图 + 文字分析

## 7. 4.5 周时间表（按周）

| 周 | 周一-周三 | 周四-周日 | 里程碑 |
|----|---------|---------|--------|
| W1 | 实现 inner_monologue.py | 实现 doremi.py | 2 baselines runnable |
| W2 | 实现 reflect.py + LIBERO 适配 | 跑实验① 主对比 | learning curve 图初稿 |
| W3 | 实验② 组件开关化 + 跑 8 组合 | 实验② 瀑布图 + 写 5.5.3 节 | chap05.5.3 定稿 |
| W4 | 跨 VLM 接口 5 个 | 跑实验③ 150 次 run | 跨 VLM 柱状图 |
| W4.5 | t-SNE + 失败案例 + 5.4 定理 | chap05 全本润色 | chap05 完整定稿 |

## 8. 风险与 fallback

| 风险 | fallback |
|------|---------|
| LIBERO-LONG 适配工作量超 1 周 | 砍 LIBERO，仅用 CABTO 原 task1-7 + RoboCasa 3 任务 |
| Gemini / Claude API 不通 | 砍到 3 VLM (Qwen + 混元 + GPT-4o) |
| baseline 复现 SR 异常 | 用原论文报告值，加注脚说明 |
| 定理 1 严格证明推不出 | 改为 empirical observation + 30 次实验数据 |

## 9. 并行调度（如有师弟师妹）

| 角色 | 任务 |
|------|------|
| 你（主线） | 实验①（最关键 baseline） + 实验③ 跨 VLM + chap05 撰写 |
| 师弟 | 实验② ablation 全套（最机械、最易并行） |
| 师妹 | 失败案例可视化 + t-SNE 主图 + LIBERO 适配 |

## 10. 联系与版本

- 分支：`feature/v15_highlight`（独立分支，不污染 main）
- 提交频率：每周 1 个 milestone PR
- 文档同步：每周更新本 README 的 checklist 状态
- 论文同步：每周把 chap05.tex 进展推到 v15 latex 工程
