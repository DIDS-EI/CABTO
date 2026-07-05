# CABTO v15 高亮工作 — 师弟师妹 Onboarding 一页纸

> **目标读者**：第一次接触 CABTO 项目、要协助做 AAAI 2026 投稿的师弟师妹。
> **承诺**：跟着这页文档走，一周内能跑通 baseline，拿到第一张学习曲线。

---

## 1. 你要协助解决的科学问题（30 秒电梯陈述）

CABTO 要回答 **"大模型驱动的具身智能体，需要什么样的上行反馈接口 $\mathcal{I}_\uparrow$，才能让高层规划器跨 episode *单调改进*？"** 现有 SOTA（Inner Monologue / DoReMi / REFLECT）的反馈都是自由文本或 scalar，香农条件熵 $H(F|\mathcal{I}_\downarrow)$ 不足，跨 episode 学习斜率 $k \approx 0$。我们提出 **4-原子结构化反馈**（pose / contact / dynamics / semantic），由定理 1 保证 $k \geq \alpha \epsilon > 0$。

---

## 2. 分工建议

| 角色 | 一周可上手任务 | 评估指标 |
|---|---|---|
| **🟦 师弟 A（代码强）** | 把 `baselines/inner_monologue_agent.py` 接到真实 OmniGibson 环境，跑 50 任务 ×5 seed | 拿到 `runs/inner_monologue_*.json` |
| **🟩 师弟 B（系统/工程强）** | 接管 `baselines/atoms/{pose,contact,dynamics,semantic}.py` 4 原子检测器骨架（已含 fake IK / fake collision / fake RNEA / Beta-Binomial 后验），把每个文件里 `# TODO(B):` 标记处换成真实 Isaac Sim / curobo / pinocchio 调用 | `python tests/test_atoms.py` 输出 `ALL ATOMS OK`（脚手架已 18/18 通过，替换后不能跌） |
| **🟨 师妹 C（实验/可视化强）** | 用 `eval_v15/metrics.py` 跑 baseline vs CABTO 对比 + 生成学习曲线 PNG | 一张可直接进论文 §5.5 的图 |

每周二晚 21:00 同步 1 小时（线上即可），周日前每人在群里发 1 句话进度。

---

## 3. 一周入门路径

### Day 1：环境搭建（半天 + 抢咖啡时间）
```bash
git clone <repo> && cd CABTO
git checkout -b feature/v15_${YOUR_NAME}     # 各自拉分支，别直接动 main
# Python env
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .            # 安装本项目
pip install numpy matplotlib openai          # baseline 必备
```

### Day 2：跑通 baseline 烟囱测试
```bash
python baselines/inner_monologue_agent.py    # 用 stub LLM 跑一遍
python eval_v15/metrics.py                   # 看到 fake-data 学习曲线即成功
python tests/test_atoms.py                   # 4 原子骨架烟囱测试（18/18）
```

### Day 3：通读必读文献（每篇限 90 分钟，做 1 页脑图）
1. **Inner Monologue** (CoRL 2022) — baseline 思想源
2. **DoReMi** (ICLR 2024) — scalar 反馈代表
3. **REFLECT** (CoRL 2023) — 自然语言反馈代表
4. **TRPO Appendix C** — 定理 1 第三步用到的方向余弦下界

### Day 4-5：上手真任务
- 选 OmniGibson 的 `pickup_object` 5 任务作为最小可重现集；
- 跑 baseline，收集 `runs/*.json`；
- 把指标喂给 `metrics.py`，**复现"baseline 学习曲线接近水平线"** 的现象（这就是 CABTO 的 selling point 的反证）。

### Day 6-7：第一次组会汇报
- 准备 5 张 slide：问题 / baseline 复现结果 / CABTO 直觉 / 下周计划 / 卡点。
- 卡点提前 24h 在群里发，方便提前准备解答。

---

## 4. 你**不需要**自己重新做的事（直接用现成的）

| 已有产物 | 路径 | 作用 |
|---|---|---|
| 论文 PDF | `~/博士论文/v15_*/04_thesis_v15.pdf` | 看 §5 章和附录 A 理解理论 |
| CABTO 扩展方案 HTML | `~/博士论文/v15_*/02_CABTO扩展方案_v15.html` | 10 节方案 + prompt 模板 + 命令行 |
| InnerMonologue baseline | `baselines/inner_monologue_agent.py` | 直接 import，不要重写 |
| 4 原子检测器骨架 | `baselines/atoms/{pose,contact,dynamics,semantic}.py` | 师弟 B 起手脚手架，签名固定，只换实现 |
| 4 原子单元测试 | `tests/test_atoms.py` | 18/18 通过；替换底层实现后必须仍然 ALL OK |
| 评测指标库 | `eval_v15/metrics.py` | 三指标 + 学习曲线 + HTML 报告，一行命令产出 |
| 论文 §5.5 表格占位 | `latex/data/chap05.tex`（4 张 \TODO{} 表） | 实验数据回来直接搜 `\TODO` 替数字 |
| CABTO README | `~/博士论文/v15_*/CABTO_README.md` | checklist + 9 周 roadmap |

---

## 5. 常见陷阱（前人踩过的坑，别再踩）

1. **不要把 4 原子约束写成自然语言再喂 LLM**——退化成 REFLECT，定理 1 不成立。必须是 JSON。
2. **`episode_idx` 必须按 *持续学习* 顺序编号**，不是 task_id；否则学习斜率算出来是噪声。
3. **VLM caption 一定要 cache**——每张图跑一次混元/Qwen 都是钱，做实验前把 caption 文件落盘。
4. **跑 50 任务时设 5 个 seed (0/1/2/3/4)**——AAAI 审稿人最爱挑 seed 数。
5. **写 prompt 时把 skill_library 限定死**——否则 LLM 会幻觉出不存在的技能，影响 SR 统计。

---

## 6. 我（蔡以帅）在哪里、什么时候找我

- **群里 @ 我**：技术卡点（24h 内回复）
- **每周二组会**：研究方向类问题（提前一天发议题）
- **不要做的事**：直接修改 main 分支 / 改 v15 LaTeX / 删 baselines 目录任何东西

---

## 7. 文献与代码致谢

本项目沿用国防科技大学一系列前置工作（Task2Morph / HBTP / MRBTP / Rsync），具体引用关系见论文 §1.4。师弟师妹的贡献都将体现在论文致谢与共同作者署名中（按贡献度排序）。

🚀 **欢迎加入 CABTO！— 蔡以帅 2026-06-13**
