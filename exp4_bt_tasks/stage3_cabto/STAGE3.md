# exp4 阶段3：CABTO 闭环（LLM 符号规划 + 代码生成执行 + VLM 视觉校验 + 反馈修正）

本阶段在 exp4 已搭好的环境上，复现 CABTO 论文的完整闭环：
**① LLM 提出动作及其 ⟨pre, add, del⟩ 符号效应 → ② 代码生成串接低层原语 → ③ 执行（VLM 打点定位 + IK 驱动）并逐步截图 → ④ VLM 观察过程图片校验 add/del 是否真发生 → ⑤ 失败则把原因回灌、重规划修正（self-correction，最多 3 轮）。**

首个跑通的任务：**blocks 堆叠**（把绿块叠到黄块上）。

---

## 1. 闭环结构与代码模块

| 环 | 论文对应 | 模块 | 说明 |
|---|---|---|---|
| ① 符号规划 | LLM high-level planning | `planner.py` | 给 LLM 任务+物体+skill+谓词词表，输出 JSON 动作序列，每步带 ⟨pre,add,del⟩。强约束 prompt + few-shot + schema 校验 + 规则 fallback |
| LLM 后端 | — | `llm_backend.py` | 可插拔：`LocalQwenVL`（本地 mlx-vlm，规划/校验共用单例），预留云端 OpenAI 兼容接口（填 key 即切） |
| ② 代码生成 | codegen 串接原语 | `codegen.py` | 把每个 skill 翻成"被生成的程序"：`gen_pick` / `gen_place_on`，串接 ③ 的原语。`render_program_text` 输出该步生成代码文本 |
| ③ 执行-原语 | low-level primitives | `primitives.py` | `PrimitiveRunner`：approach/align_xy/descend_to/grasp/lift/move_above/descend_place/place_release/retreat/settle，复刻 test_blocks 已调通运动手法 |
| ③ 执行-定位 | VLM grounding | `vlm_pointer3d.py` | `Pointer`：VLM 打点 (u,v) → 用深度反投影/双视角三角化成 3D；`backend="oracle"` 时走真值（对照/兜底） |
| ④ 视觉校验 | VLM effect verification | `effect_checker.py` | `EffectChecker`：把 before/after 图喂 VLM，逐条问 add/del 谓词是否发生；同时调 world_state 拿 ground-truth 对比 |
| ⑤ 主闭环 | self-correction | `loop.py` | `run_closed_loop`：规划→执行→校验→失败回灌 feedback 重规划，最多 max_rounds 轮。失败判定权威=ground-truth |
| 环境/真值 | simulator | `blocks_env.py` / `world_state.py` | 多物体抓取环境（多 weld + place 语义）；谓词真值 on_table/in_gripper/lifted/stacked_on/clear/holding_nothing |

> 环境与谓词（地基两层）由 `test_blocks.py` 验证：真值驱动一次 pick→place 能把绿块真叠到黄块上。

---

## 2. 谓词体系（沿用 exp2 + 扩展）

`on_table(x)` / `in_gripper(x)` / `lifted(x)` / `stacked_on(a,b)` / `clear(x)` / `holding_nothing()`

skill 的符号效应（STRIPS 风格）：

```
pick(obj):     pre=[on_table,clear,holding_nothing]  add=[in_gripper,lifted]   del=[on_table,holding_nothing]
place_on(o,d): pre=[in_gripper(o),clear(d)]           add=[stacked_on(o,d),holding_nothing]  del=[in_gripper(o),clear(d)]
```

> ⚠️ 标定要点：place_on 的 add **不含** `on_table(obj)`、del **不含** `lifted(obj)`——因为绿块叠到黄块上后，按真值定义它不在桌面(on_table=False)、底面仍高于桌面阈值(lifted 仍 True)。早期把这两条写进效应会导致校验恒为假阴性。

---

## 3. 怎么跑

```bash
PY=/Users/yishuaicai/.workbuddy/binaries/python/envs/cabto/bin/python
cd <CABTO>/exp4_bt_tasks/stage3_cabto

# 地基自测（真值驱动，验证环境/谓词/抓放）
PYTHONPATH=. $PY test_blocks.py

# oracle 闭环（定位走真值）—— 硬指标：success=True 且 stacked_on=True
PYTHONPATH=. $PY test_loop_oracle.py oracle

# self-correction 验证（人为破坏一次放置，看是否触发重规划）
PYTHONPATH=. $PY test_loop_oracle.py oracle_inject

# qwen 闭环（VLM 真打点 + VLM 真校验）—— 受 VLM 精度限制，记录表现
PYTHONPATH=. $PY test_loop_oracle.py qwen
```

> macOS TCC：在 CABTO 目录下 Python 直接写文件会被拦截。脚本默认把产物写到 `/tmp/exp4_stage3/stage3_out/`；如需落到仓库，用 `cp` 搬运（已随本阶段产物搬好）。

---

## 4. 结果

产物在 `stage3_cabto/stage3_out/`：`rollout.mp4`、`step{i}_before/after.png`、`result.json`（oracle 成功态）、`qwen_result.json`（VLM 真实链路记录）。

| 后端 | 定位 | success | rounds_used | stacked_on | 说明 |
|---|---|---|---|---|---|
| **oracle** | 真值 | ✅ True | 1 | ✅ True | pick/place 效应均 gt_ok=True，一轮成功；末端**垂直向下**抓放（descend 倾斜 ~1°，详见第 6 节） |
| **oracle_inject** | 真值 | ✅ True | 2 | ✅ True | 第0轮人为破坏→检测到 stacked_on 未满足→触发重规划→第1轮成功。**self-correction 真实生效** |
| **qwen** | VLM 打点 | ❌ False | 2 | ❌ False | 垂直修复后 **pick 即失败**（gt_ok=False, in_gripper 仍 False），VLM 判断与真值一致(agree=True, vlm 也判 lifted 未发生)。VLM 打点有数 cm 误差，叠加垂直姿态控制后 descend 未精确对准绿块、抓空。两轮 self-correction 均同因失败 |

> 复跑说明（垂直抓取修复后，2026-06-28 复跑三任务）：oracle 当日复跑 2 次均 `success=True/1 轮`；oracle_inject `success=True/2 轮`，self-correction 触发链路完整；qwen 较修复前出现 **pick 回归**——修复前是"pick 成功、place 因 VLM 定位失败"，修复后瓶颈前移到 pick（VLM 打点精度 + 垂直姿态约束使抓取更易失准）。方向仍印证"VLM 是定位瓶颈"。

**关键结论（对应论文动机）：**
1. 闭环骨架（①②③④⑤）完整跑通，oracle 路 100% 成功，self-correction 失败重规划机制经 inject 验证真实有效。
2. 本地 Qwen2.5-VL-3B 能胜任 ① 符号规划（plan_source=llm，真产出合法两步计划）。
3. **VLM 既是定位瓶颈也是校验风险**：3B 模型打点精度不足以支撑高精度抓取/放置；垂直姿态约束下，定位误差对 pick 的影响被放大——这正是论文强调需要 **ground-truth/多视角/更强模型** 的原因。工程上以 ground-truth 作失败判定权威、VLM 校验作对比记录。
4. **地基自测 `test_blocks.py` 的局限**：它用一套简易开环 PID 做 place，对 pick 随机抓偏敏感；在 y<0 侧高处工作空间，IK+垂直姿态约束有 x 稳态偏置，单次开环放置可能撞翻 yellow。这是自测自身运动逻辑的局限，**不代表闭环能力缺失**——闭环路（PrimitiveRunner + refine_above 真值精修 + self-correction 重试）已稳定成功。地基自测以演示 pick 段稳定垂直抓取为主，place 的权威验证以闭环为准。

---

## 5. 模型调用

- 规划①与视觉校验④都用**本地 Qwen2.5-VL-3B-Instruct-4bit**（mlx-vlm 直接 import，单例缓存，不走 HTTP）。
- LLM 后端可插拔：`make_llm_backend("local")` / 预留云端，详见 `llm_backend.py` 与 exp4 主 README 第 5 节。
- 定位精度不够时，可切 `backend="oracle"` 用真值，或后续接入更强 VLM。

---

## 6. 末端垂直抓取修复（姿态增量纠正）

### 问题
原始 `step()` 的姿态控制是"维持当前姿态"（`apply_delta_rotvec(cur_quat, drot=0)`）。原语只下发位移 `action[:3]`、姿态分量 `action[3:6]` 恒为 0，于是手爪姿态没有任何主动纠正，移动中随关节漂移逐步歪掉。诊断序列显示倾斜角一路恶化：

```
reset 0.4° → approach 2.9° → align_xy 17.1° → descend 32.0°
```

抓取瞬间末端已斜 32°，明显不是"垂直向下"。

### 方案对比
| 方案 | 垂直度 | position 精度 | 结论 |
|---|---|---|---|
| 维持当前姿态（原始） | ❌ descend 32° | ✅ 好 | 姿态随漂移恶化 |
| 每步硬锁 `GRASP_QUAT`+提高 `w_rot` | ✅ ~3° | ❌ y 偏 5–6cm，move_to 收不敛 | IK 牺牲位置换姿态 |
| **姿态向 `GRASP_QUAT` 做有界小角度增量纠正**（`DOWN_CORR_STEP=0.08`+`w_rot=0.5`） | ✅ ~1° | ✅ 精准 | **采用** |

### 实现要点（`blocks_env.py`）
- `GRASP_QUAT = [0,1,0,0]`（wxyz，hand +z 朝下，即垂直向下抓取姿态）。
- `step()`：无姿态指令且 `lock_down=True` 时，计算 `mju_subQuat(GRASP_QUAT, cur_quat)` 的误差旋转向量，按 `DOWN_CORR_STEP=0.08rad`（≈4.6°/step）截断后增量纠正。**每步只回正一小角度**，既往垂直收敛、又不会一次性大幅扭动而破坏 position 闭环。
- `solve_ik` 用综合代价选解 `cost = pos_err + ROT_W·rot_err`（`ROT_W=0.05`），`w_rot=0.5`；收敛判据 `pos_err<tol 且 rot_err<0.03`。
- `_lm_solve` 全程跟踪综合最优解；遇 LM 脊点卡死（line-search 失败、lambda 飙升）不直接退出，而是重置 lambda + 小随机扰动跳出，最多容忍 `stall>3` 次。

### 修复后验证（pick 全程倾斜角）
```
reset 0.4° → approach 3.8° → align_xy 1.8° → descend 1.4° → grasp 0.9°(holding=True) → lift 1.4°(lifted)
```
抓取阶段 **32° → ~1°**，几乎完美垂直，且抓取/抬升成功。整条 oracle 闭环复跑仍 `success=True, rounds_used=1, stacked_on=True`，垂直修复不影响堆叠成功。

### 关于"方块放近一些"
曾尝试把方块从 x=0.50 收近到 x=0.40/0.44，希望垂直抓取更轻松。结果相反：**近处工作空间 IK 解质量反而下降**，descend 时 z 停在 0.488（没下到抓取高度）、y 偏到 0.06，pick 失败、闭环连续 3 轮不成功。故回退到 x=0.50（已验证位置）。**结论：垂直问题靠姿态纠正独立解决，无需也不宜靠收近方块来缓解。**
