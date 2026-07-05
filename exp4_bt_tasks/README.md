# 实验4：BT 长程操作任务（5 任务 · 单/双 Franka）

本目录是 CABTO 论文 **manipulation 任务图（Fig. 5 个长程任务）** 在 MuJoCo 下的可运行复现。
在 exp1（BT 学习）与 exp2（low-level codegen + VLM 打点）之上，补充 **5 个带行为树（BT）
结构的长程操作任务**，覆盖单臂与双臂协作两类：

| # | 任务 | 臂数 | 场景描述 | BT 目标 |
|---|------|------|----------|---------|
| 1 | **cover** | 单 Franka | 黑锅(固定) + 锅盖(可抓) + 面包 + 白盘 + 红球 | 把锅盖盖到锅上 |
| 2 | **blocks** | 单 Franka | 蓝球 + 黄块 + 绿块 | 把方块堆叠 |
| 3 | **pour** | 双 Franka | 左臂持红罐、右臂扶白杯 | 把罐内倒入杯中 |
| 4 | **handover** | 双 Franka | 左臂抓绿盒 + 桌面绿色点缀物 | 抬到中间手手交接给右臂 |
| 5 | **storage** | 双 Franka | 纸箱(固定) + 两个绿色物体 | 双臂把物体放进纸箱 |

---

## 1. 目录结构

```
exp4_bt_tasks/
├── assets/panda_assets/        # Franka Panda 网格（67 mesh，随包自包含，与 exp2 同源）
├── control/
│   └── franka_mj_controller.py # MjFrankaController：move_ee_pose/move_tcp/solve_ik/
│                               #   set_gripper/drive_to（复用自 mujoco_workspace/franka_articulated）
├── scenes/
│   ├── panda_body.py           # 可加 _L/_R 后缀的固定基座 Panda body 生成器
│   ├── scene_builder.py        # ★ 5 个任务场景定义（单/双臂、桌子、相机、物体）
│   ├── render_static.py        # 用 READY 预备姿态渲染静态预览图
│   ├── scene_cover.xml         # ↓ 生成的 5 个场景（由 scene_builder.write_all 产出）
│   ├── scene_blocks.xml
│   ├── scene_pour.xml
│   ├── scene_handover.xml
│   └── scene_storage.xml
├── renders/                    # 静态预览图（static_*.png + exp4_all_scenes.png 拼图）
├── stage2_scripted/            # ★ 阶段2：cover/pour/handover/storage 的 scripted expert
│   ├── exp4_env.py             #   + rollout（通用环境层 + 通用原语库 + 4 任务 + 诊断 + 产物）
│   ├── arm_skills.py
│   ├── scene_dual_common.py
│   ├── scene_*.py / run_*.py   #   4 任务场景与脚本专家
│   ├── diag_*.py               #   可达性 / 抓取偏差诊断工具
│   ├── results/out_<task>/     #   rollout.mp4 + before/after.png + result.json（全 success）
│   └── STAGE2.md               #   阶段2 文档（结果表 + 设计 + 5 条关键经验）
├── stage3_cabto/               # 阶段3：blocks 任务完整 CABTO 闭环（LLM→codegen→VLM→校验）
└── results/                    # 阶段2/3 的 rollout / 评估产物
```

## 2. 场景设计约定（scene_builder.py）

- **桌子**：固定 box，桌面高度 `TABLE_H = 0.40`。臂基座钉在桌面高度上（固定基座，
  非 freejoint+weld）。
- **单臂**：基座 `SINGLE_BASE = (-0.18, 0, 0.40)`，面向 +x（桌子在前方）；桌板半尺寸
  `TABLE_HALF = (0.45, 0.55, 0.02)`。
- **双臂**：两臂隔桌相对而立，便于 pour/handover/storage 协作。
  - 左臂 `LEFT_BASE = (0.30, +0.855, 0.40)`，转 −90° 面向桌中心（−y）。
  - 右臂 `RIGHT_BASE = (0.30, −0.855, 0.40)`，转 +90° 面向桌中心（+y）。
  - 双臂桌板半尺寸 `TABLE_HALF_DUAL = (0.45, 0.925, 0.02)`，即 **y 向全长 1.85 m**，
    让两臂离得足够远。
- **命名**：双臂场景里两臂的所有关节/执行器/夹爪加 `_L` / `_R` 后缀区分；单臂无后缀。
  自由度：单臂 `nu = 8`（7 关节 + 1 夹爪），双臂 `nu = 16`。
- **物体**：可抓取物体用 `freejoint`；目标容器（锅 / 纸箱 / 白盘）固定。
- **相机**：每场景一个 `overview` 相机，构图对齐论文图。单臂用 `_overview_cam()`，
  双臂因桌子更长用 `_overview_cam_dual()`（相机进一步拉远抬高）。

## 3. 运行（cabto venv）

> 统一用 cabto 虚拟环境（mujoco 3.10.0）：
> `PY=/Users/yishuaicai/.workbuddy/binaries/python/envs/cabto/bin/python`
>
> ⚠️ **macOS TCC 注意**：直接在 CABTO 目录下用 Python 写文件会被系统拦截。
> 让脚本把产物输出到 `/tmp/exp4_scenes`，再 `cp` 搬回本目录。

```bash
cd exp4_bt_tasks/scenes
PY=/Users/yishuaicai/.workbuddy/binaries/python/envs/cabto/bin/python

# (1) 重新生成 5 个场景 XML（输出到 /tmp 再搬回）
PYTHONPATH=. $PY -c "import scene_builder as sb; sb.write_all('/tmp/exp4_scenes')"
cp /tmp/exp4_scenes/scene_*.xml .

# (2) 渲染 5 张静态预览图（READY 预备姿态）
PYTHONPATH=. $PY render_static.py . /tmp/exp4_scenes
cp /tmp/exp4_scenes/static_*.png ../renders/
```

渲染结果在 `renders/`：`static_cover.png` / `static_blocks.png` / `static_pour.png` /
`static_handover.png` / `static_storage.png`，以及五合一拼图 `exp4_all_scenes.png`。

## 4. 机械臂控制（control/franka_mj_controller.py）

复用自 `DP_robosuite/mujoco_workspace/franka_articulated/` 的 `MjFrankaController`，
是阶段2 脚本化专家（scripted expert）的执行后端：

- `solve_ik(target_pose)`：阻尼最小二乘 IK，从末端 6D 位姿解关节角。
- `move_ee_pose / move_tcp / drive_to`：把末端伺服到目标位姿（逐步收敛）。
- `set_gripper(open/close)`：夹爪开合。

双臂场景下，对左右臂分别用 `_L` / `_R` 后缀的关节/执行器名各实例化一个控制器即可。

## 5. 模型调用：VLM 打点（阶段3，复用 exp2）

exp4 **不新写 VLM 推理代码**，直接复用 exp2 的打点链路：

- **实现**：`../exp2_low_level_codegen/franka_grasp_dp/exp2_codegen/vlm_pointer.py`
  - `QwenVLPointer`（**首选**）/ `MolmoPointer` / `OraclePointer` 三个可插拔后端。
- **推理方式**：**mlx-vlm 本地 `import` 推理，不走 HTTP 服务**
  （`from mlx_vlm import load, generate, apply_chat_template`），Apple Silicon 上直接跑。
  对比 ollama：ollama 自带 HTTP 服务但吃 GGUF 格式、对 VL 模型支持滞后；本项目模型是
  MLX 格式，只能用 mlx-vlm。需要 HTTP 时可用 `mlx_vlm.server` 一行起 OpenAI 兼容服务，
  但 codegen 链路是 Python 的，直接 import 最省事。
- **模型**：`mlx-community/Qwen2.5-VL-3B-Instruct-4bit`，已拉到本地
  `~/.cache/huggingface/hub/`（约 2.9 GB，4-bit 量化）。
- **prompt**：`"Point to {instr}. Output the pixel coordinate."`（exp2 标定结论：
  简洁指令最准，median ≈ 20px@512 ≈ 1cm；啰嗦/强加 JSON schema/出现 "grasp" 反而变差）。
- **链路**：自然语言描述 → VLM 在 overview/front 两视角打点 → 两视角三角化得 3D 坐标
  → `MjFrankaController` 驱动末端抓取。详见 exp2 README 第 2~3 节。

## 6. 三阶段进度

- [x] **阶段1（已完成）**：按论文图搭好 5 个 BT 任务场景，全部通过 MuJoCo 加载校验，
  并渲染静态预览图。桌长经多轮调整最终定为双臂 1.85 m。
- [x] **阶段2（已完成）**：为每个任务写 scripted expert（IK + 原语串接），渲染 rollout MP4。
  blocks 的 pick→place 抓放手法在 `stage3_cabto/test_blocks.py` 调通；其余 4 个任务
  （**cover / pour / handover / storage**）的 scripted expert 在 `stage2_scripted/` 全部
  跑通，真值判定 **success=True**（cover 盖合 / pour 内容物入杯 / handover 右臂接物抬起 /
  storage 双物入箱）。建立了通用环境层 `exp4_env.py` + 通用原语库 `arm_skills.py`，并沉淀
  双臂可达区、per-arm grasp_quat 旋转、place 区清零伺服前馈等 5 条关键经验。详见
  `stage2_scripted/STAGE2.md`。
- [x] **阶段3（已完成）**：在 `stage3_cabto/` 跑通 CABTO 完整闭环——LLM 符号规划
  (⟨pre,add,del⟩) → 代码生成串接原语 → VLM 打点定位 + IK 执行 + 逐步截图 →
  VLM 视觉校验效应 → 失败回灌重规划（self-correction，≤3 轮）。
  首个任务 blocks 堆叠：**oracle 闭环一轮成功**（stacked_on=True），**self-correction
  经 inject 验证真实触发**，qwen 真实 VLM 链路如实记录了定位精度瓶颈与校验乐观误判。
  详见 `stage3_cabto/STAGE3.md`。

## 7. 关联实验

| 实验 | 关系 |
|------|------|
| **exp1**（`exps_bt_learning/`） | BT 结构学习；exp4 沿用其 BT 长程任务语义 |
| **exp2**（`exp2_low_level_codegen/`） | low-level codegen + VLM 打点；exp4 阶段3 直接复用其 `vlm_pointer` 与原语范式 |
| **exp3** | （见根目录《论文实验结果复现.md》） |

> 全仓库实验总览见根目录 `README.md` 与《论文实验结果复现.md》。
