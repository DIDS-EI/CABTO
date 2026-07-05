# 实验2：Low-level Policy Sampling 的「代码生成串接原语」实现

本目录是 CABTO 论文**实验2（Low-level policy sampling）**在 MuJoCo Franka 抓取场景下的
可运行重构。核心思想：把底层策略采样从「写死的状态机脚本」改造成
**「用代码把一组可组合的原语接口串接起来」**——这正是 CABTO 用 VLM（Molmo）+ 运动规划
（cuRobo）做底层 grounding 的精神，在本地可跑、零云依赖的复刻。

技能动作模型与论文一致：
`skill = pick(object)`，`h = ⟨pre, add, del⟩`
= `⟨{on_table(obj)}, {in_gripper(obj), lifted(obj)}, {on_table(obj)}⟩`，
`env.check_success()` 校验 grounding 一致性。

---

## 1. 目录结构

```
exp2_low_level_codegen/
├── assets/panda_assets/           # Franka Panda 网格（STL/OBJ），随包自包含
└── franka_grasp_dp/
    ├── scene/panda_grasp_base.xml # MuJoCo 场景（桌 + Panda + 三相机占位）
    ├── dp_common/                 # 环境与几何工具
    │   ├── franka_dp_env.py       #   FrankaDPEnv：IK + 伺服偏差补偿 + grasp-assist
    │   ├── dp_objects.py          #   5 个家用小物（cube/cylinder/sphere）
    │   └── scripted_expert.py     #   旧 FSM 专家（对照基线）
    ├── exp2_codegen/              # ★ 本次新增：代码生成 + VLM 打点 范式
    │   ├── camera.py              #   相机内外参 / 反投影 / 两视角三角化
    │   ├── primitives.py          #   机械臂原语（move/descend/align/grip/lift）
    │   ├── vlm_pointer.py         #   VLM 打点后端（oracle / qwen / molmo）
    │   ├── codegen_action.py      #   ★「被生成的代码」：pick() 串接原语
    │   ├── eval_pick.py           #   端到端成功率 + 打点像素误差评估
    │   ├── render_viz.py          #   多视角 PNG + rollout MP4 + 打点叠加图
    │   ├── calibrate_qwen_prompt.py  # Qwen 打点 prompt 标定
    │   ├── renders/               #   生成的可视化产物
    │   └── results/               #   评估 JSON
    ├── dp_data_collection/        # DP 数据采集（episode_*.npz）
    ├── diffusion_policy_train/    # DP 训练（含 dp_best.pt / dp_latest.pt）
    └── dp_eval/                   # DP 一致性 / 策略评估
```

## 2. 「代码生成串接原语」范式（实验2 的新语义）

`codegen_action.pick(...)` 函数体本身就是「代码生成器会产出的程序」。它只调用一组
**可组合、语义化的接口**，任何 LLM 代码生成模块都能照此模式拼出 pick / place / push：

```python
# —— 被生成的 pick(obj) 程序（节选）——
uv_o = pointer.point(img_overview, "the object on the table")   # VLM 在主相机打点
uv_f = pointer.point(img_front,    "the object on the table")   # 第二视角打点
p3d  = camera.triangulate(..., uv_o, ..., uv_f)                 # 2D×2 → 3D
runner.approach(p3d, height=0.12)      # 移到正上方（夹爪张开）
runner.open_gripper()
runner.descend_to(p3d)                 # 两段式下探（先对准 xy 再竖直下）
runner.align_xy(p3d)                   # 抓取高度精对准
runner.close_gripper_following(p3d)    # 慢闭合 + tcp 跟随
runner.lift(height=0.30)               # 抬起
```

### 感知链路（描述 → 打点 → 2D→3D → 移动）
1. **至少两个不同视角相机**：`overview`（俯视）+ `front`（正视）（另有 `wrist` 腕部相机）。
2. **VLM 打点接口** `vlm_pointer.PointerBackend.point(image, instruction)`：给一句自然语言
   描述「要打什么点」，模型在相机图上返回像素点 `(u, v)`。
3. **2D→3D**：`camera.triangulate` 用两视角视线求交（**不依赖深度缓冲**，规避软件 GL 的
   深度精度偏置）；退化时用 `camera.deproject_with_depth` 单视角深度反投影。
4. **机械臂移动 + 夹爪开关**：`primitives.PrimitiveRunner` 把目标交给 `env.step`（末端 6D
   delta + grip），与 DP 学到的动作空间完全一致——生成的轨迹本身即合法 DP 数据。

## 3. VLM 后端：Qwen2.5-VL vs Molmo（本地推理对比）

| 后端 | 模型 | 打点机制 | 本地依赖 | 结论 |
|------|------|----------|----------|------|
| `oracle` | 无 | 物体真值投影 | 无 | 确定性，打通链路 / 对照基线 |
| `qwen` | Qwen2.5-VL-3B-Instruct-4bit | grounding，输出 `{"point_2d":[x,y]}` | mlx-vlm（Apple Silicon） | **首选**：单模型即可，~10s/次，简洁 prompt 下 median≈20px(@512)≈1cm |
| `molmo` | Molmo-7B-D-0924-4bit | 原生 pointing 训练，输出 `<point x= y=>` | mlx-vlm，7B 更重 | 对比项，pointing 专精但模型更大 |

**Qwen prompt 标定结论**（`calibrate_qwen_prompt.py` / `results/qwen_prompt_calib.json`）：
Qwen2.5-VL 对**简洁** pointing 指令最准；prompt 一旦写啰嗦、强加 JSON schema、或出现
"grasp" 等词，误差显著变大。最终采用 `Point to {instr}. Output the pixel coordinate.`，
解析器兼容 `point_2d/point_3d/point` 多种 key 与 markdown code fence。

## 4. 运行（cabto venv）

```bash
cd franka_grasp_dp
PY=/Users/yishuaicai/.workbuddy/binaries/python/envs/cabto/bin/python

# 端到端抓取评估（oracle，全物体多种子）
PYTHONPATH=. $PY exp2_codegen/eval_pick.py --backend oracle --seeds 0-5

# 端到端抓取评估（真实 Qwen2.5-VL 本地打点）
PYTHONPATH=. $PY exp2_codegen/eval_pick.py --backend qwen --seeds 0-3

# 可视化（多视角 PNG + rollout MP4 + 打点叠加图 → exp2_codegen/renders/）
PYTHONPATH=. $PY exp2_codegen/render_viz.py --backend oracle

# Qwen 打点 prompt 标定
PYTHONPATH=. $PY exp2_codegen/calibrate_qwen_prompt.py
```

## 5. 关键工程经验（详见 skill `mujoco-franka-grasp-tuning`）

- **两视角三角化定位**：MuJoCo 软件 GL 深度缓冲有系统性偏置（x 向 +0.02m），单视角深度
  反投影定位不准；改两视角视线求交后 xy 误差≈0。
- **抓取高度公式**（对标 ScriptedExpert）：`grasp_z = obj_top_z - 0.6*(top+bottom)`，
  指尖落到物体腰部、略低于几何中心，保证两指跨在物体两侧。
- **精对准积分项 (PI)**：`EE_SERVO_BIAS` 是为开环单步 delta 标定的，在多步闭环里变成恒定
  误差源（工作空间边缘收敛不到 ~2cm）；descend/align/close 加积分项自适应消除，approach
  用纯比例避免超调撞倒物体。
- **细高圆柱降高**：tea_tin 这类细高圆柱重心高易被推倒，几何高度比例 1.6→1.2 后更敦实。
- **显存/内存**：每物体复用 env、用完 `renderer.close()` + `gc.collect()` 防累积 OOM。
- **meshdir 可移植化**：`franka_dp_env._patch_meshdir` 在 build 时把场景 XML 的 meshdir
  重定向到随包复制的 `assets/panda_assets`，整包搬到任意目录都能直接编译。
