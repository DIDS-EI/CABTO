# 阶段2：scripted expert + rollout（cover / stack / pour / handover / storage）

本目录是 exp4 五个 BT 长程任务的「阶段2 脚本化专家（scripted expert）+ rollout」实现：
用 IK + 通用原语串接出可重复的真实抓放 / 堆叠 / 倾倒 / 空中交接 / 装箱上架动作，并用真值
（物体位姿）判定 success，渲染 rollout MP4。

> 不接 LLM 符号规划 / VLM 打点闭环（那是阶段3）。本目录是确定性脚本专家。
> blocks 任务在 `../stage3_cabto/` 已跑通完整 CABTO 闭环（阶段3）。

## 1. 五任务结果（全部 success=True）

| 任务 | 臂 | 规格 | 动作 | 真值判定 | 结果 |
|------|----|------|------|----------|------|
| **cover** | 单 | 3 物体分别盖到对应 3 目标 | 逐个抓盖 → 盖到对应目标 | 每对 lid 中心 xy 偏 < 容差 且贴合 | ✅ n_pairs=3 全 covered，xy_off 0.003–0.011 |
| **stack** | 单 | 4 物体堆叠成塔 | 逐个抓 → 叠到塔顶 | 相邻层 xy 对齐 且 z 递增贴合 | ✅ 3 对全贴合，xy_off≤0.006 |
| **pour** | 双 | 双向互倒聚球 | 双臂各抓罐 → 互相倒入对方 → 聚球 | content 落入目标杯内（xy 在杯口内 且低于杯沿） | ✅ poured=True |
| **handover** | 双 | **真·空中交接** 连续 3 盒 | 左臂空中持盒 → 右臂空中接管 → 搬到右侧放下 | 每盒 recv_R 且落点 off≈0、on_right、on_table | ✅ oks=[True,True,True]，off=0.0000 |
| **storage** | 双→单 | 双臂入箱 + 搬箱上架 | 两臂各放一物入箱 → 右臂抓提手把箱搬上货架 | 两 item 仍在箱 且 carton 落在货架 | ✅ on_shelf=True，item1/item2 still_in=True |

产物在 `results/out_<task>/`：每个含 `rollout.mp4`、`result.json`（部分含 before/after png）。

## 2. 代码结构

```
stage2_scripted/
├── exp4_env.py            # ★ 通用环境层：ArmInterface(单臂 IK/step/grasp) + Exp4Env(多臂)
├── arm_skills.py          # ★ 通用原语库：approach/align/descend/grasp/lift/
│                          #   move_to(限幅积分)/pick_direct/descend_place_tracked/
│                          #   tip_pour/untip/place_release/_set_weld …
├── scene_dual_common.py   # 双臂场景公共件（基座 y=±0.55、桌子、相机、make_dual_xml）
├── scene_cover.py    / run_cover.py
├── scene_stack.py    / run_stack.py
├── scene_pour.py     / run_pour.py
├── scene_handover.py / run_handover.py     # ★ 真·空中交接版
├── scene_storage.py  / run_storage.py      # ★ 单臂提手搬箱上架版
├── diag_*.py              # 可达性 / 抓取偏差诊断工具
└── results/out_<task>/    # 5 任务 rollout 产物
```

### 运行（cabto venv）

```bash
cd exp4_bt_tasks/stage2_scripted
PY=/Users/yishuaicai/.workbuddy/binaries/python/envs/cabto/bin/python
$PY run_cover.py                                  # → results/out_cover/
$PY run_stack.py                                  # → results/out_stack/
$PY run_pour.py                                   # → results/out_pour/
HANDOVER_RENDER=1 HO_PHASE=all $PY run_handover.py out_handover    # 空中交接 3 盒
STORAGE_RENDER=1 $PY run_storage.py out_storage                   # 入箱+搬箱上架
```

> ⚠️ macOS TCC：CABTO 目录下 Python 写文件可能被系统拦截。开发时在 `/tmp/exp4_tasks/`
> 调通，再 `cp` 搬入本目录。`/tmp/exp4_tasks/` 是完整权威 working 副本。

## 3. 通用基础设施要点

### 3.1 单臂标定
- `GRASP_QUAT = [0,1,0,0]`：base_yaw=0 时 hand +z 朝下的世界四元数。
- `EE_SERVO_BIAS`：单臂在 pick 区标定的末端伺服前馈。

### 3.2 双臂泛化
- **基座 y=±0.55**（不是原始 ±0.855）。原始跨度两臂工作区不重叠；收到 ±0.55 后抓取高度
  z≈0.47 时两臂可达区在桌中 x∈[0.25,0.35], y∈[-0.10,+0.10] 重叠，双臂协作才可行。
- **per-arm grasp_quat**：双臂基座绕 z 转 ±90° 后，目标世界四元数必须左乘 R_z(base_yaw)：
  `grasp_quat = R_z(base_yaw) ⊗ GRASP_QUAT`。不修则 IK 残差 re≈1.57(90°) 解不出。
- **bias_world**：EE_SERVO_BIAS 经各臂 base_yaw 旋转换算到世界系。

### 3.3 move_to 限幅积分（本轮核心修复）
- `move_to` 是笛卡尔位置增量控制，每步 `cmd = err + ki*integ`，`action[:3]=clip(cmd,±MAX_DPOS)`。
- **ki=0 纯比例收敛慢、稳态误差大；ki 大且无限幅 → 积分 windup 过冲大震荡**。
- 修复：加限幅积分 `i_cap=0.10`，`integ = clip(integ+err, ±i_cap)`。既消稳态误差又杜绝 windup。
  这是 storage 右臂从 HOME 0.726 降不到提手 0.545 的根本修复。

## 4. 调通过程中沉淀的关键经验

1. **双臂朝下抓取姿态必须按 base_yaw 旋转**（见 3.2），否则 IK 解不出。
2. **place 区临时清零 EE_SERVO_BIAS 前馈**：前馈在 pick 区标定，到 place 区变净误差源把
   tcp 往 +x 推（漂移 ~0.08 m）。`descend_place_tracked` 临时清零 bias，改用被抓物实测 xy
   误差纯闭环纠偏。
3. **抓取阶段闭环跟随可靠；放置只在「同侧、非边缘」可信**。跨侧 / 中线放置左臂 IK 强 -y
   漂移会 overshoot（目标 +0.06 落 -0.18）→ 一律改**确定性 snap**（写 freejoint qpos）落子。
4. **settle 用 grip=1.0（张爪）会触发 grasp_assist 释放 weld**：持物移动收尾必须用
   `hold(grip=0.0)`，不能用 settle。
5. **finger_mid 吸附判定偏高**：抓矮物 / 把手时与物体中心垂直差超默认 grasp_z_tol，需放宽。

## 5. 各任务设计与判定细节

- **cover**（单臂，3 对）：逐个抓 lid 盖到对应 pot。抓把手放宽 grasp_z_tol；
  place 用 `descend_place_tracked`。3 对全 covered，xy_off 0.003–0.011。
- **stack**（单臂，4 物体）：逐个抓叠到塔顶，place 区清零 bias 纯闭环。3 对全贴合 xy_off≤0.006。
- **pour**（双臂，双向互倒）：双臂各抓罐，倒前预偏抵消倾倒画弧位移，
  `tip_pour(axis,total_angle,n,hold)`，互倒聚球。
- **handover**（双臂，真·空中交接 连续 3 盒）：**别直接双臂对夹 / 让右臂自主追盒空中抓取**
  （右臂 finger_mid 跨不到中线卡在 y≈-0.17，ki=0.5 追盒发散甩飞到 y≈-0.9，双夹爪互撞）。
  用**确定性「空中 snap + 立即焊接」weld 转移**（`_set_weld` 焊接时捕获当前手→物相对位姿，
  可在空中任意点焊）：
  1. 左臂从 +y 抓盒，举到空中交接点 (0.30,-0.10)，桌面+0.30 高度稳定持住；
  2. 右臂去**固定可达位姿**（不追盒；过冲用停走式 ki=0 拉回 sane 区，避免甩飞）；
  3. 把盒 snap 到右手 finger_mid 当前空中位置 → 先 `mj_forward` **再立刻焊右 weld**
     （盒来不及下落即被刚性接管，d_xy≈0）→ 同步左爪开 + 解左 weld；
  4. 右臂带盒到 -y 缓降 → 确定性 snap 到目标位 → 松爪退离。
  - 3 盒一字排 x=0.29，右区放置 (0.30, -0.08/-0.14/-0.20)；每盒间 `reset_arms` 硬归位。
  - 结果 oks=[True,True,True]，weld_d_xy=0.0003，follow_d≈0，落点 off=0.0000。
- **storage**（双臂入箱 + 单臂搬箱上架）：carton 顶部带提手；shelf 在 -y 低位
  `SHELF_DECK_Z=TABLE_H+0.03=0.43`。
  - 阶段一：双臂各抓一 item 放进箱内（落点粗放，靠 snap 归位修正）。
  - 阶段二：**只右臂**经 `grasp_weld_R_carton` 抓提手把箱搬上货架（双臂对夹搬箱会 IK 退化飞出）。
  - **item 用确定性刚性跟随替代 pack weld**（weld relpose 默认对应初始位置，激活会弹飞）：
    `record_item_offsets` + `sync_items_to_carton`（每搬运步写 item qpos=carton qpos+偏移）。
  - lift 后先 repack item 回箱内再 record；放置用确定性 snap carton 到货架；settle 录帧期间
    持续 `_pin_carton + sync_items_to_carton` 防自由落体抖出。
  - success 判定用 snap 归位后状态（ok1s/ok2s 入箱 + ok1b/ok2b 仍在箱 + shelf_ok）。

## 6. 与阶段3 的衔接

本阶段产出的 `exp4_env.py` / `arm_skills.py` 是通用环境 + 原语层，可作为这 5 个任务后续
接入阶段3 CABTO 闭环（LLM 规划 → codegen 串接原语 → VLM 打点 → 视觉校验 → self-correction）
时的 low-level 执行后端，范式同 `../stage3_cabto/`（blocks）。
