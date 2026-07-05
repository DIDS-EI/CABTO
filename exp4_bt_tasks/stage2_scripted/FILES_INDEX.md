# exp4 双臂任务 —— 代码与视频位置索引

> 5 个 exp4 双臂脚本化任务（cover / stack / pour / handover / storage）的源码、成功视频、结果与文档的统一索引。
> 全部任务 `result.json` 均为 `success=True`。

## 权威目录

```
/Users/yishuaicai/mywork/CABTO/exp4_bt_tasks/stage2_scripted/
```

## 一、代码（源文件）

每个任务一对「场景 + 主程序」，外加 3 个共享底座文件。

| 任务 | 场景文件 | 主程序 | 说明 |
|---|---|---|---|
| cover | `scene_cover.py` | `run_cover.py` | 3 物体分别盖到对应 3 目标 |
| stack | `scene_stack.py` | `run_stack.py` | 4 物体堆叠 |
| pour | `scene_pour.py` | `run_pour.py` | 双向互倒聚球 |
| handover | `scene_handover.py` | `run_handover.py` | **真·空中交接版**（一臂在空中把物体传给另一臂）|
| storage | `scene_storage.py` | `run_storage.py` | **单臂提手搬箱上架版**（双臂入箱后搬箱上架）|

**共享底座**

| 文件 | 作用 |
|---|---|
| `exp4_env.py` | 环境层（MuJoCo 封装、step_arms/hold_arms、solve_ik、grasp_assist、weld 控制）|
| `arm_skills.py` | 原语库（pick/place/move_to 等；`move_to` 含限幅积分 `i_cap=0.10`）|
| `scene_dual_common.py` | 双臂场景底座（桌面、双 Franka 基座 ±π/2 布置）|

## 二、视频（成功 rollout，全 success=True）

路径模板：`results/out_<task>/rollout.mp4`

| 任务 | 视频 | 大小 |
|---|---|---|
| cover | `results/out_cover/rollout.mp4` | ~3.3 MB |
| stack | `results/out_stack/rollout.mp4` | ~2.8 MB |
| pour | `results/out_pour/rollout.mp4` | ~1.9 MB |
| handover | `results/out_handover/rollout.mp4` | ~3.6 MB（空中交接）|
| storage | `results/out_storage/rollout.mp4` | ~5.5 MB |

每个 `results/out_<task>/` 目录下还含：
- `result.json` —— 成功判定结果（`success`、各物体落点等）
- `<task>_before.png` / `<task>_after.png` —— 任务前后截图（pour/handover 另有 init 截图）

## 三、结果摘要

| 任务 | result.json task 名 | success |
|---|---|---|
| cover | `cover` | ✅ True |
| stack | `stack` | ✅ True |
| pour | `pour_gather` | ✅ True |
| handover | `handover_air3` | ✅ True |
| storage | `storage_pack_and_shelve` | ✅ True |

## 四、说明文档

`STAGE2.md` —— 5 任务的设计、成功判定口径、关键调试经验（move_to 限幅积分、确定性 snap 落子、空中 weld 转移等）、运行命令。

## 五、运行方式

使用 cabto venv（mujoco 3.10.0）：

```bash
cd /Users/yishuaicai/mywork/CABTO/exp4_bt_tasks/stage2_scripted
PY=~/.workbuddy/binaries/python/envs/cabto/bin/python

# 渲染出 mp4（示例：handover）
HANDOVER_RENDER=1 HO_PHASE=all $PY run_handover.py results/out_handover

# 其余任务类推：run_cover.py / run_stack.py / run_pour.py / run_storage.py
# storage 渲染开关：STORAGE_RENDER=1
```

## 六、备份副本

`/tmp/exp4_tasks/` 是开发期的同步源副本，内容与上述 CABTO 权威目录一致，可作备份。

---
_最后更新：2026-06-30_
