"""数据采集脚本（MuJoCo 版）：Franka 双相机抓取 → Diffusion Policy 训练数据
======================================================================
对应 CABTO 论文 **实验2 · Low-level policy sampling**：
用脚本化专家 π（ScriptedExpert）在「末端 6D delta 动作空间」内执行 pick 技能，
逐帧记录观测-动作轨迹，并记录每集的**符号前/后状态**用于动作模型一致性分析。

动作模型语义映射（CABTO 实验2）
--------------------------------
    skill   = pick(object)
    h.pre   = {on_table(obj)}            采集开始：物体静置桌面（reset 后即满足）
    π       = ScriptedExpert             低层控制策略，输出末端 delta 序列
    h.add   = {in_gripper(obj), lifted(obj)}   π 执行后物体被夹爪抬离桌面
    h.del   = {on_table(obj)}            物体离开桌面
    验证    = env.check_success()         物体抬升超阈值 => 该次采样与 h 一致

逐帧记录（标准 BC 对齐：action[t] 在 obs[t] 下执行得到 obs[t+1]）：
    - front_rgb : 正面第三视角相机 RGB        (T, H, W, 3) uint8
    - wrist_rgb : 腕部相机 RGB                (T, H, W, 3) uint8
    - agent_pos : 本体状态(末端pos+quat+夹爪)  (T, 10)     float32
    - action    : 末端 6D 位姿 delta + 夹爪     (T, 7)      float32

每条 episode 存为一个 .npz。物体类型 / 位置随机。**仅保存成功的 episode**
（成功 = π 执行结果与动作模型 h 一致），这正是「low-level policy sampling」要
采集的「可行落地实例」。

用法:
    python collect_data.py --num_episodes 50 --object_group all
    python collect_data.py --num_episodes 80 --fixed_object wood_block --img_size 96
    python collect_data.py --num_episodes 5 --record demo.mp4   # 附带录演示视频

输出: dp_data_collection/data/episode_XXXX.npz  +  meta.json  +  sampling_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# 让 "dp_common" 包可被导入（脚本位于 dp_data_collection/ 下）
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)            # franka_grasp_dp/
sys.path.insert(0, _ROOT)

from dp_common.franka_dp_env import (  # noqa: E402
    FrankaDPEnv, ACTION_DIM, AGENT_POS_DIM, TABLE_TOP_Z,
)
from dp_common.scripted_expert import ScriptedExpert  # noqa: E402
from dp_common import dp_objects                        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="DP 数据采集（MuJoCo Franka 双相机抓取）")
    p.add_argument("--num_episodes", type=int, default=50, help="目标采集成功 episode 数")
    p.add_argument("--max_steps", type=int, default=120, help="单集最大步数")
    p.add_argument("--img_size", type=int, default=96, help="相机方形分辨率")
    p.add_argument("--object_group", choices=["primitive", "mesh", "all"],
                   default="all", help="物体集合（MuJoCo 版均为几何基元）")
    p.add_argument("--fixed_object", type=str, default=None,
                   help="只采集指定单一物体（如 wood_block），覆盖 object_group")
    p.add_argument("--out_dir", type=str, default=os.path.join(_THIS, "data"),
                   help="输出目录")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_attempts_factor", type=int, default=3,
                   help="最多尝试 num_episodes*factor 集以收集足够成功样本")
    p.add_argument("--record", type=str, default=None,
                   help="额外录制前若干集为 mp4 演示（需可用渲染）")
    p.add_argument("--record_episodes", type=int, default=3, help="录制前几集")
    p.add_argument("--easy", action="store_true",
                   help="收窄物体落点（等价 DP_EASY=1），提高采集成功率")
    p.add_argument("--no_render", action="store_true",
                   help="关闭相机渲染（沙箱/无 GPU 时只采状态与动作，图像为零图）")
    return p.parse_args()


# ----- 符号状态谓词（用于动作模型 h 的前/后条件记录） ----- #
def symbolic_state(env: FrankaDPEnv) -> dict:
    obj_pos, _ = env.get_object_pose()
    obj_z = float(obj_pos[2])
    on_table = obj_z < (TABLE_TOP_Z + env._cur_bottom + 0.03)
    lifted = obj_z > (TABLE_TOP_Z + env._cur_bottom + 0.10)
    return {
        "obj_z": round(obj_z, 4),
        "on_table": bool(on_table),
        "lifted": bool(lifted),
        "in_gripper": bool(env.is_holding()),
    }


def run_episode(env: FrankaDPEnv, expert: ScriptedExpert, max_steps: int,
                seed: int, record: bool):
    """跑一集，返回 (frames_dict, success, pre_state, post_state)。"""
    obs = env.reset(seed=seed)
    expert.reset()
    pre_state = symbolic_state(env)   # h.pre：物体静置桌面

    front_list, wrist_list, agent_list, action_list = [], [], [], []
    if record:
        env.start_record()

    success = False
    for t in range(max_steps):
        action, done = expert.act()
        # 记录「当前观测 + 即将执行的动作」
        front_list.append(obs["front_rgb"])
        wrist_list.append(obs["wrist_rgb"])
        agent_list.append(obs["agent_pos"])
        action_list.append(action)

        obs, info = env.step(action)
        if info["success"]:
            success = True
        if done:
            break

    post_state = symbolic_state(env)  # h.add/del：π 执行后的状态
    data = {
        "front_rgb": np.asarray(front_list, dtype=np.uint8),
        "wrist_rgb": np.asarray(wrist_list, dtype=np.uint8),
        "agent_pos": np.asarray(agent_list, dtype=np.float32),
        "action": np.asarray(action_list, dtype=np.float32),
    }
    return data, success, pre_state, post_state


def main():
    args = parse_args()
    if args.easy:
        os.environ["DP_EASY"] = "1"
    os.makedirs(args.out_dir, exist_ok=True)

    # 物体类型在 build() 时固定（MuJoCo 编译后不可增删 body），
    # 按「物体类型轮转」逐类单独 build env 采集，得到真正多物体混合数据集。
    if args.fixed_object is not None:
        obj_names = [args.fixed_object]
        n_types = 1
        quota = [args.num_episodes]
        print(f">> 单一物体模式：只采集 {args.fixed_object}")
    else:
        specs = dp_objects.get_registry(args.object_group)
        obj_names = [s.name for s in specs]
        n_types = len(obj_names)
        quota = [args.num_episodes // n_types] * n_types
        for i in range(args.num_episodes - sum(quota)):
            quota[i % n_types] += 1

    saved = 0
    total_attempts = 0
    obj_counter = {}
    sampling_log = []      # 实验2：记录每次采样的 h.pre / h.post / 一致性
    t0 = time.time()
    global_seed = args.seed

    for ti, (obj_name, n_target) in enumerate(zip(obj_names, quota)):
        if n_target <= 0:
            continue
        print(f"\n=== 采集物体 [{ti + 1}/{n_types}] {obj_name}：目标 {n_target} 集 ===")
        env = FrankaDPEnv(
            img_size=args.img_size,
            fixed_object=obj_name,
            show_viewer=False,
            record_cam=(args.record is not None),
            seed=global_seed,
            render_enabled=(not args.no_render),
        )
        env.build()
        expert = ScriptedExpert(env, rng=np.random.default_rng(global_seed))

        got = 0
        attempts = 0
        max_attempts = n_target * args.max_attempts_factor + 5

        while got < n_target and attempts < max_attempts:
            ep_seed = global_seed + 1000 + attempts
            record_this = (args.record is not None) and (saved < args.record_episodes)

            data, success, pre_s, post_s = run_episode(
                env, expert, args.max_steps, ep_seed, record_this)
            attempts += 1
            total_attempts += 1

            # 一致性：π 执行后是否达成 h.add（物体被抬起且夹持）
            consistent = bool(success and post_s["lifted"])
            sampling_log.append({
                "object": obj_name, "seed": ep_seed,
                "pre": pre_s, "post": post_s,
                "success": bool(success), "consistent_with_h": consistent,
                "steps": int(data["action"].shape[0]),
            })

            if record_this and args.record is not None:
                rec_path = args.record if args.record.endswith(".mp4") else args.record + ".mp4"
                rec_path = rec_path.replace(".mp4", f"_{saved:02d}.mp4")
                env.stop_record(rec_path, fps=30)
                print(f"   [录制] 演示视频 -> {rec_path}")

            if not success:
                continue

            path = os.path.join(args.out_dir, f"episode_{saved:04d}.npz")
            np.savez_compressed(path, **data)
            obj_counter[obj_name] = obj_counter.get(obj_name, 0) + 1
            saved += 1
            got += 1
            T = data["action"].shape[0]
            print(f"[成功 {saved}] {obj_name} ({got}/{n_target}) 步数={T} "
                  f"obj_z={post_s['obj_z']:.3f} -> {os.path.basename(path)}")

        if got < n_target:
            print(f"   [提醒] {obj_name} 仅采到 {got}/{n_target} 集（尝试 {attempts} 次）")

        global_seed += 1
        del expert, env

    # 实验2 统计：采样总数、一致样本数、各物体可达率
    n_attempt = len(sampling_log)
    n_consistent = sum(1 for r in sampling_log if r["consistent_with_h"])
    per_obj = {}
    for r in sampling_log:
        o = r["object"]
        per_obj.setdefault(o, [0, 0])
        per_obj[o][1] += 1
        if r["consistent_with_h"]:
            per_obj[o][0] += 1
    per_obj_rate = {o: round(c / n, 3) if n else 0.0 for o, (c, n) in per_obj.items()}

    meta = {
        "num_episodes": saved,
        "attempts": total_attempts,
        "img_size": args.img_size,
        "object_group": args.object_group,
        "action_dim": ACTION_DIM,
        "agent_pos_dim": AGENT_POS_DIM,
        "action_format": "[dx,dy,dz, drx,dry,drz, grip]; 平移m, 旋转rotvec(rad), grip 1=open 0=close",
        "cameras": ["front (third-person)", "wrist (attached to hand)"],
        "object_distribution": obj_counter,
        "render_enabled": (not args.no_render),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    exp2 = {
        "skill": "pick(object)",
        "h": {"pre": ["on_table(obj)"],
              "add": ["in_gripper(obj)", "lifted(obj)"],
              "del": ["on_table(obj)"]},
        "policy": "ScriptedExpert (end-effector 6D delta FSM)",
        "n_sampled": n_attempt,
        "n_consistent_with_h": n_consistent,
        "consistency_rate": round(n_consistent / n_attempt, 3) if n_attempt else 0.0,
        "per_object_consistency_rate": per_obj_rate,
        "records": sampling_log,
    }
    with open(os.path.join(args.out_dir, "sampling_log.json"), "w") as f:
        json.dump(exp2, f, indent=2, ensure_ascii=False)

    print(f"\n采集完成: 成功 {saved} 集 / 尝试 {total_attempts} 次, 用时 {meta['elapsed_sec']}s")
    print(f"物体分布: {obj_counter}")
    print(f"[实验2] low-level policy 采样一致率: {exp2['consistency_rate']} "
          f"({n_consistent}/{n_attempt})  各物体: {per_obj_rate}")
    print(f"输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
