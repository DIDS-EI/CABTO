"""实验2 评估脚本（MuJoCo 版）：动作模型一致性验证 + Diffusion Policy 闭环评估
==============================================================================
对应 CABTO 论文 **实验2 · Low-level policy sampling** 的评估环节。

两种评估模式（可单独或一起跑）：

(A) consistency  —— 动作模型 grounding 一致性验证（实验2 的核心指标）
    用脚本化专家 π（ScriptedExpert）在一批 **全新随机种子** 的初始状态上执行
    skill=pick(object)，逐次检查 π 的执行结果是否与动作模型
        h = ⟨pre, add, del⟩ = ⟨{on_table(obj)}, {in_gripper(obj),lifted(obj)}, {on_table(obj)}⟩
    一致。统计整体一致率与各物体一致率。这一批种子与 collect_data 采集
    所用种子区间（global+1000+attempts）刻意错开，作为「留出验证集」，
    用来检验 h 的 grounding 在未见初始状态上是否依然成立（泛化）。

(B) policy       —— 训练好的 Diffusion Policy 闭环成功率评估
    加载 checkpoint（含权重 + 归一化统计），在仿真里以 receding-horizon
    方式滚动执行 DP 预测的末端 delta 动作，统计抓取成功率。
    若没有可用 checkpoint（尚未训练）则自动跳过。
    注意：image-based DP 需要相机渲染；若运行环境无 GPU 渲染（图像全黑），
    DP 评估结果不具参考性，会在报告里标注 render_warning。

用法:
    # 只跑一致性验证（不需要 torch / checkpoint）
    python eval.py --mode consistency --episodes_per_object 8 --easy

    # 只跑 DP 闭环评估（需要已训练的 checkpoint）
    python eval.py --mode policy --ckpt ../diffusion_policy_train/checkpoints/dp_best.pt

    # 两者都跑
    python eval.py --mode both --episodes_per_object 8 --easy

输出: dp_eval/results/eval_consistency.json / eval_policy.json / eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# 让 "dp_common" / "diffusion_policy_train" 包可被导入（脚本位于 dp_eval/ 下）
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)            # franka_grasp_dp/
sys.path.insert(0, _ROOT)

from dp_common.franka_dp_env import (  # noqa: E402
    FrankaDPEnv, ACTION_DIM, AGENT_POS_DIM, TABLE_TOP_Z,
)
from dp_common.scripted_expert import ScriptedExpert  # noqa: E402
from dp_common import dp_objects                        # noqa: E402


# 动作模型 h（与 collect_data.py 保持一致）
ACTION_MODEL_H = {
    "pre": ["on_table(obj)"],
    "add": ["in_gripper(obj)", "lifted(obj)"],
    "del": ["on_table(obj)"],
}


def parse_args():
    p = argparse.ArgumentParser(description="实验2 评估：一致性验证 + DP 闭环")
    p.add_argument("--mode", choices=["consistency", "policy", "both"],
                   default="consistency", help="评估模式")
    p.add_argument("--episodes_per_object", type=int, default=8,
                   help="一致性/策略评估每个物体的试验次数")
    p.add_argument("--max_steps", type=int, default=120, help="单集最大步数")
    p.add_argument("--img_size", type=int, default=96, help="相机方形分辨率")
    p.add_argument("--object_group", choices=["primitive", "mesh", "all"],
                   default="all", help="评估物体集合")
    p.add_argument("--fixed_object", type=str, default=None,
                   help="只评估指定单一物体")
    p.add_argument("--out_dir", type=str, default=os.path.join(_THIS, "results"))
    p.add_argument("--seed", type=int, default=7777,
                   help="评估种子基址（与采集区间错开，作留出验证集）")
    p.add_argument("--easy", action="store_true",
                   help="收窄物体落点（等价 DP_EASY=1）")
    p.add_argument("--no_render", action="store_true",
                   help="关闭相机渲染（一致性验证不依赖图像）")
    # DP 策略评估
    p.add_argument("--ckpt", type=str,
                   default=os.path.join(_ROOT, "diffusion_policy_train",
                                        "checkpoints", "dp_best.pt"),
                   help="DP checkpoint 路径（policy 模式）")
    p.add_argument("--action_horizon", type=int, default=8,
                   help="DP receding-horizon 每次实际执行的动作步数")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 符号状态谓词（与 collect_data.symbolic_state 一致）
# --------------------------------------------------------------------------- #
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


def check_h_consistency(pre: dict, post: dict) -> tuple[bool, dict]:
    """检查一次执行 (pre -> post) 是否满足动作模型 h。
    返回 (consistent, 逐谓词检查明细)。"""
    detail = {
        # h.pre 在执行前必须成立
        "pre.on_table(obj)": bool(pre["on_table"]),
        # h.add 在执行后必须成立
        "add.in_gripper(obj)": bool(post["in_gripper"]),
        "add.lifted(obj)": bool(post["lifted"]),
        # h.del 在执行后必须不成立（物体已离桌）
        "del.on_table(obj)=False": bool(not post["on_table"]),
    }
    consistent = all(detail.values())
    return consistent, detail


# --------------------------------------------------------------------------- #
# (A) 一致性验证
# --------------------------------------------------------------------------- #
def run_consistency(args) -> dict:
    if args.easy:
        os.environ["DP_EASY"] = "1"

    if args.fixed_object is not None:
        obj_names = [args.fixed_object]
    else:
        specs = dp_objects.get_registry(args.object_group)
        obj_names = [s.name for s in specs]

    records = []
    per_obj = {}
    t0 = time.time()
    global_seed = args.seed

    for ti, obj_name in enumerate(obj_names):
        print(f"\n=== [一致性] 物体 [{ti+1}/{len(obj_names)}] {obj_name}："
              f"{args.episodes_per_object} 次试验 ===")
        env = FrankaDPEnv(
            img_size=args.img_size, fixed_object=obj_name,
            show_viewer=False, record_cam=False, seed=global_seed,
            render_enabled=(not args.no_render),
        )
        env.build()
        expert = ScriptedExpert(env, rng=np.random.default_rng(global_seed))

        per_obj.setdefault(obj_name, [0, 0])
        for k in range(args.episodes_per_object):
            ep_seed = global_seed + 50000 + k     # 与采集区间(1000+)错开
            env.reset(seed=ep_seed)
            expert.reset()
            pre = symbolic_state(env)

            for _ in range(args.max_steps):
                action, done = expert.act()
                _, info = env.step(action)
                if done:
                    break
            post = symbolic_state(env)

            consistent, detail = check_h_consistency(pre, post)
            per_obj[obj_name][1] += 1
            if consistent:
                per_obj[obj_name][0] += 1
            records.append({
                "object": obj_name, "seed": ep_seed,
                "pre": pre, "post": post,
                "consistent_with_h": consistent, "detail": detail,
            })
            mark = "✓" if consistent else "✗"
            print(f"  [{mark}] {obj_name} seed={ep_seed} "
                  f"obj_z {pre['obj_z']:.3f}->{post['obj_z']:.3f} "
                  f"add(grip={post['in_gripper']},lift={post['lifted']})")

        global_seed += 1
        del expert, env

    n = len(records)
    n_ok = sum(1 for r in records if r["consistent_with_h"])
    per_obj_rate = {o: round(c / t, 3) if t else 0.0 for o, (c, t) in per_obj.items()}

    result = {
        "task": "experiment2 / low-level policy sampling — consistency eval",
        "skill": "pick(object)",
        "action_model_h": ACTION_MODEL_H,
        "policy": "ScriptedExpert (end-effector 6D delta FSM)",
        "validation_protocol": "held-out random seeds (base=%d, offset=50000), "
                               "disjoint from collect_data seeds (offset=1000)" % args.seed,
        "n_trials": n,
        "n_consistent_with_h": n_ok,
        "consistency_rate": round(n_ok / n, 3) if n else 0.0,
        "per_object_consistency_rate": per_obj_rate,
        "easy_spawn": bool(args.easy),
        "elapsed_sec": round(time.time() - t0, 1),
        "records": records,
    }
    return result


# --------------------------------------------------------------------------- #
# (B) DP 闭环评估
# --------------------------------------------------------------------------- #
def run_policy(args) -> dict:
    import torch
    from diffusion_policy_train.model import DiffusionPolicy
    from diffusion_policy_train.dataset import normalize, unnormalize

    if args.easy:
        os.environ["DP_EASY"] = "1"

    if not os.path.exists(args.ckpt):
        print(f"[policy] 未找到 checkpoint: {args.ckpt} —— 跳过 DP 评估"
              f"（请先运行 diffusion_policy_train/train.py）")
        return {"skipped": True, "reason": "checkpoint not found", "ckpt": args.ckpt}

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f">> [policy] 设备: {device}  ckpt: {args.ckpt}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in ck["stats"].items()}

    model = DiffusionPolicy(
        action_dim=cfg["action_dim"], agent_dim=cfg["agent_dim"],
        n_obs_steps=cfg["n_obs_steps"], horizon=cfg["horizon"],
        num_diffusion_steps=cfg["num_diffusion_steps"], device=device,
    ).to(device)
    model.scheduler = type(model.scheduler)(
        num_steps=cfg["num_diffusion_steps"], device=device)
    # 优先用 EMA 权重（推理质量更好）
    model.load_state_dict(ck.get("ema_model", ck["model"]))
    model.eval()

    n_obs = cfg["n_obs_steps"]
    horizon = cfg["horizon"]
    exec_h = min(args.action_horizon, horizon)

    if args.fixed_object is not None:
        obj_names = [args.fixed_object]
    else:
        specs = dp_objects.get_registry(args.object_group)
        obj_names = [s.name for s in specs]

    records = []
    per_obj = {}
    render_zero_warn = False
    t0 = time.time()
    global_seed = args.seed

    def to_obs_tensor(obs_buf):
        """obs_buf: list[dict] 末 n_obs 帧 -> (1,To,...) 张量。"""
        fr = np.stack([o["front_rgb"] for o in obs_buf], 0).astype(np.float32) / 255.0
        wr = np.stack([o["wrist_rgb"] for o in obs_buf], 0).astype(np.float32) / 255.0
        fr = np.transpose(fr, (0, 3, 1, 2))[None]   # (1,To,3,H,W)
        wr = np.transpose(wr, (0, 3, 1, 2))[None]
        ag = np.stack([o["agent_pos"] for o in obs_buf], 0).astype(np.float32)
        ag = normalize(ag, stats["agent_lo"], stats["agent_hi"])[None]  # (1,To,D)
        return (torch.from_numpy(fr).to(device),
                torch.from_numpy(wr).to(device),
                torch.from_numpy(ag).to(device))

    for ti, obj_name in enumerate(obj_names):
        print(f"\n=== [DP] 物体 [{ti+1}/{len(obj_names)}] {obj_name}："
              f"{args.episodes_per_object} 次试验 ===")
        env = FrankaDPEnv(
            img_size=args.img_size, fixed_object=obj_name,
            show_viewer=False, record_cam=False, seed=global_seed,
            render_enabled=True,   # DP 需要图像
        )
        env.build()
        per_obj.setdefault(obj_name, [0, 0])

        for k in range(args.episodes_per_object):
            ep_seed = global_seed + 50000 + k
            obs = env.reset(seed=ep_seed)
            obs_buf = [obs] * n_obs    # 首帧重复填满观测窗
            success = False
            steps = 0
            while steps < args.max_steps:
                fr, wr, ag = to_obs_tensor(obs_buf[-n_obs:])
                if float(fr.abs().sum()) == 0.0:
                    render_zero_warn = True
                with torch.no_grad():
                    a_norm = model.predict_action(fr, wr, ag)[0].cpu().numpy()
                a_real = unnormalize(a_norm, stats["action_lo"], stats["action_hi"])
                for j in range(exec_h):
                    if steps >= args.max_steps:
                        break
                    obs, info = env.step(a_real[j].astype(np.float32))
                    obs_buf.append(obs)
                    steps += 1
                    if info["success"]:
                        success = True
                        break
                if success:
                    break

            per_obj[obj_name][1] += 1
            if success:
                per_obj[obj_name][0] += 1
            post = symbolic_state(env)
            records.append({
                "object": obj_name, "seed": ep_seed,
                "success": bool(success), "steps": int(steps),
                "post": post,
            })
            mark = "✓" if success else "✗"
            print(f"  [{mark}] {obj_name} seed={ep_seed} steps={steps} "
                  f"obj_z={post['obj_z']:.3f} lifted={post['lifted']}")

        global_seed += 1
        del env

    n = len(records)
    n_ok = sum(1 for r in records if r["success"])
    per_obj_rate = {o: round(c / t, 3) if t else 0.0 for o, (c, t) in per_obj.items()}

    result = {
        "task": "experiment2 / low-level policy sampling — DP closed-loop eval",
        "ckpt": args.ckpt,
        "device": device,
        "config": cfg,
        "action_horizon_executed": exec_h,
        "n_trials": n,
        "n_success": n_ok,
        "success_rate": round(n_ok / n, 3) if n else 0.0,
        "per_object_success_rate": per_obj_rate,
        "easy_spawn": bool(args.easy),
        "render_warning": ("camera render returned all-zero images; DP (image-based) "
                           "results are NOT reliable on this machine"
                           if render_zero_warn else None),
        "elapsed_sec": round(time.time() - t0, 1),
        "records": records,
    }
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"mode": args.mode}

    if args.mode in ("consistency", "both"):
        cons = run_consistency(args)
        with open(os.path.join(args.out_dir, "eval_consistency.json"), "w") as f:
            json.dump(cons, f, indent=2, ensure_ascii=False)
        summary["consistency"] = {
            "consistency_rate": cons["consistency_rate"],
            "per_object": cons["per_object_consistency_rate"],
            "n_trials": cons["n_trials"],
        }
        print(f"\n[实验2 一致性] 整体一致率 {cons['consistency_rate']} "
              f"({cons['n_consistent_with_h']}/{cons['n_trials']})  "
              f"各物体: {cons['per_object_consistency_rate']}")

    if args.mode in ("policy", "both"):
        pol = run_policy(args)
        with open(os.path.join(args.out_dir, "eval_policy.json"), "w") as f:
            json.dump(pol, f, indent=2, ensure_ascii=False)
        if pol.get("skipped"):
            summary["policy"] = {"skipped": True, "reason": pol["reason"]}
            print(f"\n[DP 评估] 跳过：{pol['reason']}")
        else:
            summary["policy"] = {
                "success_rate": pol["success_rate"],
                "per_object": pol["per_object_success_rate"],
                "n_trials": pol["n_trials"],
                "render_warning": pol["render_warning"],
            }
            print(f"\n[DP 评估] 成功率 {pol['success_rate']} "
                  f"({pol['n_success']}/{pol['n_trials']})  "
                  f"各物体: {pol['per_object_success_rate']}")
            if pol["render_warning"]:
                print(f"  [警告] {pol['render_warning']}")

    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已写入: {args.out_dir}")


if __name__ == "__main__":
    main()
