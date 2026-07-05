"""端到端评估代码生成 pick（oracle / qwen 后端）
==================================================

把 vlm_pointer + camera + primitives 通过 codegen_action.pick 串起来，
在多物体 × 多布局上评估抓取成功率，并记录 VLM 打点像素误差（相对 oracle 真值）。

跑法（cabto 环境）：
  PYTHONPATH=. <python> exp2_codegen/eval_pick.py --backend oracle --seeds 0-9
  PYTHONPATH=. <python> exp2_codegen/eval_pick.py --backend qwen  --seeds 0-3
"""
from __future__ import annotations

import os, sys, json, argparse, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from dp_common.franka_dp_env import FrankaDPEnv, TABLE_TOP_Z
from exp2_codegen import camera as C
from exp2_codegen import codegen_action as CG
from exp2_codegen.vlm_pointer import make_pointer

IMG = 512
W = H = IMG
OBJECTS = ["dice_cube", "wood_block", "soda_can", "tea_tin", "rubber_ball"]


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def px_err_vs_oracle(env, pts):
    """pts: {cam:[u,v]}。返回各相机相对 oracle 真值投影的像素误差。"""
    opos, _ = env.get_object_pose()
    top = opos.copy(); top[2] = TABLE_TOP_Z + env._cur_top
    errs = {}
    for cam, uv in pts.items():
        if uv is None:
            errs[cam] = None; continue
        gt = C.project_point(env.m, env.d, W, H, cam, top)
        errs[cam] = None if gt is None else float(np.hypot(uv[0]-gt[0], uv[1]-gt[1]))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="oracle", choices=["oracle", "qwen"])
    ap.add_argument("--seeds", default="0-5")
    ap.add_argument("--objects", nargs="*", default=OBJECTS)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    # qwen：只加载一次模型，跨物体/布局复用
    shared_qwen = None
    if args.backend == "qwen":
        from exp2_codegen.vlm_pointer import QwenVLPointer
        print("[load] Qwen2.5-VL ...", flush=True)
        shared_qwen = QwenVLPointer()
        instruction = "the object on the table"
    else:
        instruction = "the top center of the object on the table to grasp"

    results = []
    n_succ = 0; n_total = 0
    all_px = []
    for obj in args.objects:
        env = FrankaDPEnv(img_size=IMG, fixed_object=obj,
                          render_enabled=True, seed=0)
        env.build()
        dep = mujoco.Renderer(env.m, IMG, IMG)
        dep.enable_depth_rendering()
        for seed in seeds:
            env.reset(seed=seed)
            if args.backend == "oracle":
                pointer = make_pointer("oracle",
                                       project_fn=CG._make_oracle_project_fn(env, (W, H)))
            else:
                pointer = shared_qwen
            rec = CG.pick(env, pointer, dep, (W, H),
                          point_cam="overview", verify_cam="front",
                          instruction=instruction)
            n_total += 1
            ok = bool(rec.get("success"))
            n_succ += int(ok)
            errs = px_err_vs_oracle(env, rec.get("target_px", {}))
            for v in errs.values():
                if v is not None:
                    all_px.append(v)
            results.append({"obj": obj, "seed": seed, "success": ok,
                            "px_err": {k: (None if v is None else round(v,1)) for k,v in errs.items()},
                            "target_px": rec.get("target_px"),
                            "triangulated": rec.get("triangulated"),
                            "cross_view_disagreement": rec.get("cross_view_disagreement"),
                            "error": rec.get("error")})
            print(f"  {obj:12s} seed{seed} success={ok} px_err={ {k:(None if v is None else round(v,1)) for k,v in errs.items()} }",
                  flush=True)
        dep.close(); del env, dep; gc.collect()

    summary = {"backend": args.backend, "n_succ": n_succ, "n_total": n_total,
               "success_rate": round(n_succ / max(n_total,1), 3)}
    if all_px:
        summary["px_err_median"] = round(float(np.median(all_px)), 1)
        summary["px_err_mean"] = round(float(np.mean(all_px)), 1)
    print(f"\n==== {args.backend}: 成功率 {n_succ}/{n_total} = {summary['success_rate']:.0%}"
          + (f"，打点像素误差 median={summary.get('px_err_median')} mean={summary.get('px_err_mean')}" if all_px else "")
          + " ====")

    op = os.path.join(os.path.dirname(__file__), "results", f"eval_pick_{args.backend}.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"[saved] {op}")


if __name__ == "__main__":
    main()
