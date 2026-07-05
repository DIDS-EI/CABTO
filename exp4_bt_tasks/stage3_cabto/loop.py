"""E. ⑤ 主闭环（self-correction）
==================================
run_closed_loop(task, objects, max_rounds, backend, render):
  建 env / llm+vlm backend(单例) / planner / pointer / runner / checker。
  round 循环：
    1. planner.plan(task, objects, feedback) 得 plan。
    2. 顺序执行每步：执行前截 before(overview)，execute_step 执行，执行后截 after。
    3. EffectChecker.check 校验 add/del（gt 真值路 + VLM 视觉路）。
       某步效应不满足(gt_ok=False) → 记失败原因，组装 feedback，break，进下一 round。
    4. 全部步骤满足 → success，停止。或达 max_rounds 停止。
  产物存 stage3_out/：rollout.mp4、step{i}_before/after.png、result.json。

失败判定权威 = ground-truth(gt_ok)，VLM 校验作对比记录（VLM 不可靠不应误判
oracle 闭环）。self-correction 在 gt_ok=False 时触发重规划。
"""

from __future__ import annotations

import os
import json

import numpy as np
import imageio

from blocks_env import BlocksEnv
from llm_backend import make_llm_backend, make_vlm_backend
from planner import Planner
from vlm_pointer3d import Pointer
from primitives import PrimitiveRunner
from codegen import execute_step, render_program_text
from effect_checker import EffectChecker
import world_state as ws

OUT_DIR = "/tmp/exp4_stage3/stage3_out"


def _build_feedback(step, gt_violations, vlm_violations):
    name = step.get("name")
    args = step.get("args", {})
    lines = [f"Step '{name}' with args {args} did not achieve its intended effect."]
    if gt_violations:
        lines.append("Ground-truth violations: " + "; ".join(gt_violations))
    if vlm_violations:
        lines.append("Visual(VLM) violations: " + "; ".join(vlm_violations))
    lines.append("Re-plan to achieve the goal, ensuring preconditions hold.")
    return "\n".join(lines)


def run_closed_loop(task, objects, max_rounds=3, backend="qwen", render=True,
                    inject_failure=False):
    """inject_failure: 调试用，人为破坏第一轮放置（验证 self-correction 触发）。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n===== run_closed_loop backend={backend} task='{task}' =====")

    env = BlocksEnv(render=render, img_size=512)
    llm = make_llm_backend("local")          # 规划用（本地 Qwen，单例）
    if backend == "oracle":
        vlm = None                            # oracle 不需要 VLM 打点
        use_vlm_check = False
    else:
        vlm = make_vlm_backend("local")       # 与 llm 共用单例实例
        use_vlm_check = True

    planner = Planner(llm)
    pointer = Pointer(vlm, env, backend=("oracle" if backend == "oracle" else "qwen"))
    checker = EffectChecker(vlm) if vlm is not None else EffectChecker(None)

    env.reset()
    env.start_record()
    env.record_frame("overview")
    runner = PrimitiveRunner(env, recorder=lambda: env.record_frame("overview"))

    oracle_refine = (backend == "oracle")     # oracle 路允许用真值精修放置 xy

    result = {"task": task, "objects": objects, "backend": backend,
              "max_rounds": max_rounds, "rounds": [], "success": False,
              "rounds_used": 0}

    feedback = None
    success = False
    step_img_idx = 0

    for rnd in range(max_rounds):
        plan, meta = planner.plan(task, objects, feedback=feedback)
        rnd_rec = {"round": rnd, "plan_source": meta.get("source"),
                   "plan": plan, "steps": []}
        print(f"\n--- round {rnd} | plan_source={meta.get('source')} | "
              f"{len(plan)} steps ---")
        for st in plan:
            print("   ", render_program_text(st).splitlines()[0])

        round_ok = True
        for si, step in enumerate(plan):
            state_before = ws.compute_state(env)
            before_img = env.render_cam("overview")

            # 执行该步（codegen 串接原语）
            exec_log = execute_step(env, runner, pointer, step,
                                    oracle_refine=oracle_refine)

            # 调试：人为制造一次放置失败（把绿块挪开），验证 self-correction
            if inject_failure and step["name"] == "place_on" and rnd == 0:
                b = env.obj_qadr[step["args"]["obj"]]
                env.d.qpos[b:b + 2] += 0.15      # 平移出对齐范围
                import mujoco
                mujoco.mj_forward(env.m, env.d)

            after_img = env.render_cam("overview")

            # 校验效应（gt 真值 + VLM 视觉）
            chk = checker.check(before_img, after_img, step, env=env,
                                state_before=state_before, use_vlm=use_vlm_check)

            # 存 before/after 图
            bpath = os.path.join(OUT_DIR, f"step{step_img_idx}_before.png")
            apath = os.path.join(OUT_DIR, f"step{step_img_idx}_after.png")
            imageio.imwrite(bpath, before_img)
            imageio.imwrite(apath, after_img)

            step_rec = {
                "idx": step_img_idx, "name": step["name"], "args": step["args"],
                "program_text": render_program_text(step),
                "exec_log": exec_log,
                "gt_ok": chk["gt_ok"], "gt_violations": chk["gt_violations"],
                "vlm_ok": chk["vlm_ok"], "vlm_violations": chk["vlm_violations"],
                "vlm_raw": chk["vlm_raw"], "agree": chk["agree"],
                "before_img": bpath, "after_img": apath,
            }
            rnd_rec["steps"].append(step_rec)
            step_img_idx += 1

            print(f"   [step {si}] {step['name']}{tuple(step['args'].values())} "
                  f"gt_ok={chk['gt_ok']} vlm_ok={chk['vlm_ok']} agree={chk['agree']}")
            if chk["gt_violations"]:
                print("        gt_violations:", chk["gt_violations"])
            if chk["vlm_violations"]:
                print("        vlm_violations:", chk["vlm_violations"])

            # 失败判定权威 = gt_ok。失败则组装 feedback，break 重规划。
            if chk["gt_ok"] is False:
                feedback = _build_feedback(step, chk["gt_violations"],
                                           chk["vlm_violations"])
                round_ok = False
                print(f"   !! 该步效应不满足，触发 self-correction，进入下一轮")
                break

        result["rounds"].append(rnd_rec)
        result["rounds_used"] = rnd + 1
        if round_ok:
            success = True
            print(f"--- round {rnd} 全部步骤效应满足 ===> SUCCESS ---")
            break
        else:
            # 失败后 reset 环境重试（self-correction 重做整条 plan）
            if rnd + 1 < max_rounds:
                env.reset()
                runner = PrimitiveRunner(env,
                                         recorder=lambda: env.record_frame("overview"))

    result["success"] = success

    # ---- 最终真值快照 ----
    final_state = ws.compute_state(env)
    result["final_stacked_on_green_yellow"] = bool(
        final_state.get("stacked_on(green_block,yellow_block)", False))
    result["final_true_predicates"] = sorted(k for k, v in final_state.items() if v)

    # ---- 存产物 ----
    env.save_video(os.path.join(OUT_DIR, "rollout.mp4"), fps=30)
    rpath = os.path.join(OUT_DIR, "result.json")
    with open(rpath, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[ok] result.json -> {rpath}")
    print(f"[result] success={success} rounds_used={result['rounds_used']} "
          f"stacked_on(green,yellow)={result['final_stacked_on_green_yellow']}")
    return result


if __name__ == "__main__":
    import sys
    bk = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    run_closed_loop("stack the green block on the yellow block",
                    ["green_block", "yellow_block", "blue_block"],
                    max_rounds=3, backend=bk, render=True)
