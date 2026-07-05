"""⑤ 双臂 CABTO 闭环主程序（handover / pour / storage）
========================================================
把五环串起来跑通一个双臂任务：
  ① dual_planner 生成动作序列（LLM 或规则 fallback）
  ② dual_codegen.render_program_text 生成代码文本 + DualExecutor 执行
  ③ 执行（复用 stage2 ArmSkills 原语，oracle 真值定位）
  ④ dual_effect_checker 每步 gt 校验（VLM 可选记录）
  ⑤ 若某步 gt 失败 → 收集 feedback → 重新规划（self-correction，max_rounds）

用法：
  PYTHONPATH=. python dual_loop.py <task> [out_dir]
  task ∈ {handover, pour, storage}

产出：<out_dir>/rollout.mp4, before.png, after.png, result.json, program.txt
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

import _bridge as B
import dual_planner
import dual_codegen
import dual_world_state as WS
from dual_effect_checker import DualEffectChecker

import scene_pour_simple as sp_simple

sh = B.scene_handover
sp = sp_simple                       # pour 用简化场景
ss = B.scene_storage

# ---- 三个任务的配置 ----
TASKS = {
    "handover": {
        "instruction": "hand the box over from the left arm to the right arm "
                       "and place it on the right side of the table",
        "scene": lambda: sh.build_scene_xml(),
        "objects": [f"box{i}" for i in range(sh.N_BOX)],
        "plan_objects": ["box0"],
        "grasp_z_tol": 0.16,
        "block_half": sh.BOX_HALF[2],
        "goal": ["at_place(box0)"],
    },
    "pour": {
        "instruction": "pour the ball from the left can into the right cup",
        "scene": lambda: sp.build_scene_xml(),
        "objects": ["canL", "ballL"],
        "plan_objects": ["canL", "ballL", "cupR"],
        "grasp_z_tol": 0.09,
        "block_half": sp.CAN_HH,
        "goal": ["in_cup(ballL,cupR)"],
    },
    "storage": {
        "instruction": "put the item into the carton and store the carton on the shelf",
        "scene": lambda: ss.build_scene_xml(),
        "objects": ["item_g1", "item_g2", "carton"],
        "plan_objects": ["item_g1", "carton"],
        "grasp_z_tol": 0.12,
        "block_half": ss.ITEM_HALF[2],
        "goal": ["in_carton(item_g1)", "on_shelf(carton)"],
    },
}


def build_env_and_skills(task_key, render=True):
    cfg = TASKS[task_key]
    env = B.Exp4Env(cfg["scene"](), arms=[("_L", -np.pi / 2), ("_R", np.pi / 2)],
                    obj_names=cfg["objects"], render=render, img_size=560)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]
    aL.grasp_z_tol = cfg["grasp_z_tol"]
    aR.grasp_z_tol = cfg["grasp_z_tol"]
    rec = (lambda: env.record_frame("overview")) if render else (lambda: None)
    skL = B.ArmSkills(env, aL, recorder=(rec if render else None), rec_every=2,
                      block_half=cfg["block_half"])
    skR = B.ArmSkills(env, aR, recorder=(rec if render else None), rec_every=2,
                      block_half=cfg["block_half"])
    return env, skL, skR, rec


def goal_satisfied(env, goal):
    st = WS.compute_state(env)
    return all(st.get(g, False) for g in goal), st


def run_task(task_key, out_dir, render=True, use_vlm=False, max_rounds=3,
             llm_backend=None):
    cfg = TASKS[task_key]
    os.makedirs(out_dir, exist_ok=True)
    WS.set_task(task_key)

    planner = dual_planner.Planner(llm_backend)
    checker = DualEffectChecker(vlm_backend=None)

    feedback = None
    rounds_log = []
    final_success = False
    program_text = ""

    for rnd in range(1, max_rounds + 1):
        WS.set_task(task_key)  # 清 flags
        env, skL, skR, rec = build_env_and_skills(task_key, render=render)
        exe = dual_codegen.DualExecutor(env, skL, skR, recorder=rec)

        # ① 规划
        plan, meta = planner.plan(cfg["instruction"], cfg["plan_objects"],
                                  feedback=feedback)
        program_text = dual_codegen.render_program_text(plan)

        if render:
            env.start_record()
            for _ in range(10):
                rec()
            env.save_png(os.path.join(out_dir, "before.png"), "overview")

        step_logs = []
        plan_failed_step = None
        for i, step in enumerate(plan):
            state_before = WS.compute_state(env)
            try:
                exe.execute_step(step)
            except Exception as e:
                step_logs.append({"step": step["name"], "args": step["args"],
                                  "error": str(e), "gt_ok": False})
                plan_failed_step = (i, step, [f"execution error: {e}"])
                break
            res = checker.check(step, env, state_before=state_before,
                                use_vlm=use_vlm)
            step_logs.append({"step": step["name"], "args": step["args"],
                              "add": step["add"], "gt_ok": res["gt_ok"],
                              "gt_violations": res["gt_violations"]})
            if not res["gt_ok"]:
                plan_failed_step = (i, step, res["gt_violations"])
                break

        # settle & 判定最终目标
        for _ in range(30):
            env.hold_arms(grips={"_L": 1.0, "_R": 1.0})
            rec()
        ok, final_state = goal_satisfied(env, cfg["goal"])

        rounds_log.append({
            "round": rnd, "plan_source": meta.get("source"),
            "n_steps": len(plan), "steps": step_logs,
            "goal": cfg["goal"],
            "goal_state": {g: bool(final_state.get(g, False)) for g in cfg["goal"]},
            "success": ok,
        })

        if render:
            env.save_png(os.path.join(out_dir, "after.png"), "overview")
            env.save_video(os.path.join(out_dir, "rollout.mp4"), fps=30)

        if ok:
            final_success = True
            break

        # ⑤ 失败 → 生成 feedback 重规划
        if plan_failed_step is not None:
            i, step, viols = plan_failed_step
            feedback = (f"Step {i+1} ({step['name']}{step['args']}) did not achieve "
                        f"its effect: {viols}. Goal predicates {cfg['goal']} not all "
                        f"satisfied. Re-plan to achieve the goal.")
        else:
            feedback = (f"Plan finished but goal {cfg['goal']} not satisfied "
                        f"(state={rounds_log[-1]['goal_state']}). Re-plan.")

    result = {
        "task": task_key,
        "instruction": cfg["instruction"],
        "backend": "oracle",
        "success": final_success,
        "rounds": rounds_log,
        "n_rounds": len(rounds_log),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "program.txt"), "w") as f:
        f.write(program_text)

    print(f"\n[{task_key}] success={final_success} rounds={len(rounds_log)} "
          f"goal={result['rounds'][-1]['goal_state']}")
    print("[saved]", out_dir)
    return final_success


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "handover"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/exp4_stage3b_out/{task}"
    render = os.environ.get("RENDER", "1") != "0"
    ok = run_task(task, out, render=render, use_vlm=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
