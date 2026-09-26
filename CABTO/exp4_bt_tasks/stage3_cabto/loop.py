"""执行手写模板，检查效应及最终目标，失败反馈用于现有 Planner 重规划。

policy_feedback 只落盘供审查，当前模板不会自动消费；未实现策略生成/修复闭环。
backend='oracle' 指感知真值路径（及既有放置精修），并非全系统 oracle。
BlocksEnv 使用 weld 抓取辅助，success 仅表示 assisted 模式下目标真值通过。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import os
import json
import re
import traceback

import imageio

from blocks_env import BlocksEnv
from llm_backend import make_llm_backend, make_vlm_backend
from planner import Planner
from vlm_pointer3d import Pointer
from primitives import PrimitiveRunner
from codegen import execute_step, gen_pick, gen_place_on
from effect_checker import EffectChecker
from feedback import build_failure_feedback
import world_state as ws

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage3_out")


def _resolve_goals(task, objects, goal_predicates):
    if goal_predicates is None:
        # 只接受完整、明确的单目标句；否定、附加任务和未知任务必须显式给目标。
        text = " ".join(task.lower().strip().rstrip(".").split())
        pattern = (r"(?:stack|place|put) (?:the )?green(?: block|_block)? "
                   r"(?:on|onto|on top of) (?:the )?yellow(?: block|_block)?")
        if (re.fullmatch(pattern, text)
                and {"green_block", "yellow_block"}.issubset(objects)):
            return ["stacked_on(green_block,yellow_block)"]
        raise ValueError("此任务不能安全推断默认目标，请显式提供 goal_predicates。")
    if (not isinstance(goal_predicates, (list, tuple)) or not goal_predicates
            or any(not isinstance(p, str) or not p.strip() for p in goal_predicates)):
        raise ValueError("goal_predicates 必须是非空谓词字符串列表，按合取真值验收。")
    return list(goal_predicates)


def _program_provenance(step):
    functions = [execute_step]
    template = {"pick": gen_pick, "place_on": gen_place_on}.get(step.get("name"))
    if template is not None:
        functions.append(template)
    sources, origins = [], []
    for function in functions:
        source = inspect.getsource(function)
        sources.append(source)
        origins.append({
            "module": function.__module__, "function": function.__qualname__,
            "file": inspect.getsourcefile(function),
            "line": inspect.getsourcelines(function)[1],
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        })
    text = "\n\n".join(sources)
    return text, {"source_type": "handwritten_template", "functions": origins,
                  "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


class _PointerTrace:
    """记录本步实际定位调用，不改变 Pointer 的算法或回退行为。"""

    def __init__(self, pointer):
        self.pointer = pointer
        self.calls = []

    def locate_3d(self, *args, **kwargs):
        call = {"args": list(args), "kwargs": dict(kwargs),
                "backend": self.pointer.backend, "source": "unknown", "returned": False}
        self.calls.append(call)
        point = self.pointer.locate_3d(*args, **kwargs)
        call.update(returned=True, located_3d=[float(x) for x in point],
                    source=("oracle" if self.pointer.backend == "oracle"
                            else getattr(self.pointer, "_last_src", "unknown")))
        if self.pointer.backend != "oracle":
            call["last_check"] = deepcopy(getattr(self.pointer, "last_check", None))
        return point

    def __getattr__(self, name):
        return getattr(self.pointer, name)


def _save_feedback(out_dir, label, record, failure):
    if "program" in record:
        failure["observed_facts"]["program"] = record["program"]
    record["feedback"] = failure
    record["feedback_path"] = os.path.join(out_dir, f"{label}_feedback.json")
    record["policy_feedback_path"] = os.path.join(out_dir, f"{label}_policy_feedback.txt")
    with open(record["feedback_path"], "w", encoding="utf-8") as f:
        json.dump(failure, f, indent=2, ensure_ascii=False)
    with open(record["policy_feedback_path"], "w", encoding="utf-8") as f:
        f.write(failure["policy_feedback"])
    facts = failure["observed_facts"]
    brief_facts = {k: facts[k] for k in
                   ("step", "preconditions", "effects", "exec_log", "gt_violations",
                    "pointer_provenance", "program_sha256")}
    brief_facts["exception"] = ({k: v for k, v in facts["exception"].items()
                                  if k != "traceback"} if facts["exception"] else None)
    brief = {"diagnosis_status": failure["diagnosis_status"],
             "observed_facts": brief_facts,
             "hypotheses": [{"id": h["id"], "status": h["status"]}
                            for h in failure["hypotheses"]],
             "next_tests": failure["next_tests"], "routes": failure["routes"],
             "candidate_model_patch": failure["candidate_model_patch"],
             "policy_feedback_consumed": False}
    record["planner_feedback"] = json.dumps(brief, ensure_ascii=False, separators=(",", ":"))
    return record["planner_feedback"]


def run_closed_loop(task, objects, max_rounds=3, backend="qwen", render=True,
                    inject_failure=False, *, output_dir=None, planner_backend="local",
                    goal_predicates=None):
    """保留旧位置参数；goal_predicates 是必须全部为真的非空谓词列表。

    planner_backend='rule' 使用 Planner(None)，不加载规划 LLM；若感知仍选 qwen，
    仍需 VLM。完全无模型调用需同时选 backend='oracle'。
    inject_failure 保留原有调试行为。空计划不作为执行成功。
    """
    goals = _resolve_goals(task, objects, goal_predicates)
    if planner_backend not in ("local", "rule"):
        raise ValueError("planner_backend 只支持 local 或 rule。")
    out_dir = os.fspath(output_dir) if output_dir is not None else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== 感知 backend={backend} | planner={planner_backend} | task='{task}' =====")

    env = BlocksEnv(render=render, img_size=512)
    llm = make_llm_backend("local") if planner_backend == "local" else None
    if backend == "oracle":
        vlm = None
        use_vlm_check = False
    else:
        vlm = make_vlm_backend("local")
        use_vlm_check = True

    planner = Planner(llm)
    pointer = _PointerTrace(Pointer(vlm, env, backend=("oracle" if backend == "oracle" else "qwen")))
    checker = EffectChecker(vlm)
    env.reset()
    env.start_record()
    env.record_frame("overview")
    runner = PrimitiveRunner(env, recorder=lambda: env.record_frame("overview"))
    oracle_refine = (backend == "oracle")

    result = {"task": task, "objects": objects, "backend": backend,
              "backend_scope": "perception", "planner_backend": planner_backend,
              "oracle_refine": oracle_refine, "goal_predicates": goals,
              "execution_mode": "assisted", "assisted": True,
              "assistance": ["BlocksEnv weld grasp assistance"],
              "success_semantics": "assisted 模式下最终目标真值满足，非严格物理成功",
              "policy_feedback_consumed": False,
              "policy_repair_closed_loop": False,
              "max_rounds": max_rounds, "rounds": [], "success": False,
              "steps_passed": 0, "goal_ok": False, "rounds_used": 0}
    feedback = None
    success = False
    step_img_idx = 0

    for rnd in range(max_rounds):
        round_before = deepcopy(ws.compute_state(env))
        plan, meta = planner.plan(task, objects, feedback=feedback)
        rnd_rec = {"round": rnd, "plan_source": meta.get("source"),
                   "plan": plan, "steps": [], "steps_passed": 0}
        print(f"\n--- round {rnd} | plan_source={meta.get('source')} | {len(plan)} steps ---")
        round_ok = bool(plan)
        for si, step in enumerate(plan):
            state_before = deepcopy(ws.compute_state(env))
            before_img = env.render_cam("overview")
            program_text, program = _program_provenance(step)
            pointer.calls = []
            exec_log, exception = None, None
            try:
                exec_log = execute_step(env, runner, pointer, step,
                                        oracle_refine=oracle_refine)
            except Exception as exc:
                # execute_step 未返回时，其局部日志不可得；不伪造已完成原语。
                exception = {"type": type(exc).__name__, "module": type(exc).__module__,
                             "message": str(exc), "traceback": traceback.format_exc()}

            if exception is None and inject_failure and step["name"] == "place_on" and rnd == 0:
                b = env.obj_qadr[step["args"]["obj"]]
                env.d.qpos[b:b + 2] += 0.15
                import mujoco
                mujoco.mj_forward(env.m, env.d)

            after_img = env.render_cam("overview")
            state_after = deepcopy(ws.compute_state(env))
            chk = checker.check(before_img, after_img, step, env=env,
                                state_before=state_before, use_vlm=use_vlm_check)
            bpath = os.path.join(out_dir, f"step{step_img_idx}_before.png")
            apath = os.path.join(out_dir, f"step{step_img_idx}_after.png")
            imageio.imwrite(bpath, before_img)
            imageio.imwrite(apath, after_img)
            provenance = {"backend": pointer.backend, "calls": deepcopy(pointer.calls),
                          "oracle_refine_enabled": oracle_refine and step["name"] == "place_on"}
            step_ok = (exception is None and chk["gt_ok"] is True
                       and not (exec_log or {}).get("error"))
            step_rec = {
                "idx": step_img_idx, "name": step["name"], "args": step.get("args", {}),
                "program_text": program_text, "program": program,
                "exec_log": exec_log, "exception": exception, "step_passed": step_ok,
                "state_before": state_before, "state_after": state_after,
                "pointer_provenance": provenance,
                "gt_ok": chk["gt_ok"], "gt_violations": chk["gt_violations"],
                "vlm_ok": chk["vlm_ok"], "vlm_violations": chk["vlm_violations"],
                "vlm_raw": chk["vlm_raw"], "agree": chk["agree"],
                "before_img": bpath, "after_img": apath,
            }
            rnd_rec["steps"].append(step_rec)
            step_img_idx += 1
            print(f"   [step {si}] {step['name']} passed={step_ok} gt_ok={chk['gt_ok']}")
            if step_ok:
                rnd_rec["steps_passed"] += 1
                result["steps_passed"] += 1
            else:
                failure = build_failure_feedback(
                    step, state_before, state_after, exec_log, exception,
                    chk["gt_violations"], {"before": bpath, "after": apath},
                    program_text, provenance)
                failure["observed_facts"]["vlm_violations"] = chk["vlm_violations"]
                feedback = _save_feedback(out_dir, f"step{step_rec['idx']}", step_rec, failure)
                round_ok = False
                break

        round_state = ws.compute_state(env)
        goal_ok = all(bool(round_state.get(p, False)) for p in goals)
        rnd_rec.update(all_steps_passed=round_ok, goal_ok=goal_ok)
        if not plan or (round_ok and not goal_ok):
            violations = [f"目标未满足: {p}" for p in goals if not round_state.get(p, False)]
            if not plan:
                violations.append("空计划未执行任何步骤，不能验收执行成功")
            failure = build_failure_feedback(
                {"name": "goal_check", "args": {}, "add": goals, "del": []},
                round_before, round_state,
                exec_log={"plan_empty": not bool(plan), "steps_passed": rnd_rec["steps_passed"]},
                gt_violations=violations)
            feedback = _save_feedback(out_dir, f"round{rnd}_goal", rnd_rec, failure)
        result["rounds"].append(rnd_rec)
        result["rounds_used"] = rnd + 1
        if round_ok and goal_ok:
            success = True
            print(f"--- round {rnd} 步骤通过且目标真值满足（assisted） ---")
            break
        if rnd + 1 < max_rounds:
            env.reset()
            runner = PrimitiveRunner(env, recorder=lambda: env.record_frame("overview"))

    final_state = ws.compute_state(env)
    result["goal_ok"] = all(bool(final_state.get(p, False)) for p in goals)
    result["success"] = success and result["goal_ok"]
    result["final_stacked_on_green_yellow"] = bool(
        final_state.get("stacked_on(green_block,yellow_block)", False))
    result["final_true_predicates"] = sorted(k for k, v in final_state.items() if v)
    env.save_video(os.path.join(out_dir, "rollout.mp4"), fps=30)
    rpath = os.path.join(out_dir, "result.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[ok] result.json -> {rpath}")
    print(f"[result] success={result['success']} goal_ok={result['goal_ok']} "
          f"steps_passed={result['steps_passed']} mode=assisted")
    return result


if __name__ == "__main__":
    import sys
    bk = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    run_closed_loop("stack the green block on the yellow block",
                    ["green_block", "yellow_block", "blue_block"],
                    max_rounds=3, backend=bk, render=True)
