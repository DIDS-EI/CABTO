"""Generate receiver-present and coordinated-pour programs, then run formal BT."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"))
from core.curriculum_bridge import DOCS as BASE_DOCS
from core.held_basin_pour_bridge import HeldBasinRuntime, DOCS, domain, goal
from formal_bt import ModelLibrary, action_id
from llm_backend import make_llm_backend
from policy_codegen import PolicyProgram, PolicySampler
from probe_curriculum_generated_bt import execute as base_execute


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def by_name(models, name): return next(m for m in models if m["name"] == name)


def execute(runtime, model, program, path):
    rec = base_execute(runtime, model, program, path)
    required = {"present_receiver": ["move_basin"],
                "coordinated_pour": ["align_receiver", "move_source", "tip_source"]}.get(model["program_kind"], [])
    calls = [x["method"] for x in rec["trace"]]
    missing = [x for x in required if x not in calls]
    if missing:
        rec["effect_ok"] = False; rec["diagnostics"]["missing_transition_calls"] = missing
    rec["diagnostics"]["required_transition_calls"] = required
    (path / "feedback.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "scenes/held_basin_pour_v1.json")
    ap.add_argument("--programs", type=Path, default=ROOT / "outputs/cabto_curriculum_codegen/final_recheck/result.json")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--reuse", type=Path)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text()); models = domain(); target = goal()
    paths = [Path(__file__), ROOT / "core/held_basin_pour_bridge.py", ROOT / "probe_held_basin_pour.py",
             ROOT / "probe_feasible_curriculum.py", ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto/policy_codegen.py"]
    hashes = {str(p): sha(p) for p in paths}
    for p in paths: (args.output / (p.stem + "_source.py")).write_bytes(p.read_bytes())
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    result = {"task": "one_arm_pours_sphere_other_arm_holds_basin", "config_path": str(args.config),
              "config_sha256": sha(args.config), "model_source": "handwritten_STRIPS",
              "planner": "formal_BT_Expansion_unmodified", "program_source": "actual_local_model_or_verified_cache",
              "oracle": True, "weld_assisted": True, "object_snap": False, "arm_rehome": False,
              "samples": [], "episodes": [], "source_hashes_at_start": hashes, "completed": False}
    programs = {}
    try:
        old = json.loads(args.programs.read_text())
        pick_row = next(r for r in old["samples"] if r["kind"] == "pick")
        picked = next(a for a in reversed(pick_row["attempts"]) if a["accepted"])
        if picked.get("source_type") != "backend_generated" or hashlib.sha256(picked["source"].encode()).hexdigest() != picked["sha256"]:
            raise ValueError("pick cache provenance mismatch")
        if pick_row["api_docs"] != BASE_DOCS["pick"]: raise ValueError("pick API mismatch")
        programs["pick"] = PolicyProgram(picked["source"], BASE_DOCS["pick"])
        result["pick_reused_from"] = str(args.programs); result["pick_hash"] = picked["sha256"]

        if args.reuse:
            previous = json.loads(args.reuse.read_text())
            if not previous["completed"] or previous["config_sha256"] != result["config_sha256"]:
                raise ValueError("invalid held-basin program cache")
            for row in previous["samples"]:
                accepted = next(a for a in reversed(row["attempts"]) if a["accepted"])
                if accepted["sha256"] != hashlib.sha256(accepted["source"].encode()).hexdigest():
                    raise ValueError("cache source hash mismatch")
                programs[row["kind"]] = PolicyProgram(accepted["source"], DOCS[row["kind"]])
                result["samples"].append(row)
            result["reused_from"] = str(args.reuse)
        else:
            backend = make_llm_backend("local")
            for kind, name in (("present_receiver", "present_basin"), ("coordinated_pour", "pour_into_held_basin")):
                count = [0]
                def evaluate(program, kind=kind, name=name):
                    count[0] += 1; path = args.output / "sampling" / kind / f"trial{count[0]}"
                    runtime = HeldBasinRuntime(config, path, 0)
                    try:
                        setup = [by_name(models, "pick_basin")]
                        if kind == "coordinated_pour":
                            setup += [by_name(models, "present_basin"), by_name(models, "pick_source_cup")]
                        for i, m in enumerate(setup):
                            prog = programs[m["program_kind"]]
                            rec = execute(runtime, m, prog, path / f"setup{i}_{m['name']}")
                            if not rec["effect_ok"]:
                                return {"valid_trial": False, "code_executed": False, "goal_ok": False,
                                        "effect_ok": False, "diagnostics": {"setup_failed": rec}}
                        rec = execute(runtime, by_name(models, name), program, path / "candidate")
                        if kind == "coordinated_pour":
                            stable = runtime.scout.stable(lambda: target <= runtime.state())
                            rec["effect_ok"] = bool(rec["effect_ok"] and stable)
                            rec["goal_ok"] = bool(rec["goal_ok"] and stable)
                            rec["diagnostics"]["stable_caught_in_held_basin"] = stable
                        print("HELD_BASIN_SAMPLE", kind, count[0], rec["effect_ok"], rec.get("exception"), flush=True)
                        return rec
                    finally:
                        runtime.save(path); runtime.close()
                row = PolicySampler(backend, focused_repair=True).synthesize_and_test(
                    by_name(models, name), DOCS[kind], evaluate, args.attempts, args.output / "sampling" / kind)
                row["kind"] = kind; row["api_docs"] = DOCS[kind]; result["samples"].append(row)
                if not row["success"]: result["blocked_on"] = kind; return
                accepted = next(a for a in reversed(row["attempts"]) if a["accepted"])
                programs[kind] = PolicyProgram(accepted["source"], DOCS[kind])

        result["program_hashes"] = {k: hashlib.sha256(v.source.encode()).hexdigest() for k, v in programs.items()}
        for seed in range(args.episodes):
            path = args.output / f"seed{seed}"; path.mkdir(parents=True, exist_ok=True)
            runtime = HeldBasinRuntime(config, path, seed)
            ep = {"seed": seed, "steps": [], "success": False, "goal": sorted(target)}
            result["episodes"].append(ep)
            try:
                before = runtime.state(); ep["initial_state"] = sorted(before)
                ep["scene_sha256"] = runtime.scout.result["scene_sha256"]
                ep["initial_penetrations"] = runtime.scout.result["initial_penetrations"]
                if target <= before: raise ValueError("initial task already solved")
                bt = ModelLibrary(models).build(before, target)
                ep["kernel"] = bt.kernel
                (path / "tree.json").write_text(bt.export()); (path / "tree.dot").write_text(bt.export("dot"))
                for step in range(12):
                    status, m = bt.tick(runtime.state())
                    if status != "running": break
                    rec = execute(runtime, m, programs[m["program_kind"]], path / f"step{step}")
                    ep["steps"].append(rec)
                    print("HELD_BASIN_BT", seed, step, action_id(m), rec["effect_ok"], flush=True)
                    if not rec["effect_ok"]: break
                stable = runtime.stable_goal(); status, _ = bt.tick(runtime.state())
                ep.update(bt_status=status, stable=stable,
                          success=bool(stable and status == "success" and all(r["effect_ok"] for r in ep["steps"])),
                          final_state=sorted(runtime.state()), final_sphere_position=runtime.scout.pos("ballL").tolist(),
                          final_sphere_local_in_basin=runtime.scout.basin_local(runtime.scout.pos("ballL")).tolist(),
                          final_basin_center=runtime.scout.basin_center().tolist(), pour_events=runtime.pour_events,
                          assistance_events=runtime.scout.events)
            except Exception as exc:
                ep["exception"] = f"{type(exc).__name__}: {exc}"; print("HELD_BASIN_FAIL", seed, ep["exception"], flush=True)
            finally:
                runtime.save(path); runtime.close(); (path / "result.json").write_text(json.dumps(ep, indent=2))
                print("HELD_BASIN_FINAL", seed, ep["success"], flush=True)
        result["success_count"] = sum(e["success"] for e in result["episodes"])
        result["completed"] = len(result["episodes"]) == args.episodes and all(e["success"] for e in result["episodes"])
    finally:
        result["config_unchanged"] = sha(args.config) == result["config_sha256"]
        result["source_hashes_at_end"] = {str(p): sha(p) for p in paths}
        result["implementation_unchanged_during_run"] = result["source_hashes_at_end"] == hashes
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__": main()
