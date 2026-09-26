"""Generate return/home wrappers and run collision-audited full formal BT."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"))
from core.curriculum_bridge import DOCS as PICK_DOCS
from core.held_basin_pour_bridge import DOCS as POUR_DOCS
from core.smooth_held_basin_bridge import DOCS, SmoothHeldBasinRuntime, domain, goal
from formal_bt import ModelLibrary, action_id
from llm_backend import make_llm_backend
from policy_codegen import PolicyProgram, PolicySampler
from probe_curriculum_generated_bt import execute as base_execute


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def by_name(models, name): return next(m for m in models if m["name"] == name)


def execute(runtime, model, program, path):
    rec = base_execute(runtime, model, program, path)
    required = {
        "separate_receiver": ["separate_receiver"],
        "restore_source": ["restore_upright"],
        "return_source": ["return_object"],
        "return_basin": ["return_object"],
        "controlled_home": ["controlled_home"],
    }.get(model["program_kind"], [])
    calls = [x["method"] for x in rec["trace"]]; missing = [x for x in required if x not in calls]
    rec["diagnostics"]["required_transition_calls"] = required
    if missing:
        rec["effect_ok"] = False; rec["diagnostics"]["missing_transition_calls"] = missing
    (path / "feedback.json").write_text(json.dumps(rec, indent=2))
    return rec


def load_existing(base_path, pour_path):
    programs = {}; base = json.loads(base_path.read_text()); pour = json.loads(pour_path.read_text())
    row = next(r for r in base["samples"] if r["kind"] == "pick"); a = next(a for a in reversed(row["attempts"]) if a["accepted"])
    if a.get("source_type") != "backend_generated" or a["sha256"] != hashlib.sha256(a["source"].encode()).hexdigest():
        raise ValueError("pick cache provenance mismatch")
    if row.get("api_docs") != PICK_DOCS["pick"]: raise ValueError("pick cache API mismatch")
    programs["pick"] = PolicyProgram(a["source"], PICK_DOCS["pick"])
    provenance = {"pick": a["sha256"]}
    for row in pour["samples"]:
        a = next(a for a in reversed(row["attempts"]) if a["accepted"])
        if a.get("source_type") != "backend_generated" or a["sha256"] != hashlib.sha256(a["source"].encode()).hexdigest():
            raise ValueError("pour cache provenance mismatch")
        if row.get("api_docs") != POUR_DOCS[row["kind"]]: raise ValueError("pour cache API mismatch")
        programs[row["kind"]] = PolicyProgram(a["source"], POUR_DOCS[row["kind"]]); provenance[row["kind"]] = a["sha256"]
    return programs, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "scenes/held_basin_pour_smooth_v2.json")
    ap.add_argument("--base-programs", type=Path, default=ROOT / "data/programs/curriculum_pick_result.json")
    ap.add_argument("--pour-programs", type=Path, default=ROOT / "data/programs/held_basin_pour_result.json")
    ap.add_argument("--attempts", type=int, default=4); ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--reuse", type=Path)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text()); models = domain(); target = goal()
    paths = [Path(__file__), ROOT / "core/smooth_held_basin_bridge.py", ROOT / "probe_smooth_held_basin_reference.py",
             ROOT / "probe_held_basin_collision.py", ROOT / "core/held_basin_pour_bridge.py", ROOT / "probe_held_basin_pour.py",
             ROOT / "probe_feasible_curriculum.py", ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto/policy_codegen.py"]
    hashes = {str(p): sha(p) for p in paths}
    for p in paths: (args.output / (p.stem + "_source.py")).write_bytes(p.read_bytes())
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    result = {"task": "smooth_held_basin_pour_return_and_controlled_home", "config_path": str(args.config),
              "config_sha256": sha(args.config), "planner": "formal_BT_Expansion_unmodified",
              "model_source": "handwritten_STRIPS", "program_source": "actual_local_model_or_verified_cache",
              "oracle": True, "weld_assisted": True, "object_snap": False, "qpos_write": False,
              "instant_rehome": False, "collision_audited": True, "samples": [], "episodes": [],
              "source_hashes_at_start": hashes, "completed": False}
    programs, prior = load_existing(args.base_programs, args.pour_programs); result["prior_program_hashes"] = prior
    setup_names = ["pick_source_cup", "pick_basin", "present_basin", "pour_into_held_basin"]
    new_order = ["separate_receiver_after_pour", "restore_source_upright", "return_source_cup", "home_source_arm", "return_basin", "home_receiver_arm"]
    try:
        if args.reuse:
            old = json.loads(args.reuse.read_text())
            if old["config_sha256"] != result["config_sha256"] or not old["completed"]:
                raise ValueError("invalid full smooth cache")
            for row in old["samples"]:
                a = next(a for a in reversed(row["attempts"]) if a["accepted"])
                if a.get("source_type") != "backend_generated": raise ValueError("cache lacks backend-generated provenance")
                if a["sha256"] != hashlib.sha256(a["source"].encode()).hexdigest(): raise ValueError("cache hash mismatch")
                if row.get("api_docs") != DOCS[row["kind"]]: raise ValueError("cache API contract mismatch")
                programs[row["kind"]] = PolicyProgram(a["source"], DOCS[row["kind"]]); result["samples"].append(row)
            result["reused_from"] = str(args.reuse)
        else:
            backend = make_llm_backend("local")
            specs = [("separate_receiver", "separate_receiver_after_pour"),
                     ("restore_source", "restore_source_upright"), ("return_source", "return_source_cup"),
                     ("controlled_home", "home_source_arm"), ("return_basin", "return_basin")]
            for kind, action_name in specs:
                trial = [0]
                def evaluate(program, kind=kind, action_name=action_name):
                    trial[0] += 1; path = args.output / "sampling" / kind / f"trial{trial[0]}"
                    runtime = SmoothHeldBasinRuntime(config, path, 0)
                    try:
                        setup = list(setup_names)
                        for name in new_order:
                            if name == action_name: break
                            setup.append(name)
                        for i, name in enumerate(setup):
                            m = by_name(models, name); prog = programs[m["program_kind"]]
                            rec = execute(runtime, m, prog, path / f"setup{i}_{name}")
                            if not rec["effect_ok"]:
                                return {"valid_trial": False, "code_executed": False, "goal_ok": False,
                                        "effect_ok": False, "diagnostics": {"setup_failed": rec}}
                        rec = execute(runtime, by_name(models, action_name), program, path / "candidate")
                        stable = runtime.scout.stable(lambda: set(by_name(models, action_name)["add"]) <= runtime.state())
                        rec["effect_ok"] = bool(rec["effect_ok"] and stable and runtime.collision_free())
                        rec["goal_ok"] = bool(rec["goal_ok"] and stable)
                        rec["diagnostics"].update(stable_effect=stable, collision_free=runtime.collision_free(),
                                                  collision_groups=runtime.audit.summary())
                        print("SMOOTH_SAMPLE", kind, trial[0], rec["effect_ok"], rec.get("exception"), flush=True)
                        return rec
                    finally:
                        try:
                            runtime.save(path)
                        finally:
                            runtime.close()
                row = PolicySampler(backend, focused_repair=True).synthesize_and_test(
                    by_name(models, action_name), DOCS[kind], evaluate, args.attempts, args.output / "sampling" / kind)
                row["kind"] = kind; row["api_docs"] = DOCS[kind]; result["samples"].append(row)
                if not row["success"]: result["blocked_on"] = kind; return
                accepted = next(a for a in reversed(row["attempts"]) if a["accepted"])
                programs[kind] = PolicyProgram(accepted["source"], DOCS[kind])

        result["program_hashes"] = {k: hashlib.sha256(v.source.encode()).hexdigest() for k, v in programs.items()}
        for seed in range(args.episodes):
            path = args.output / f"seed{seed}"; path.mkdir(parents=True, exist_ok=True)
            runtime = SmoothHeldBasinRuntime(config, path, seed)
            ep = {"seed": seed, "steps": [], "success": False, "goal": sorted(target)}; result["episodes"].append(ep)
            try:
                before = runtime.state(); ep.update(initial_state=sorted(before), scene_sha256=runtime.scout.result["scene_sha256"],
                                                    initial_penetrations=runtime.scout.result["initial_penetrations"])
                if target <= before: raise ValueError("initial task already solved")
                bt = ModelLibrary(models).build(before, target); ep["kernel"] = bt.kernel
                (path / "tree.json").write_text(bt.export()); (path / "tree.dot").write_text(bt.export("dot"))
                for step in range(20):
                    status, m = bt.tick(runtime.state())
                    if status != "running": break
                    rec = execute(runtime, m, programs[m["program_kind"]], path / f"step{step}"); ep["steps"].append(rec)
                    print("SMOOTH_BT", seed, step, action_id(m), rec["effect_ok"], flush=True)
                    if not rec["effect_ok"]: break
                stable = runtime.stable_goal(); status, _ = bt.tick(runtime.state())
                ep.update(bt_status=status, stable=stable, success=bool(stable and status == "success" and all(x["effect_ok"] for x in ep["steps"])),
                          final_state=sorted(runtime.state()), return_geometry={o: runtime.return_geometry(o) for o in ("canL", "basin")},
                          collision_free=runtime.collision_free(), collision_groups=runtime.audit.summary(),
                          sphere_local=runtime.scout.basin_local(runtime.scout.pos("ballL")).tolist(),
                          return_events=runtime.return_events, pour_events=runtime.pour_events, assistance_events=runtime.scout.events)
            except Exception as exc:
                ep["exception"] = f"{type(exc).__name__}: {exc}"; print("SMOOTH_FAIL", seed, ep["exception"], flush=True)
            finally:
                try:
                    runtime.save(path)
                finally:
                    runtime.close()
                (path / "result.json").write_text(json.dumps(ep, indent=2))
                print("SMOOTH_FINAL", seed, ep["success"], flush=True)
        result["success_count"] = sum(e["success"] for e in result["episodes"])
        result["completed"] = len(result["episodes"]) == args.episodes and all(e["success"] for e in result["episodes"])
    finally:
        result["config_unchanged"] = sha(args.config) == result["config_sha256"]
        result["source_hashes_at_end"] = {str(p): sha(p) for p in paths}
        result["implementation_unchanged_during_run"] = hashes == result["source_hashes_at_end"]
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__": main()
