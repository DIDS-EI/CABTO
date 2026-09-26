"""Generate restricted cooperative programs and run a formal BT for shelf transport."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import xml.etree.ElementTree as ET
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"))
from core.curriculum_bridge import DOCS as BASE_DOCS
from core.dual_lift_shelf_bridge import DualLiftRuntime, DOCS, lift_domain, lift_goal
from formal_bt import ModelLibrary, action_id
from llm_backend import make_llm_backend
from policy_codegen import PolicyProgram, PolicySampler
from probe_curriculum_generated_bt import execute as base_execute
from probe_feasible_curriculum import build_scene


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_config(config):
    """Remove presentation-only fields before transferring cached programs."""
    value = json.loads(json.dumps(config))
    value.pop("version", None)
    value.pop("visual_style", None)
    value.get("packing", {}).pop("label", None)
    value.get("packing", {}).pop("item_visual_styles", None)
    return value


def collision_signature(config):
    """Canonical task-physics signature; ignores materials, sites and cameras."""
    xml, _ = build_scene(config, "packing", 0)
    root = ET.fromstring(xml)
    records=[]
    for body in root.iter("body"):
        bname=body.get("name","")
        records.append(("body",bname,body.get("pos","0 0 0"),body.get("quat","1 0 0 0")))
        for joint in list(body.findall("joint"))+list(body.findall("freejoint")):
            records.append(("joint",bname,joint.tag,tuple(sorted((k,v) for k,v in joint.attrib.items() if k!="name"))))
        for geom in body.findall("geom"):
            if geom.get("contype")=="0" and geom.get("conaffinity")=="0":continue
            attrs=tuple(sorted((k,v) for k,v in geom.attrib.items() if k not in ("name","material","rgba","group")))
            records.append(("geom",bname,attrs))
    equality=root.find("equality")
    if equality is not None:
        for eq in equality:
            records.append(("equality",eq.tag,tuple(sorted((k,v) for k,v in eq.attrib.items() if k!="name"))))
    return records


def timed_build(models, start, goal, seconds=60):
    def expired(*_):
        raise TimeoutError("formal BT expansion exceeded configured wall-clock budget")
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        return ModelLibrary(models).build(start, goal)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def model_by(models, name, arm=None):
    rows = [m for m in models if m["name"] == name]
    if arm is not None:
        rows = [m for m in rows if m["args"]["arm"] == str(arm)]
    if len(rows) != 1:
        raise ValueError(f"expected one model {name}/{arm}, got {len(rows)}")
    return rows[0]


def execute(runtime, model, program, path):
    rec = base_execute(runtime, model, program, path)
    required = {
        "dual_grasp": "attach_both",
        "cooperative_move": "move_tray_to",
        "dual_release": "detach_both",
        "dual_withdraw": "raise_both",
    }.get(model["program_kind"])
    if required:
        calls = [x["method"] for x in rec["trace"]]
        rec["diagnostics"]["required_transition_call"] = required
        rec["diagnostics"]["required_transition_observed"] = required in calls
        if required not in calls:
            rec["effect_ok"] = False
            rec["diagnostics"]["missing_transition_evidence"] = required
        (path / "feedback.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "scenes/packing_dual_lift_shelf_storage_v3.json")
    ap.add_argument("--programs", type=Path, default=ROOT / "data/programs/curriculum_pick_result.json")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--reuse", type=Path, help="reuse accepted cooperative programs from a prior run")
    args = ap.parse_args()
    if min(args.attempts, args.episodes) < 1:
        ap.error("positive budgets required")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    models = lift_domain()
    goal = lift_goal()

    paths = [Path(__file__), ROOT / "core/dual_lift_shelf_bridge.py", ROOT / "probe_dual_lift_shelf.py",
             ROOT / "probe_free_tray_packing.py", ROOT / "probe_feasible_curriculum.py",
             ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto/policy_codegen.py",
             ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto/formal_bt.py"]
    hashes = {str(p): sha(p) for p in paths}
    for p in paths:
        (args.output / (p.stem + "_source.py")).write_bytes(p.read_bytes())
    (args.output / "config.json").write_text(json.dumps(config, indent=2))

    result = {
        "task": "pack_two_items_then_cooperative_dual_arm_tray_lift_to_shelf",
        "config_path": str(args.config), "config_sha256": sha(args.config),
        "model_source": "handwritten_STRIPS", "planner": "formal_BT_Expansion_unmodified",
        "program_source": "actual_local_model_or_verified_cache",
        "perception": "oracle", "grasp": "two_explicit_geometry_checked_welds",
        "object_snap": False, "arm_rehome": False, "symbolic_models": models,
        "samples": [], "episodes": [], "source_hashes_at_start": hashes, "completed": False,
    }
    programs = {}
    try:
        old = json.loads(args.programs.read_text())
        for row in old["samples"]:
            if row["kind"] not in ("pick", "place", "park"):
                continue
            accepted = next(a for a in reversed(row["attempts"]) if a["accepted"])
            if accepted.get("source_type") != "backend_generated":
                raise ValueError("base program cache lacks backend-generated provenance")
            if hashlib.sha256(accepted["source"].encode()).hexdigest() != accepted["sha256"]:
                raise ValueError("base program source hash mismatch")
            if row["api_docs"] != BASE_DOCS[row["kind"]]:
                raise ValueError("base program API contract mismatch")
            programs[row["kind"]] = PolicyProgram(accepted["source"], BASE_DOCS[row["kind"]])
        result["base_programs"] = {k: hashlib.sha256(v.source.encode()).hexdigest() for k, v in programs.items()}
        result["base_programs_reused_from"] = str(args.programs)

        pack_order = [("pick_source", 0), ("place_tray", 0), ("park", 0),
                      ("pick_source", 1), ("place_tray", 1), ("park", 1)]
        custom_order = ["dual_grasp_tray", "dual_lift_tray", "dual_translate_tray",
                        "dual_lower_tray", "dual_release_tray", "dual_withdraw_hands"]

        def finish(runtime, path):
            runtime.save(path)
            runtime.close()

        def setup(runtime, path, through=None):
            records = []
            for name, arm in pack_order:
                m = model_by(models, name, arm)
                rec = execute(runtime, m, programs[m["program_kind"]], path / f"setup_{len(records)}_{name}_{arm}")
                records.append(rec)
                if not rec["effect_ok"]:
                    return records
            for name in custom_order:
                if name == through:
                    break
                m = model_by(models, name)
                if m["program_kind"] not in programs:
                    break
                rec = execute(runtime, m, programs[m["program_kind"]], path / f"setup_{len(records)}_{name}")
                records.append(rec)
                if not rec["effect_ok"]:
                    break
            return records

        if args.reuse:
            previous = json.loads(args.reuse.read_text())
            same_hash = previous["config_sha256"] == result["config_sha256"]
            previous_path = Path(previous.get("config_path", ""))
            previous_config = json.loads(previous_path.read_text()) if previous_path.is_file() else None
            visual_only = previous_config is not None and physical_config(previous_config) == physical_config(config) \
                and collision_signature(previous_config) == collision_signature(config)
            if not previous["completed"] or not (same_hash or visual_only):
                raise ValueError("cooperative cache is incomplete or changes physical task collision semantics")
            result["cooperative_programs_reused_from"] = str(args.reuse)
            result["cooperative_program_transfer"] = "same_config" if same_hash else "visual_only_same_physics"
            result["reuse_source_config_path"] = str(previous_path)
            rows = previous["samples"]
            for row in rows:
                accepted = next(a for a in reversed(row["attempts"]) if a["accepted"])
                if hashlib.sha256(accepted["source"].encode()).hexdigest() != accepted["sha256"]:
                    raise ValueError("cooperative cache source hash mismatch")
                programs[row["kind"]] = PolicyProgram(accepted["source"], DOCS[row["kind"]])
                result["samples"].append(row)
        else:
            backend = make_llm_backend("local")
            sample_specs = [
                ("dual_grasp", model_by(models, "dual_grasp_tray"), "dual_grasp_tray"),
                ("cooperative_move", model_by(models, "dual_lift_tray"), "dual_lift_tray"),
                ("dual_release", model_by(models, "dual_release_tray"), "dual_release_tray"),
                ("dual_withdraw", model_by(models, "dual_withdraw_hands"), "dual_withdraw_hands"),
            ]
            for kind, sample_model, through in sample_specs:
                trial = [0]
                def evaluate(program, kind=kind, sample_model=sample_model, through=through):
                    trial[0] += 1
                    path = args.output / "sampling" / kind / f"trial{trial[0]}"
                    runtime = DualLiftRuntime(config, path, 0)
                    try:
                        setup_records = setup(runtime, path, through=through)
                        if any(not r["effect_ok"] for r in setup_records):
                            return {"valid_trial": False, "code_executed": False, "goal_ok": False,
                                    "effect_ok": False, "diagnostics": {"setup_failed": setup_records[-1]}}
                        if kind == "cooperative_move":
                            records = []
                            for name in ("dual_lift_tray", "dual_translate_tray", "dual_lower_tray"):
                                m = model_by(models, name)
                                rec = execute(runtime, m, program, path / f"candidate_{name}")
                                records.append(rec)
                                if not rec["effect_ok"]:
                                    break
                            ok = len(records) == 3 and all(r["effect_ok"] for r in records)
                            verdict = {"valid_trial": True, "code_executed": all(r["code_executed"] for r in records),
                                       "goal_ok": ok, "effect_ok": ok,
                                       "trace": [x for r in records for x in r["trace"]],
                                       "diagnostics": {"waypoints": records,
                                                       "contents_preserved": all(runtime.packed(i) for i in (0, 1))}}
                        else:
                            verdict = execute(runtime, sample_model, program, path / "candidate")
                            if kind == "dual_withdraw" and verdict["effect_ok"]:
                                post = []
                                for arm in (0, 1):
                                    m = model_by(models, "park_after_shelf", arm)
                                    r = execute(runtime, m, programs["park"], path / f"lateral_park_{arm}")
                                    post.append(r)
                                stable = runtime.scout.stable(runtime.scout.goal_now)
                                verdict["diagnostics"].update(post_park=post, stable_after_lateral_parking=stable)
                                verdict["effect_ok"] = bool(verdict["effect_ok"] and all(r["effect_ok"] for r in post) and stable)
                                verdict["goal_ok"] = verdict["effect_ok"]
                        print("DUAL_SAMPLE", kind, trial[0], verdict["effect_ok"], verdict.get("exception"), flush=True)
                        return verdict
                    finally:
                        finish(runtime, path)
                row = PolicySampler(backend, focused_repair=True).synthesize_and_test(
                    sample_model, DOCS[kind], evaluate, args.attempts, args.output / "sampling" / kind)
                row["kind"] = kind
                row["api_docs"] = DOCS[kind]
                result["samples"].append(row)
                if not row["success"]:
                    result["blocked_on"] = kind
                    return
                accepted = next(a for a in reversed(row["attempts"]) if a["accepted"])
                programs[kind] = PolicyProgram(accepted["source"], DOCS[kind])

        result["program_hashes"] = {k: hashlib.sha256(v.source.encode()).hexdigest() for k, v in programs.items()}
        for seed in range(args.episodes):
            path = args.output / f"seed{seed}"
            path.mkdir(parents=True, exist_ok=True)
            runtime = DualLiftRuntime(config, path, seed)
            ep = {"seed": seed, "steps": [], "success": False, "goal": sorted(goal)}
            result["episodes"].append(ep)
            try:
                before = runtime.state()
                ep.update(initial_state=sorted(before), scene_sha256=runtime.scout.result["scene_sha256"],
                          initial_penetrations=runtime.scout.result["initial_penetrations"])
                if goal <= before:
                    raise ValueError("invalid trial: shelf task already solved")
                bt = timed_build(models, before, goal)
                ep["kernel"] = bt.kernel
                (path / "tree.json").write_text(bt.export())
                (path / "tree.dot").write_text(bt.export("dot"))
                for step in range(32):
                    status, model = bt.tick(runtime.state())
                    if status != "running":
                        break
                    rec = execute(runtime, model, programs[model["program_kind"]], path / f"step{step}")
                    ep["steps"].append(rec)
                    print("DUAL_BT", seed, step, action_id(model), rec["effect_ok"], flush=True)
                    if not rec["effect_ok"]:
                        break
                stable = runtime.stable_goal()
                status, _ = bt.tick(runtime.state())
                pos, quat = runtime.tray_pose()
                ep.update(bt_status=status, stable=stable,
                          success=bool(stable and status == "success" and all(r["effect_ok"] for r in ep["steps"])),
                          final_state=sorted(runtime.state()), final_tray_position=pos.tolist(),
                          final_tray_quaternion_wxyz=quat.tolist(),
                          final_containment=runtime.scout.items_contained(),
                          shelf_contact=runtime.scout.tray_supported_by_shelf(),
                          assistance_events=runtime.scout.events)
            except Exception as exc:
                ep["exception"] = f"{type(exc).__name__}: {exc}"
                print("DUAL_EPISODE_FAIL", seed, ep["exception"], flush=True)
            finally:
                finish(runtime, path)
                (path / "result.json").write_text(json.dumps(ep, indent=2))
                print("DUAL_FINAL", seed, ep["success"], flush=True)
        result["success_count"] = sum(ep["success"] for ep in result["episodes"])
        result["completed"] = len(result["episodes"]) == args.episodes and all(ep["success"] for ep in result["episodes"])
    finally:
        result["config_unchanged"] = sha(args.config) == result["config_sha256"]
        result["source_hashes_at_end"] = {str(p): sha(p) for p in paths}
        result["implementation_unchanged_during_run"] = hashes == result["source_hashes_at_end"]
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
