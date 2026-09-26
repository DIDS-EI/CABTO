"""Physical reference for collision-audited pour, object return and controlled HOME."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"))
from core.curriculum_bridge import DOCS as BASE_DOCS
from core.held_basin_pour_bridge import DOCS as POUR_DOCS
from core.smooth_held_basin_bridge import SmoothHeldBasinRuntime, domain
from policy_codegen import PolicyProgram
from probe_curriculum_generated_bt import execute


def model(models, name): return next(m for m in models if m["name"] == name)


def load_programs(base_path, pour_path):
    base = json.loads(base_path.read_text()); pour = json.loads(pour_path.read_text()); result = {}
    row = next(r for r in base["samples"] if r["kind"] == "pick")
    a = next(a for a in reversed(row["attempts"]) if a["accepted"])
    result["pick"] = PolicyProgram(a["source"], BASE_DOCS["pick"])
    for row in pour["samples"]:
        a = next(a for a in reversed(row["attempts"]) if a["accepted"])
        result[row["kind"]] = PolicyProgram(a["source"], POUR_DOCS[row["kind"]])
    return result


def direct_return(api):
    api["return_object"]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "scenes/held_basin_pour_smooth_v2.json")
    ap.add_argument("--base-programs", type=Path, default=ROOT / "data/programs/curriculum_pick_result.json")
    ap.add_argument("--pour-programs", type=Path, default=ROOT / "data/programs/held_basin_pour_result.json")
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text()); runtime = SmoothHeldBasinRuntime(config, args.output, args.seed)
    models = domain(); programs = load_programs(args.base_programs, args.pour_programs)
    result = {"source": "handwritten_return_home_reference_with_cached_generated_setup",
              "seed": args.seed, "steps": [], "success": False, "object_snap": False,
              "qpos_write": False, "instant_rehome": False, "controlled_actuator_home": True,
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    try:
        for name in ("pick_source_cup", "pick_basin", "present_basin", "pour_into_held_basin"):
            m = model(models, name); rec = execute(runtime, m, programs[m["program_kind"]], args.output / name)
            result["steps"].append(rec)
            if not rec["effect_ok"]: raise RuntimeError(f"setup action failed: {name}")
        runtime.api(model(models, "separate_receiver_after_pour"))["separate_receiver"]()
        runtime.api(model(models, "separate_receiver_after_pour"))["separate_receiver"]()
        runtime.api(model(models, "restore_source_upright"))["restore_upright"]()
        direct_return(runtime.api(model(models, "return_source_cup")))
        runtime.api(model(models, "home_source_arm"))["controlled_home"]()
        direct_return(runtime.api(model(models, "return_basin")))
        runtime.api(model(models, "home_receiver_arm"))["controlled_home"]()
        result["success"] = runtime.stable_goal()
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        result.update(final_state=sorted(runtime.state()), return_geometry={o: runtime.return_geometry(o) for o in ("canL", "basin")},
                      return_events=runtime.return_events, collision_free=runtime.collision_free(),
                      collision_groups=runtime.audit.summary(), sphere_local=runtime.scout.basin_local(runtime.scout.pos("ballL")).tolist())
        try:
            runtime.save(args.output)
        finally:
            runtime.close()
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print("SMOOTH_REFERENCE_FINAL", result["success"], result.get("exception"), result["collision_groups"], flush=True)


if __name__ == "__main__": main()
