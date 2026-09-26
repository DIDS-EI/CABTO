"""Replay previously model-generated API programs on the latest Cover/Blocks scenes.

The formal BT is rebuilt from the latest active configuration. Generated source is
loaded verbatim from the retained archive, then bound to a trusted task-local API
adapter whose physical transitions use RebuiltSingleArm. The adapter boundary is
reported explicitly; no new LLM call is made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from PIL import Image

from core.single_arm_rebuilt import RebuiltSingleArm

ROOT = Path(__file__).resolve().parent
STAGE3 = ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"
sys.path.insert(0, str(STAGE3))
from formal_bt import ModelLibrary, action_id
from policy_codegen import PolicyProgram


def build_domain(config):
    pairs = config["pairs"]
    domain = []
    for i, (obj, support) in enumerate(pairs):
        prior = [] if i == 0 else [f"at_target({pairs[i - 1][0]})"]
        domain += [
            {"name": "pick", "args": {"object": obj, "support": support},
             "pre": ["empty()", f"at_source({obj})"] + prior,
             "add": [f"holding({obj})"], "del": ["empty()", f"at_source({obj})"]},
            {"name": "place", "args": {"object": obj, "support": support},
             "pre": [f"holding({obj})"], "add": [f"at_target({obj})", "empty()"],
             "del": [f"holding({obj})"]},
        ]
    target = {f"at_target({obj})" for obj, _ in pairs} | {"empty()", "home_completed()"}
    domain.append({"name": "home", "args": {},
                   "pre": ["empty()"] + [f"at_target({obj})" for obj, _ in pairs],
                   "add": ["home_completed()"], "del": []})
    return domain, target


def accepted_sources(task, archive):
    data = json.loads(archive.read_text())
    rows = data["samples"] if task == "cover" else data["actions"]
    out = {}
    for row in rows:
        kind = row.get("kind") or row.get("name")
        accepted = next(attempt for attempt in reversed(row["attempts"]) if attempt["accepted"])
        mapped = {"pick_lid": "pick", "place_lid": "place", "place_on": "place"}.get(kind, kind)
        source = accepted["source"]
        docs = row.get("api_docs") or {name: f"api.{name}(...)" for name in sorted(set(re.findall(r"api\.([A-Za-z_]\w*)", source)))}
        out[mapped] = {"source": source, "api_docs": docs}
    return out


def api_for(runtime, model, task):
    obj = model["args"].get("object")
    support = model["args"].get("support")
    if task == "cover":
        return {
            "pick_lid_complete": lambda: runtime.pick(obj),
            "place_lid_complete": lambda: runtime.place(obj, support),
        }
    completed = {"pick": False, "place": False}

    def locate(name):
        # Historical Blocks code used green_block/yellow_block symbols. Bind those
        # symbolic roles to the current BT-grounded payload/support without changing
        # the generated source text.
        if name in (obj, "green_block"):
            return tuple(runtime.grip_point(obj))
        if name in (support, "yellow_block"):
            return tuple(runtime.pos(support))
        return tuple(runtime.pos(name))

    def grasp(_xy):
        if not completed["pick"]:
            runtime.pick(obj); completed["pick"] = True

    def place_release():
        if not completed["place"]:
            runtime.place(obj, support); completed["place"] = True

    def target_pose():
        p = runtime.pos(support).copy()
        p[2] += 2 * runtime.config["cube_half_size"] + .003
        return tuple(p)

    return {
        "locate": locate,
        "approach": lambda _p: None,
        "align_xy": lambda _p: None,
        "descend_to": lambda _z, _xy: None,
        "grasp": grasp,
        "record_grab_offset": lambda _name: None,
        "lift": lambda *args, **kwargs: None,
        "target_pose": target_pose,
        "move_above": lambda p: tuple(p[:2]),
        "refine_above": lambda p: tuple(p[:2]),
        "descend_place": lambda _z, _xy: None,
        "place_release": place_release,
        "retreat": lambda: None,
        "settle": lambda: None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["cover", "blocks"])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    sources = accepted_sources(args.task, args.programs)
    runtime = RebuiltSingleArm(config, args.seed, render=not args.no_render)
    domain, target = build_domain(config)
    tree = ModelLibrary(domain).build(runtime.state(), target)
    (args.output / "tree.json").write_text(tree.export())
    (args.output / "tree.dot").write_text(tree.export("dot"))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    result = {
        "task": args.task,
        "scene": config.get("scene_id"),
        "seed": args.seed,
        "program_source": "verbatim_previously_model_generated_programs",
        "program_archive": str(args.programs),
        "program_sha256": {k: hashlib.sha256(v["source"].encode()).hexdigest() for k, v in sources.items()},
        "binding": "trusted_task_local_atomic_physical_adapter",
        "formal_bt": True,
        "oracle": True,
        "success": False,
        "steps": [],
    }
    try:
        if runtime.env.render_enabled:
            runtime.env.save_png(str(args.output / "before.png"), "overview")
        for step in range(len(domain) + 1):
            before = runtime.state(); status, model = tree.tick(before)
            if status != "running":
                break
            if model["name"] == "home":
                runtime.home(); source = "api.home()"
            else:
                row = sources[model["name"]]
                program = PolicyProgram(row["source"], row["api_docs"])
                api = api_for(runtime, model, args.task)
                program.validate_api(api)
                program.run(api)
                source = row["source"]
            after = runtime.state()
            effect_ok = set(model["add"]) <= after and not set(model["del"]) & after and (set(model["pre"]) - set(model["del"])) <= after
            result["steps"].append({"step": step, "action": action_id(model), "before": sorted(before), "after": sorted(after), "effect_ok": effect_ok, "source": source})
            (args.output / f"step{step:02d}_policy.py").write_text(source)
            if not effect_ok:
                raise RuntimeError(f"effect check failed for {action_id(model)}")
        runtime.mark("final_stability")
        for _ in range(max(1, round(runtime.c["stable_seconds"] / (runtime.dt * runtime.steps)))):
            runtime.wait(runtime.dt * runtime.steps, 1.0)
            runtime.stability_ok = runtime.stability_ok and target <= runtime.state()
        assessed = runtime.assess(); final_status, _ = tree.tick(runtime.state())
        result.update(assessed, bt_status=final_status, final_state=sorted(runtime.state()))
        result["success"] = bool(assessed["success"] and final_status == "success" and all(s["effect_ok"] for s in result["steps"]))
        if runtime.env.render_enabled:
            runtime.env.save_png(str(args.output / "final.png"), "overview")
            runtime.env.save_png(str(args.output / "final_front.png"), "front")
            runtime.env.save_video(str(args.output / "rollout.mp4"), fps=30)
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["success"] = False
    finally:
        result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        runtime.close()
    print(json.dumps({"task": args.task, "success": result["success"], "steps": len(result["steps"]), "bt_status": result.get("bt_status"), "exception": result.get("exception")}, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
