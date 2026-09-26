"""Contact audit for the held-basin pour without changing controller or state."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import mujoco

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"))
from core.curriculum_bridge import DOCS as BASE_DOCS
from core.held_basin_pour_bridge import DOCS, HeldBasinRuntime, domain, goal
from formal_bt import ModelLibrary
from policy_codegen import PolicyProgram
from probe_held_basin_codegen import execute


def body_name(model, geom_id):
    bid = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body#{bid}"


def arm_side(name):
    if name.endswith("_L"): return "L"
    if name.endswith("_R"): return "R"
    return None


class Audit:
    def __init__(self, runtime):
        self.runtime = runtime; self.phase = "initial"; self.rows = []; self.counts = Counter()
        self.original = mujoco.mj_step
    def install(self):
        def audited(model, data, *args, **kwargs):
            out = self.original(model, data, *args, **kwargs)
            self.sample()
            return out
        mujoco.mj_step = audited
    def restore(self):
        mujoco.mj_step = self.original
    def sample(self):
        m, d = self.runtime.env.m, self.runtime.env.d
        for c in d.contact:
            a, b = body_name(m, c.geom1), body_name(m, c.geom2)
            sa, sb = arm_side(a), arm_side(b)
            kind = None
            if sa and sb and sa != sb: kind = "inter_arm"
            elif (sa == "L" and b == "basin") or (sb == "L" and a == "basin"): kind = "left_arm_basin"
            elif (sa == "R" and b == "canL") or (sb == "R" and a == "canL"): kind = "right_arm_source"
            elif {a, b} == {"canL", "basin"}: kind = "container_container"
            if kind:
                key = (self.phase, kind, *sorted((a, b)))
                self.counts[key] += 1
                self.rows.append({"time": float(d.time), "phase": self.phase, "kind": kind,
                                  "body_a": a, "body_b": b, "distance_m": float(c.dist)})
    def summary(self):
        groups = []
        for key, n in self.counts.items():
            phase, kind, a, b = key
            ds = [r["distance_m"] for r in self.rows if r["phase"] == phase and r["kind"] == kind
                  and sorted((r["body_a"], r["body_b"])) == [a, b]]
            groups.append({"phase": phase, "kind": kind, "body_pair": [a, b],
                           "step_samples": n, "min_distance_m": min(ds)})
        return sorted(groups, key=lambda r: (r["phase"], r["kind"], r["body_pair"]))


def load_programs(base_path, custom_path):
    base = json.loads(base_path.read_text()); custom = json.loads(custom_path.read_text())
    programs = {}
    row = next(r for r in base["samples"] if r["kind"] == "pick")
    a = next(a for a in reversed(row["attempts"]) if a["accepted"])
    programs["pick"] = PolicyProgram(a["source"], BASE_DOCS["pick"])
    for row in custom["samples"]:
        a = next(a for a in reversed(row["attempts"]) if a["accepted"])
        programs[row["kind"]] = PolicyProgram(a["source"], DOCS[row["kind"]])
    return programs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--base-programs", type=Path, default=ROOT / "outputs/cabto_curriculum_codegen/final_recheck/result.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text()); models = domain(); runtime = HeldBasinRuntime(cfg, args.output, args.seed)
    programs = load_programs(args.base_programs, args.programs); audit = Audit(runtime); audit.install()
    result = {"success": False, "steps": [], "scope": "contact audit; no controller/state modification"}
    try:
        bt = ModelLibrary(models).build(runtime.state(), goal())
        for i in range(8):
            status, model = bt.tick(runtime.state())
            if status != "running": break
            audit.phase = model["name"]
            rec = execute(runtime, model, programs[model["program_kind"]], args.output / f"step{i}")
            result["steps"].append(rec)
            if not rec["effect_ok"]: break
        audit.phase = "stable_goal"; stable = runtime.stable_goal(); status, _ = bt.tick(runtime.state())
        result.update(success=bool(stable and status == "success"), stable=stable, bt_status=status)
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        audit.restore(); result["contact_groups"] = audit.summary(); result["contact_samples"] = audit.rows
        result["final_positions"] = {o: runtime.scout.pos(o).tolist() for o in runtime.env.obj_names}
        result["final_basin_center"] = runtime.scout.basin_center().tolist()
        result["final_state"] = sorted(runtime.state())
        try:
            runtime.save(args.output)
        finally:
            runtime.close()
        (args.output / "contact_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"success": result["success"], "groups": result["contact_groups"]}, indent=2))


if __name__ == "__main__": main()
