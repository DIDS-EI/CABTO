"""Restore one difficulty: pack into a FREE tray that can slide on the table.

Scene dimensions, object sizes and contact parameters are frozen; only the tray
gains a free joint and mass. Reference mode calibrates the physical feasibility;
codegen mode reuses the already-accepted generated pick/place/park programs and
schedules them with the unmodified formal BT Expansion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"))
from probe_feasible_curriculum import Curriculum, containment
from core.curriculum_bridge import Runtime, DOCS, domain
from formal_bt import ModelLibrary, action_id
from policy_codegen import PolicyProgram
from probe_curriculum_generated_bt import execute


class FreeTrayScout(Curriculum):
    def tray_pose(self):
        return self.env.get_object_pose("tray")

    def packed(self, i):
        pos, quat = self.tray_pose()
        return containment(self.env.get_object_pose(f"item{i}"), (pos, quat),
                           self.p["item_half_size"], self.p["tray_inner_half_size"],
                           floor=self.p["tray_wall"], ceiling=self.p["tray_height"])["inside"] \
               and self.released(f"item{i}")

    def slot_world(self, i):
        pos, quat = self.tray_pose()
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        return pos[:2] + (rot @ np.array([*self.p["slots_xy"][i], 0.0]))[:2]

    def tray_stable(self):
        pos, quat = self.tray_pose()
        drift = float(np.linalg.norm(pos[:2] - self._tray_xy0))
        tilt = 2 * math.acos(float(np.clip(abs(quat[0]), -1, 1)))
        return drift < 0.025 and abs(pos[2] - self.h) < 0.012 and tilt < math.radians(8)

    def packing(self):
        self._tray_xy0 = self.tray_pose()[0][:2].copy()
        self.result["initial_goal_false"] = not any(self.packed(i) for i in (0, 1))
        if not self.result["initial_goal_false"]:
            raise ValueError("invalid trial: an item starts inside the tray")
        self.result["initial_tray"] = self.tray_pose()[0].tolist()
        for i in (0, 1):
            name = f"item{i}"
            self.pick(i, name, self.pos(name)[2] + 0.015)
            slot = self.slot_world(i)
            self.place(i, name, slot, self.h + self.p["tray_wall"])
            self.park(i)
            self.mark("packed_probe", item=name, tray_xy=self.tray_pose()[0][:2].tolist(),
                      packed=self.packed(i))
        def goal():
            return self.packed(0) and self.packed(1) and self.tray_stable()
        self.result["success"] = self.stable(goal)
        self.result["tray_drift_mm"] = float(np.linalg.norm(self.tray_pose()[0][:2] - self._tray_xy0) * 1000)
        self.result["final_tray"] = self.tray_pose()[0].tolist()
        self.result["scope"] = "free sliding tray packing; no box lifting or shelf transport"


class FreeTrayRuntime(Runtime):
    def __init__(self, config, out, seed=0):
        super().__init__(config, "packing", out, seed)
        self._tray_xy0 = self.env.get_object_pose("tray")[0][:2].copy()

    def tray_pose(self):
        return self.env.get_object_pose("tray")

    def slot_world(self, i):
        pos, quat = self.tray_pose()
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        return pos[:2] + (rot @ np.array([*self.p["slots_xy"][i], 0.0]))[:2]

    def packed(self, i):
        pos, quat = self.tray_pose()
        return containment(self.env.get_object_pose(f"item{i}"), (pos, quat),
                           self.p["item_half_size"], self.p["tray_inner_half_size"],
                           floor=self.p["tray_wall"], ceiling=self.p["tray_height"])["inside"] \
               and self.scout.released(f"item{i}")

    def slot_free(self, i):
        center = self.slot_world(i)
        half = np.asarray(self.p["item_half_size"])
        for j in (0, 1):
            o = f"item{j}"; p = self.scout.pos(o)
            if self.scout.released(o) and np.all(np.abs(p[:2] - center) < 2 * half[:2] + .002) \
                    and abs(p[2] - (self.h + self.p["tray_wall"] + half[2])) < .020:
                return False
        return True

    def tray_stable(self):
        pos, quat = self.tray_pose()
        drift = float(np.linalg.norm(pos[:2] - self._tray_xy0))
        tilt = 2 * math.acos(float(np.clip(abs(quat[0]), -1, 1)))
        return drift < 0.025 and abs(pos[2] - self.h) < 0.012 and tilt < math.radians(8)

    def api(self, m):
        api = super().api(m)
        if m["program_kind"] == "place":
            i = int(m["args"]["arm"])
            def target_pose():
                s = self.slot_world(i)
                return (float(s[0]), float(s[1]), float(self.h + self.p["tray_wall"]))
            api["target_pose"] = target_pose
        return api

    def stable_goal(self):
        return self.scout.stable(lambda: self.goal <= self.state() and self.tray_stable())


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "scenes/packing_free_tray_v1.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codegen", action="store_true", help="reuse accepted generated programs and run formal BT")
    ap.add_argument("--programs", type=Path, default=ROOT / "outputs/cabto_curriculum_codegen/final_recheck/result.json")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "config_sha256.txt").write_text(sha(args.config))

    if not args.codegen:
        scout = FreeTrayScout(config, "packing", args.output, args.seed)
        scout.run()
        return

    old = json.loads(args.programs.read_text())
    source_cfg = json.loads(Path(old["config_path"]).read_text()) if "config_path" in old else config
    def comparable(c):
        c = json.loads(json.dumps(c))
        c.pop("version", None)
        for k in ("tray_free", "tray_mass", "label"):
            c["packing"].pop(k, None)
        return json.dumps(c, sort_keys=True)
    if comparable(source_cfg) != comparable(config):
        raise ValueError("program transfer requires identical packing geometry except tray freedom; check table/robots/items/tray/slots")
    result_note = {"program_config_sha256": old.get("config_sha256"),
                   "target_config_sha256": sha(args.config),
                   "transfer": "programs are position-agnostic; only place target_pose binding switches to the live tray pose"}
    progs = {}
    for row in old["samples"]:
        if row["kind"] not in ("pick", "place", "park"):
            continue
        a = next(a for a in reversed(row["attempts"]) if a["accepted"])
        if hashlib.sha256(a["source"].encode()).hexdigest() != a["sha256"]:
            raise ValueError("cache hash mismatch")
        if row["api_docs"] != DOCS[row["kind"]]:
            raise ValueError("cache API mismatch")
        progs[row["kind"]] = PolicyProgram(a["source"], DOCS[row["kind"]])

    models = domain("packing")
    for seed in range(3):
        p = args.output / f"seed{seed}"; p.mkdir(parents=True, exist_ok=True)
        runtime = FreeTrayRuntime(config, p, seed)
        ep = {"seed": seed, "steps": [], "success": False, "goal": sorted(runtime.goal)}
        try:
            before = runtime.state()
            ep["initial_state"] = sorted(before)
            ep["scene_sha256"] = runtime.scout.result["scene_sha256"]
            ep["tray_drift0_mm"] = float(np.linalg.norm(runtime.tray_pose()[0][:2] - runtime._tray_xy0) * 1000)
            if runtime.goal <= before:
                raise ValueError("invalid trial: goal already true")
            bt = ModelLibrary(models).build(before, runtime.goal)
            ep["kernel"] = bt.kernel
            (p / "tree.json").write_text(bt.export()); (p / "tree.dot").write_text(bt.export("dot"))
            for step in range(20):
                status, m = bt.tick(runtime.state())
                if status != "running":
                    break
                rec = execute(runtime, m, progs[m["program_kind"]], p / f"step{step}")
                ep["steps"].append(rec)
                print("FREETRAY_BT", seed, step, action_id(m), rec["effect_ok"], flush=True)
                if not rec["effect_ok"]:
                    break
            stable = runtime.stable_goal()
            status, _ = bt.tick(runtime.state())
            ep.update(success=bool(stable and status == "success" and all(v["effect_ok"] for v in ep["steps"])),
                      stable=stable, bt_status=status, tray_drift_mm=float(np.linalg.norm(runtime.tray_pose()[0][:2] - runtime._tray_xy0) * 1000),
                      final_tray=runtime.tray_pose()[0].tolist(), final_state=sorted(runtime.state()))
        except Exception as e:
            ep["exception"] = f"{type(e).__name__}: {e}"
            print("FREETRAY_FAIL", seed, ep["exception"], flush=True)
        finally:
            runtime.save(p)
            runtime.close()
            (p / "result.json").write_text(json.dumps(ep, indent=2))
            print("FREETRAY_FINAL", seed, ep["success"], ep.get("tray_drift_mm"), flush=True)


if __name__ == "__main__":
    main()
