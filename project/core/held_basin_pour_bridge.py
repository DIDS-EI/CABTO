"""STRIPS and restricted APIs for pouring into a basin held by the other arm."""
from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"))
from core.curriculum_bridge import DOCS as BASE_DOCS, Runtime
from probe_held_basin_pour import HeldBasinPour

PRESENT_DOCS = {
    "target_pose": "api.target_pose() -> tuple(x,y,z): current desired basin HANDLE pose at the catch station. No arguments.",
    "move_basin": "api.move_basin(p): move the already-held basin to handle pose p. One XYZ tuple argument.",
    "settle": "api.settle(): wait while keeping the basin held. No arguments."
}
COORDINATED_POUR_DOCS = {
    "source_pose": "api.source_pose() -> tuple(x,y,z): desired source-cup handle pose above the basin. Read this first.",
    "move_source": "api.move_source(p): move the already-held source cup to p without tilting. Call before re-aligning the basin.",
    "receiver_pose": "api.receiver_pose() -> tuple(x,y,z): desired basin handle pose. Read after moving the source cup.",
    "align_receiver": "api.align_receiver(p): re-align the held basin immediately before tip_source. Source movement can make the loaded receiving arm sag, so this call MUST be after move_source.",
    "tip_source": "api.tip_source(): physically tilt the held source cup past horizontal; does not move the sphere directly or set success. Call immediately after align_receiver.",
    "settle": "api.settle(): wait while both grasps remain active. No arguments."
}
DOCS = {**BASE_DOCS, "present_receiver": PRESENT_DOCS, "coordinated_pour": COORDINATED_POUR_DOCS}


def act(name, kind, arm, obj, pre, add, delete, description):
    return {"name": name, "args": {"arm": str(arm), "obj": obj, "dest": name},
            "program_kind": kind, "pre": pre, "add": add, "del": delete,
            "description": description}


def domain():
    return [
        act("pick_basin", "pick", 1, "basin", ["empty(1)", "at_source(basin)"],
            ["holding(1,basin)"], ["empty(1)", "at_source(basin)", "parked(1)"],
            "Pick the handled basin with the receiving arm and lift it. Use the documented pick APIs."),
        act("present_basin", "present_receiver", 1, "basin", ["holding(1,basin)"],
            ["basin_ready()"], [],
            "Move the already-held basin to the catch station using target_pose, move_basin, then settle. Do not release it."),
        act("pick_source_cup", "pick", 0, "canL", ["empty(0)", "at_source(canL)", "content_in_source()"],
            ["holding(0,canL)"], ["empty(0)", "at_source(canL)", "parked(0)"],
            "Pick the handled source cup containing the sphere and lift it. Use the documented pick APIs."),
        act("pour_into_held_basin", "coordinated_pour", 0, "canL",
            ["holding(0,canL)", "holding(1,basin)", "content_in_source()", "basin_ready()"],
            ["sphere_in_basin()", "caught_by_held_basin()"], ["content_in_source()"],
            "Required order: move the held source cup above the catch zone first; then re-align the loaded receiving basin immediately before physically tipping the source; finally settle. The receiving arm must keep holding the basin throughout."),
    ]


def goal():
    return {"sphere_in_basin()", "caught_by_held_basin()", "holding(1,basin)"}


class HeldBasinRuntime(Runtime):
    def __init__(self, config, out, seed=0):
        self.scout = HeldBasinPour(deepcopy(config), out, seed)
        self.config = config; self.task = "pour"; self.env = self.scout.env
        self.p = config["pour"]; self.b = self.p["held_basin"]; self.h = config["table"]["top_z"]
        self.initial = {o: self.scout.pos(o).copy() for o in self.env.obj_names}
        self.goal = goal(); self.pour_events = self.scout.pour_events

    def content_in_source(self, _i=0):
        return self.scout.sphere_in_source()

    def state(self):
        state = set()
        for i, arm in enumerate(self.env.arms):
            attached = [o for o in ("canL", "basin") if arm.is_holding(o)]
            if not attached:
                state.add(f"empty({i})")
                ee = arm.get_ee_pose()[0]; side = 1 if i == 0 else -1
                if side * ee[1] >= .20 and ee[2] >= self.h + .20:
                    state.add(f"parked({i})")
            for obj in attached:
                state.add(f"holding({i},{obj})")
        for obj in ("canL", "basin"):
            now = self.scout.pos(obj)
            if self.scout.released(obj) and np.linalg.norm(now[:2] - self.initial[obj][:2]) < .020 \
                    and abs(now[2] - self.initial[obj][2]) < .010:
                state.add(f"at_source({obj})")
        if self.scout.sphere_in_source(): state.add("content_in_source()")
        if self.scout.basin_ready(): state.add("basin_ready()")
        if self.scout.sphere_in_basin(): state.add("sphere_in_basin()")
        if any(e.get("before_in_source") and not e.get("before_in_basin")
               and e.get("basin_ready_before_tip") and e.get("actual_tipped_past_horizontal")
               and not e.get("after_in_source") and e.get("after_in_basin")
               and e.get("basin_still_held") for e in self.pour_events):
            state.add("caught_by_held_basin()")
        return state

    @staticmethod
    def _xyz(value):
        if not isinstance(value, (tuple, list)) or len(value) != 3 \
                or not all(type(v) in (int, float) and math.isfinite(v) for v in value):
            raise ValueError(f"expected one finite XYZ tuple, got {value!r}")
        return np.asarray(value, float)

    def _basin_target(self):
        xy = np.asarray(self.b["receive_center_xy"]) + [self.b["basin_handle_offset"], 0.0]
        return np.array([*xy, self.b["receiver_transport_z"]])

    def _source_target(self):
        xy = np.asarray(self.b["receive_center_xy"]) + np.asarray(self.b["source_handle_offset_xy"])
        return np.array([*xy, self.b["source_transport_z"]])

    def api(self, model):
        kind = model["program_kind"]
        if kind == "pick":
            return super().api(model)
        if kind == "present_receiver":
            target = self._basin_target()
            def target_pose(): return tuple(map(float, target))
            def move_basin(value):
                point = self._xyz(value)
                if np.linalg.norm(point - target) > 1e-6: raise ValueError("use target_pose() unchanged")
                self.scout.transport(1, "basin", point[:2], z=point[2])
            def settle(): self.scout.hold(12)
            values = locals()
        elif kind == "coordinated_pour":
            rt = self._basin_target(); st = self._source_target()
            event = {}; order={"source_moved":False,"receiver_aligned":False}
            def receiver_pose(): return tuple(map(float, rt))
            def align_receiver(value):
                if not order["source_moved"]: raise RuntimeError("move_source must occur before final receiver alignment")
                point = self._xyz(value)
                if np.linalg.norm(point - rt) > 1e-6: raise ValueError("use receiver_pose() unchanged")
                self.scout.transport(1, "basin", point[:2], z=point[2]); self.scout.hold(8)
                if not self.scout.basin_ready(): raise RuntimeError("held basin not ready after alignment")
                order["receiver_aligned"]=True
            def source_pose(): return tuple(map(float, st))
            def move_source(value):
                point = self._xyz(value)
                if np.linalg.norm(point - st) > 1e-6: raise ValueError("use source_pose() unchanged")
                self.scout.transport(0, "canL", point[:2], z=point[2]);order["source_moved"]=True;order["receiver_aligned"]=False
            def tip_source():
                if not order["receiver_aligned"]: raise RuntimeError("align_receiver must occur immediately after move_source and before tip_source")
                event.update(before_in_source=self.scout.sphere_in_source(),
                             before_in_basin=self.scout.sphere_in_basin(),
                             basin_ready_before_tip=self.scout.basin_ready())
                self.pour_events.append(event)
                if not event["before_in_source"] or event["before_in_basin"] or not event["basin_ready_before_tip"]:
                    raise RuntimeError("invalid physical pre-tip evidence")
                self.scout.skills[0].tip_pour_joint(axis=self.p["tilt_axis"], total_angle=self.p["tilt_radians"],
                                                    n=90, hold=110, can_name="canL")
                q = self.env.get_object_pose("canL")[1]
                normal = Rotation.from_quat(q[[1,2,3,0]]).apply([0,0,1])
                event.update(actual_tipped_past_horizontal=bool(normal[2] < 0),
                             actual_source_normal=normal.tolist(),
                             after_in_source=self.scout.sphere_in_source(),
                             after_in_basin=self.scout.sphere_in_basin(),
                             basin_still_held=self.env.arms[1].is_holding("basin"))
                if event["after_in_source"] or not event["actual_tipped_past_horizontal"] \
                        or not event["after_in_basin"] or not event["basin_still_held"]:
                    raise RuntimeError("physical sphere transfer failed")
            def settle(): self.scout.hold(16)
            values = locals()
        else:
            raise ValueError(f"unsupported kind {kind}")
        return {name: values[name] for name in DOCS[kind]}

    def stable_goal(self):
        return self.scout.stable(lambda: self.goal <= self.state())

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.env.save_png(str(path / "final.png")); self.env.save_png(str(path / "final_front.png"), "front")
        self.env.save_video(str(path / "rollout.mp4"), fps=30)

    def close(self):
        if self.env.renderer: self.env.renderer.close()
