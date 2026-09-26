"""Collision-audited held-basin pour with object return and controlled home motion."""
from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/stage2_scripted"))
from core.held_basin_pour_bridge import DOCS as BASE_DOCS, HeldBasinRuntime, domain as base_domain
from exp4_env import HOME_QPOS
from probe_five_scene_preflight import geom_corners
from probe_held_basin_collision import Audit

SEPARATE_DOCS = {"separate_receiver": "api.separate_receiver(): the ONLY available method. Call exactly once. It moves the held basin straight outward to its negative-y safe lane while retaining the sphere."}
RESTORE_DOCS = {"restore_upright": "api.restore_upright(): the ONLY available method. Call exactly once with no arguments. It already checks holding and actuates the tilted source cup back upright; do not sense or assign any pose."}
RETURN_DOCS = {"return_object": "api.return_object(): the ONLY available method. Call exactly once. It carries the bound held object through a collision-separated waypoint, lowers it to its calibrated original table pose, releases it, retreats outward, and settles. It never writes object state."}
HOME_DOCS = {"controlled_home": "api.controlled_home(): with an EMPTY open hand, move through a safe outer Cartesian waypoint, then interpolate actuator targets to HOME. It never writes qpos. No arguments."}
DOCS = {**BASE_DOCS, "separate_receiver": SEPARATE_DOCS, "restore_source": RESTORE_DOCS, "return_source": RETURN_DOCS,
        "return_basin": RETURN_DOCS, "controlled_home": HOME_DOCS}


def action(name, kind, arm, obj, pre, add, delete, description):
    return {"name": name, "args": {"arm": str(arm), "obj": obj, "dest": name},
            "program_kind": kind, "pre": list(pre), "add": list(add), "del": list(delete),
            "description": description}


def domain():
    models = deepcopy(base_domain())
    for m in models:
        m["pre"].append("collision_free()")
        if m["name"] == "pick_source_cup":
            m["del"].extend(["home(0)", "source_returned()", "hand_raised(0)"])
        elif m["name"] == "pick_basin":
            m["del"].extend(["home(1)", "basin_returned()", "hand_raised(1)"])
        elif m["name"] == "pour_into_held_basin":
            m["del"].append("source_upright()")
            m["add"].append("payload_retained()")
    caught = ["sphere_in_basin()", "caught_by_held_basin()", "payload_retained()", "collision_free()"]
    models.extend([
        action("separate_receiver_after_pour", "separate_receiver", 1, "basin",
               ["holding(0,canL)", "holding(1,basin)", "basin_ready()"] + caught,
               ["receiver_separated()"], ["basin_ready()"],
               "Write exactly def policy(api): api.separate_receiver(). The only API moves the held basin outward while retaining the sphere."),
        action("restore_source_upright", "restore_source", 0, "canL",
               ["holding(0,canL)", "receiver_separated()"] + caught, ["source_upright()"], [],
               "Write exactly def policy(api): api.restore_upright(). The only API is restore_upright; do not inspect pose, held state, or add variables."),
        action("return_source_cup", "return_source", 0, "canL",
               ["holding(0,canL)", "source_upright()"] + caught,
               ["source_returned()", "empty(0)", "hand_raised(0)"], ["holding(0,canL)"],
               "Write exactly this function with no variables and no arguments to the method: def policy(api): api.return_object(). The only API returns and releases the source cup, then retreats."),
        action("home_source_arm", "controlled_home", 0, "none",
               ["source_returned()", "empty(0)", "hand_raised(0)"] + caught,
               ["home(0)"], [],
               "The source arm is empty and vertically clear. Call controlled_home exactly once to move by actuators to HOME."),
        action("return_basin", "return_basin", 1, "basin",
               ["holding(1,basin)", "home(0)"] + caught,
               ["basin_returned()", "empty(1)", "hand_raised(1)"],
               ["holding(1,basin)", "basin_ready()"],
               "Write exactly this function with no variables and no arguments to the method: def policy(api): api.return_object(). The only API slowly returns the basin with its caught sphere, releases it, then retreats."),
        action("home_receiver_arm", "controlled_home", 1, "none",
               ["basin_returned()", "empty(1)", "hand_raised(1)", "home(0)"] + caught,
               ["home(1)"], [],
               "The receiver arm is empty and vertically clear. Call controlled_home exactly once to move by actuators to HOME."),
    ])
    return models


def goal():
    return {"sphere_in_basin()", "caught_by_held_basin()", "payload_retained()", "source_returned()",
            "basin_returned()", "empty(0)", "empty(1)", "home(0)", "home(1)", "collision_free()"}


class SmoothHeldBasinRuntime(HeldBasinRuntime):
    def __init__(self, config, out, seed=0):
        super().__init__(config, out, seed)
        self.goal = goal(); self.pre_tip_target = None; self.return_events = []
        self.initial_quat = {o: self.env.get_object_pose(o)[1].copy() for o in ("canL", "basin")}
        self.initial_rot = {o: self.env.d.xmat[self.env.obj_bid[o]].reshape(3, 3).copy() for o in ("canL", "basin")}
        self.audit = Audit(self); self.payload_monitor_active = False; self.payload_escaped = False
        raw_sample = self.audit.sample
        def sample_with_payload_guard():
            raw_sample()
            if self.payload_monitor_active and not self.scout.sphere_in_basin():
                self.payload_escaped = True
        self.audit.sample = sample_with_payload_guard
        self.audit.install(); self._audit_restored = False
        self.home_dirty = [False, False]; self.home_events = set()

    def collision_free(self):
        tol = float(self.b.get("forbidden_penetration_tolerance_m", 0.0))
        return not any(row["distance_m"] < -tol for row in self.audit.rows)

    def _handle_height(self, obj):
        return self.p["handle_height"] if obj == "canL" else self.b["basin_handle_height"]

    def _return_origin_xy(self, obj):
        if obj == "canL" and self.b.get("source_drop_center_xy"):
            return np.asarray(self.b["source_drop_center_xy"], float) + [self.p["handle_offset"], 0.0]
        return self.initial[obj][:2]

    def return_geometry(self, obj):
        p, q = self.env.get_object_pose(obj); rot = self.env.d.xmat[self.env.obj_bid[obj]].reshape(3, 3)
        corners = geom_corners(self.env.m,self.env.d,obj); bottom = float(corners[:,2].min())
        rotation_error = 2 * math.acos(float(np.clip(abs(np.dot(q,self.initial_quat[obj])),-1,1)))
        xy_error = float(np.linalg.norm(p[:2] - self._return_origin_xy(obj)))
        released = self.scout.released(obj)
        rotation_tol = math.radians(self.b.get("return_rotation_tolerance_deg", 10.0))
        ok = bool(xy_error < .020 and abs(bottom - self.h) < .010
                  and rotation_error < rotation_tol and released)
        return {"returned": ok, "xy_error_m": xy_error, "bottom_gap_m": bottom - self.h,
                "rotation_error_deg": math.degrees(rotation_error), "released": released,
                "origin": p.tolist()}

    def home(self, i):
        arm = self.env.arms[i]
        q = np.array([self.env.d.qpos[a] for a in arm.arm_qadr])
        obj = "canL" if i == 0 else "basin"
        evidence = (not self.home_dirty[i]) or (i in self.home_events)
        return bool(evidence and self.scout.released(obj) and arm._grip_cmd >= .5
                    and np.linalg.norm(q - HOME_QPOS) < .10)

    def state(self):
        state = super().state()
        if self.collision_free(): state.add("collision_free()")
        if "caught_by_held_basin()" in state and not self.payload_escaped and self.scout.sphere_in_basin():
            state.add("payload_retained()")
        if self.return_geometry("canL")["returned"]: state.add("source_returned()")
        if self.return_geometry("basin")["returned"]: state.add("basin_returned()")
        basin_center = self.scout.basin_center()
        if self.env.arms[1].is_holding("basin") and np.linalg.norm(
                basin_center[:2] - np.asarray(self.b["basin_safe_lane_center_xy"])) < .035:
            state.add("receiver_separated()")
        rot = self.env.d.xmat[self.env.obj_bid["canL"]].reshape(3, 3)
        upright_tol = self.b.get("transport_upright_tolerance_deg", 15.0)
        if rot[2, 2] > math.cos(math.radians(upright_tol)): state.add("source_upright()")
        for i in (0, 1):
            if f"empty({i})" in state and self.env.arms[i].get_ee_pose()[0][2] > self.h + .25:
                state.add(f"hand_raised({i})")
            if self.home(i): state.add(f"home({i})")
        return state

    def _go_object_origin(self, i, obj, target_origin):
        sk = self.scout.skills[i]; ee = sk.ee(); current = self.scout.pos(obj)
        target_ee = ee + np.asarray(target_origin, float) - current
        err = sk.move_to(target_ee, grip=0.0, tol=.007, max_steps=180, ki=.35)
        if err > .035: raise RuntimeError(f"{obj} safe-lane residual {err:.4f}m")
        sk.refine_above(np.asarray(target_origin[:2]), high_z=target_origin[2], obj_name=obj)
        return float(err)

    def _move_object_slow(self, i, obj, target_origin, max_steps=360):
        arm = self.env.arms[i]; sk = self.scout.skills[i]; target = np.asarray(target_origin, float)
        saved = arm.bias_world.copy(); arm.bias_world = np.zeros(3)
        try:
            for step in range(max_steps):
                error = target - self.scout.pos(obj)
                if float(np.linalg.norm(error)) < .010:
                    self.scout.hold(6); break
                action = np.zeros(7); action[:3] = np.clip(error, -.008, .008); action[6] = 0.0
                self.env.step_arms({arm.s: action}, sub_steps=8)
                if step % 2 == 0: sk._tick()
        finally: arm.bias_world = saved
        residual = float(np.linalg.norm(target - self.scout.pos(obj)))
        if residual > .025: raise RuntimeError(f"slow {obj} path residual {residual:.4f}m")
        if obj == "basin" and not self.scout.sphere_in_basin():
            raise RuntimeError("sphere escaped during slow basin transport")
        return residual

    def _safe_origin(self, obj):
        if obj == "canL": xy = self.b["source_safe_lane_xy"]
        else: xy = np.asarray(self.b["basin_safe_lane_center_xy"]) + [self.b["basin_handle_offset"], 0.0]
        return np.array([*xy, self.h + .32])

    def _controlled_home(self, i):
        obj = "canL" if i == 0 else "basin"
        if not self.scout.released(obj) or self.env.arms[i].is_holding():
            raise RuntimeError("controlled home requires this arm to release its task object")
        arm = self.env.arms[i]; sk = self.scout.skills[i]
        safe = np.asarray(self.b["home_cartesian"][i], float)
        err = sk.move_to(safe, grip=1.0, tol=.012, max_steps=180, ki=.25)
        if err > .04: raise RuntimeError(f"home staging residual {err:.4f}m")
        start = arm.arm_target.copy(); steps = int(self.b["controlled_home_steps"])
        self.return_events.append({"event": "controlled_actuator_home", "arm": i,
                                   "q_start": start.tolist(), "q_goal": HOME_QPOS.tolist(),
                                   "steps": steps, "time": float(self.env.d.time)})
        for k in range(1, steps + 1):
            self.env.step_joint(arm, start + (HOME_QPOS - start) * (k / steps), grip=1.0, sub_steps=8)
            sk._tick()
        for _ in range(12): self.env.step_joint(arm, HOME_QPOS, grip=1.0, sub_steps=8); sk._tick()
        self.home_events.add(i)
        if not self.home(i): raise RuntimeError(f"arm {i} did not converge to controlled HOME")

    def api(self, model):
        kind = model["program_kind"]; self.audit.phase = model["name"]
        if kind not in ("separate_receiver", "restore_source", "return_source", "return_basin", "controlled_home"): 
            if kind == "pick":
                self.home_dirty[int(model["args"]["arm"])] = True
            api = super().api(model)
            if kind == "coordinated_pour":
                original = api["tip_source"]
                def capture_tip():
                    self.pre_tip_target = self.env.arms[0].arm_target.copy()
                    result = original()
                    if not self.scout.sphere_in_basin():
                        raise RuntimeError("cannot activate payload retention before sphere is in basin")
                    self.payload_monitor_active = True
                    return result
                api["tip_source"] = capture_tip
            return api
        i = int(model["args"]["arm"]); obj = model["args"]["obj"]
        arm = self.env.arms[i]; sk = self.scout.skills[i]
        def separate_receiver():
            if not arm.is_holding("basin") or not self.scout.sphere_in_basin():
                raise RuntimeError("receiver separation requires held basin with caught sphere")
            current_center = self.scout.basin_center()
            target_center = np.array([*self.b["basin_safe_lane_center_xy"], current_center[2]])
            target_origin = self.scout.pos("basin") + target_center - current_center
            self._move_object_slow(1, "basin", target_origin)
            self.scout.hold(12)
            if not self.scout.sphere_in_basin():
                raise RuntimeError("caught sphere left basin during receiver separation")
        def restore_upright():
            if self.pre_tip_target is None: raise RuntimeError("no actual pre-tip arm target recorded")
            if not arm.is_holding("canL"): raise RuntimeError("source cup no longer held")
            start = arm.arm_target.copy()
            target = arm.solve_ik(sk.ee() + arm.bias_world, arm.grasp_quat, restarts=64, w_rot=3.0)
            arm.lock_down = False
            self.return_events.append({"event": "actuated_source_untip", "q_start": start.tolist(),
                                       "q_goal": target.tolist(), "time": float(self.env.d.time)})
            for k in range(1, 91): self.env.step_joint(arm, start + (target - start) * (k / 90), grip=0.0); sk._tick()
            for _ in range(100): self.env.step_joint(arm, target, grip=0.0); sk._tick()
            arm.lock_down = True
            # Re-enable the normal downward-orientation servo and give it a small
            # upward correction segment. Joint reversal alone can leave the
            # compliant held cup 10–15 degrees short of upright.
            current = sk.ee()
            err = sk.move_to([current[0], current[1], current[2] + .025], grip=0.0,
                             tol=.012, max_steps=80, ki=.0)
            if err > .035:
                raise RuntimeError(f"upright correction residual {err:.4f}m")
            # Position may converge before the orientation servo has finished.
            sk.hold(grip=0.0, n=80)
            rot = self.env.d.xmat[self.env.obj_bid["canL"]].reshape(3, 3)
            upright_tol = self.b.get("transport_upright_tolerance_deg", 15.0)
            if rot[2, 2] < math.cos(math.radians(upright_tol)):
                raise RuntimeError(f"source cup remained tilted after actuated restore: normal_z={rot[2,2]:.4f}")
        def safe_pose(): return tuple(map(float, self._safe_origin(obj)))
        def move_safe(value):
            point = self._xyz(value)
            if np.linalg.norm(point - self._safe_origin(obj)) > 1e-6: raise ValueError("use safe_pose() unchanged")
            return self._move_object_slow(i, obj, point) if obj == "basin" else self._go_object_origin(i, obj, point)
        def target_pose():
            key = "source_return_preoffset_xy" if obj == "canL" else "basin_return_preoffset_xy"
            xy = self._return_origin_xy(obj) + np.asarray(self.b.get(key, [0.0, 0.0]), float)
            return (float(xy[0]), float(xy[1]), float(self.h))
        def move_above(value):
            p = self._xyz(value); target = [p[0], p[1], self.h + .30]
            return self._move_object_slow(i, obj, target) if obj == "basin" else self._go_object_origin(i, obj, target)
        def align_target(value):
            p = self._xyz(value); return tuple(map(float, sk.refine_above(p[:2], high_z=self.h + .30, obj_name=obj)))
        def lower(value):
            p = self._xyz(value); sk.record_grab_offset(obj); saved = sk.block_half
            try:
                sk.block_half = self._handle_height(obj)
                result = sk.descend_place_tracked(p[:2], p[2], obj, gap=.003, suppress_feedforward=True)
                self.scout.hold(20)
                return result
            finally: sk.block_half = saved
        def release():
            arm._grasp_enabled = False
            for _ in range(8):
                self.env.hold_arms({arm.s: 1.0}, sub_steps=4); sk._tick()
            arm.release_all()
            if arm.is_holding(obj): raise RuntimeError("object weld remained active after release")
        def retreat():
            if arm.is_holding(obj): raise RuntimeError("release object before retreat")
            saved = arm.bias_world.copy(); arm.bias_world = np.zeros(3)
            try:
                ee = sk.ee(); side = 1.0 if i == 0 else -1.0
                target = [ee[0], ee[1] + side * .10, self.h + .28]
                err = sk.move_to(target, grip=1.0, tol=.015, max_steps=180, ki=.15)
            finally: arm.bias_world = saved
            if err > .04 or sk.ee()[2] < self.h + .23:
                raise RuntimeError(f"outward retreat failed: err={err:.4f}, z={sk.ee()[2]:.4f}")
            return float(err)
        def settle(): self.scout.hold(16)
        def return_object():
            if self.return_geometry(obj)["returned"] and self.scout.released(obj):
                settle(); return
            safe = safe_pose(); move_safe(safe)
            target = target_pose(); move_above(target); align_target(target); lower(target)
            release(); retreat(); settle()
        def controlled_home(): return self._controlled_home(i)
        values = locals(); docs = DOCS[kind]
        return {name: values[name] for name in docs}

    def stable_goal(self):
        self.audit.phase = "final_stability"
        return self.scout.stable(lambda: self.goal <= self.state() and self.scout.sphere_in_basin())

    def save(self, path):
        super().save(path)
        (Path(path) / "collision_audit.json").write_text(__import__("json").dumps({
            "collision_free": self.collision_free(), "groups": self.audit.summary(),
            "samples": self.audit.rows}, indent=2))

    def close(self):
        if not self._audit_restored:
            self.audit.restore(); self._audit_restored = True
        super().close()
