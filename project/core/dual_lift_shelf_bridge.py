"""STRIPS observations and restricted APIs for cooperative tray shelf transport.

The scene and low-level cooperative controllers are calibrated separately in
probe_dual_lift_shelf.py. Generated programs only see finite plain-data APIs;
they cannot access MuJoCo state, write object poses, rehome arms, or set success.
"""
from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.curriculum_bridge import DOCS as BASE_DOCS, domain as base_domain
from probe_dual_lift_shelf import DualLiftShelfScout
from probe_free_tray_packing import FreeTrayRuntime


DUAL_GRASP_DOCS = {
    "target_handles": "api.target_handles() -> tuple(left_xyz, right_xyz): read both tray-handle points. Assign the WHOLE return to exactly one local variable, e.g. handles = api.target_handles(); tuple unpacking is forbidden.",
    "move_both_above": "api.move_both_above(handles): move both OPEN hands above the handles. Pass the ONE handles variable returned by target_handles().",
    "move_both_to": "api.move_both_to(handles): descend both OPEN hands to the handles. Pass the same ONE handles variable; this call is required before close_both().",
    "close_both": "api.close_both(): close both grippers at their current handle poses. It does not attach the tray.",
    "attach_both": "api.attach_both(): after geometric checks, activate the two declared grasp-assist welds. No arguments.",
    "settle": "api.settle(): wait while preserving both gripper commands. No arguments."
}
COOPERATIVE_MOVE_DOCS = {
    "target_tray_pose": "api.target_tray_pose() -> tuple(x,y,z): read the bound tray-base waypoint for this action. No arguments.",
    "move_tray_to": "api.move_tray_to(p): synchronously translate both CLOSED hands so the jointly held tray reaches one XYZ waypoint p. One tuple argument.",
    "settle": "api.settle(): wait while preserving the dual grasp. No arguments."
}
DUAL_RELEASE_DOCS = {
    "open_both": "api.open_both(): open both grippers without moving their Cartesian positions. No arguments.",
    "detach_both": "api.detach_both(): deactivate both declared tray grasp-assist welds after opening. No arguments.",
    "settle": "api.settle(): wait for the released tray to rest on the shelf. No arguments."
}
DUAL_WITHDRAW_DOCS = {
    "raise_both": "api.raise_both(): move both OPEN hands vertically upward before any lateral parking motion. No arguments.",
    "settle": "api.settle(): wait while preserving open grippers. No arguments."
}
DOCS = {**BASE_DOCS, "dual_grasp": DUAL_GRASP_DOCS,
        "cooperative_move": COOPERATIVE_MOVE_DOCS,
        "dual_release": DUAL_RELEASE_DOCS,
        "dual_withdraw": DUAL_WITHDRAW_DOCS}

DESCRIPTIONS = {
    "dual_grasp": "Both arms are empty and parked after two items were packed. Required order: handles = target_handles(); move_both_above(handles); move_both_to(handles); close_both(); attach_both(); settle(). Assign only to one local name, never unpack a tuple. Finish jointly holding the tray on the table.",
    "cooperative_move": "Both arms already jointly hold the tray. Read this action's tray-base waypoint, synchronously move the tray to it, and settle. Do not release either gripper.",
    "dual_release": "The jointly held tray is supported by the shelf. Open both grippers, detach both declared welds, and settle. Do not move laterally.",
    "dual_withdraw": "The tray is released on the shelf and both open hands are still near its handles. Raise both hands vertically and settle; do not perform lateral parking."
}


def action(name, kind, pre, add, delete):
    return {"name": name, "args": {"arm": "both", "obj": "tray", "dest": name},
            "program_kind": kind, "pre": list(pre), "add": list(add),
            "del": list(delete), "description": DESCRIPTIONS[kind]}


def lift_domain():
    models = deepcopy(base_domain("packing"))
    for m in models:
        if m["name"] == "park":
            # Existing generated place does not retreat vertically. Preserve its
            # previously validated table-packing sequence, but make this park
            # unavailable once the tray has left the table.
            m["pre"].append("tray_at_table()")
    for i in (0, 1):
        models.append({"name": "park_after_shelf", "args": {"arm": str(i), "obj": "none", "dest": "park"},
                       "program_kind": "park",
                       "pre": [f"empty({i})", f"hands_safe_for_lateral({i})", "tray_on_shelf()"],
                       "add": [f"parked({i})"], "del": [],
                       "description": "After the tray is released on the shelf and this open hand was raised vertically, move the empty arm to its safe parking pose."})
    packed = ["packed(0)", "packed(1)"]
    models.extend([
        action("dual_grasp_tray", "dual_grasp",
               packed + ["empty(0)", "empty(1)", "parked(0)", "parked(1)", "tray_at_table()"],
               ["dual_holding_tray()"],
               ["empty(0)", "empty(1)", "parked(0)", "parked(1)", "hands_raised()",
                "hands_safe_for_lateral(0)", "hands_safe_for_lateral(1)"]),
        action("dual_lift_tray", "cooperative_move",
               packed + ["dual_holding_tray()", "tray_at_table()"],
               ["tray_lifted()"], ["tray_at_table()"]),
        action("dual_translate_tray", "cooperative_move",
               packed + ["dual_holding_tray()", "tray_lifted()"],
               ["tray_above_shelf()"], ["tray_lifted()"]),
        action("dual_lower_tray", "cooperative_move",
               packed + ["dual_holding_tray()", "tray_above_shelf()"],
               ["tray_on_shelf_held()"], ["tray_above_shelf()"]),
        action("dual_release_tray", "dual_release",
               packed + ["dual_holding_tray()", "tray_on_shelf_held()"],
               ["empty(0)", "empty(1)", "tray_on_shelf()", "hands_near_shelf()"],
               ["dual_holding_tray()", "tray_on_shelf_held()"]),
        action("dual_withdraw_hands", "dual_withdraw",
               packed + ["empty(0)", "empty(1)", "tray_on_shelf()", "hands_near_shelf()"],
               ["hands_raised()", "hands_safe_for_lateral(0)", "hands_safe_for_lateral(1)"],
               ["hands_near_shelf()"]),
    ])
    return models


def lift_goal():
    return {"packed(0)", "packed(1)", "tray_on_shelf()", "parked(0)", "parked(1)"}


class DualLiftRuntime(FreeTrayRuntime):
    def __init__(self, config, out, seed=0):
        self.scout = DualLiftShelfScout(deepcopy(config), "packing", out, seed)
        self.config = config
        self.task = "packing"
        self.env = self.scout.env
        self.p = config["packing"]
        self.h = config["table"]["top_z"]
        self.initial = {o: self.scout.pos(o).copy() for o in self.env.obj_names}
        self._tray_xy0 = self.tray_pose()[0][:2].copy()
        self.goal = lift_goal()
        self.pour_events = []

    def _dual_holding(self):
        return all(self.scout.tray_weld_active(i) and self.env.arms[i]._grip_cmd < 0.5
                   for i in (0, 1))

    def _rotation_error(self):
        quat = self.tray_pose()[1]
        return float(np.linalg.norm(Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_rotvec()))

    def _tray_near(self, xyz, xy_tol=0.035, z_tol=0.025):
        pos = self.tray_pose()[0]
        return bool(np.linalg.norm(pos[:2] - np.asarray(xyz[:2])) < xy_tol
                    and abs(pos[2] - xyz[2]) < z_tol
                    and self._rotation_error() < math.radians(self.p["shelf_tilt_tolerance_deg"]))

    def state(self):
        state = super().state()
        pos = self.tray_pose()[0]
        dual = self._dual_holding()
        detached = not any(self.scout.tray_weld_active(i) for i in (0, 1))
        if self._tray_near([*self._tray_xy0, self.h], xy_tol=.030, z_tol=.018):
            state.add("tray_at_table()")
        if dual:
            state.add("dual_holding_tray()")
            if self._tray_near([*self._tray_xy0, self.p["carry_tray_base_z"]], xy_tol=.050, z_tol=.035):
                state.add("tray_lifted()")
            if self._tray_near([*self.p["shelf_xy"], self.p["carry_tray_base_z"]], xy_tol=.045, z_tol=.035):
                state.add("tray_above_shelf()")
            if self._tray_near([*self.p["shelf_xy"], self.p["shelf_top_z"]], xy_tol=.030, z_tol=.020) \
                    and self.scout.tray_supported_by_shelf() and self.scout.tray_footprint_on_shelf():
                state.add("tray_on_shelf_held()")
        if self.scout.tray_on_shelf():
            state.add("tray_on_shelf()")
        handles = [self.scout.handle_world(i) for i in (0, 1)]
        if detached:
            open_hands = [self.env.arms[i]._grip_cmd >= .5 for i in (0, 1)]
            distances = [np.linalg.norm(self.env.arms[i].get_ee_pose()[0] - handles[i]) for i in (0, 1)]
            if all(open_hands) and max(distances) < .080:
                state.add("hands_near_shelf()")
            for i in (0, 1):
                if open_hands[i] and self.env.arms[i].get_ee_pose()[0][2] >= self.h + .20:
                    state.add(f"hands_safe_for_lateral({i})")
            raised_z = self.p["shelf_top_z"] + self.p["lift_handle_z"] + .075
            if all(open_hands) and all(self.env.arms[i].get_ee_pose()[0][2] >= raised_z for i in (0, 1)):
                state.add("hands_raised()")
        return state

    @staticmethod
    def _points(value, count):
        if not isinstance(value, (tuple, list)) or len(value) != count:
            raise ValueError(f"expected {count} coordinate tuples")
        result = []
        for point in value:
            if not isinstance(point, (tuple, list)) or len(point) != 3 \
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in point):
                raise ValueError(f"invalid XYZ coordinate {point!r}")
            result.append(np.asarray(point, float))
        return result

    def _attach_both(self, handles):
        tcp = [float(np.linalg.norm(self.env.arms[i].get_ee_pose()[0] - handles[i])) for i in (0, 1)]
        finger = [float(np.linalg.norm(self.env.arms[i]._finger_mid() - handles[i])) for i in (0, 1)]
        if max(tcp) > .045 or max(finger) > .12:
            raise RuntimeError(f"dual handle geometry check failed: tcp={tcp}, finger={finger}")
        for i, arm in enumerate(self.env.arms):
            if arm._grip_cmd >= .5:
                raise RuntimeError("attach requires both grippers closed")
            arm._set_weld("tray", True)
            self.scout.events.append({"event": "generated_api_dual_tray_grasp_weld", "arm": i,
                                      "sim_time": float(self.env.d.time), "finger_error_m": finger[i]})
        self.scout.hold(12)
        if not self._dual_holding():
            raise RuntimeError("both tray welds were not active after checked attachment")

    def api(self, model):
        kind = model["program_kind"]
        if kind in BASE_DOCS:
            api = super().api(model)
            if kind == "park" and self.config.get("visual_style", {}).get("physical_cardboard"):
                # The accepted legacy place program releases at box-floor height.
                # With real thick walls/flaps, a direct lateral park can push the
                # free box.  Preserve the generated one-call park interface but
                # execute a physical vertical clearance segment first.
                base_move = api["move_to"]
                i = int(model["args"]["arm"])
                def move_to(point):
                    ee = self.env.arms[i].get_ee_pose()[0]
                    safe = np.array([ee[0], ee[1], max(ee[2], self.h + .34)])
                    self.scout.go(i, safe, 1.0)
                    self.scout.events.append({"event": "physical_prepark_vertical_clearance", "arm": i,
                                              "sim_time": float(self.env.d.time), "target": safe.tolist()})
                    return base_move(point)
                api["move_to"] = move_to
            return api
        if kind == "dual_grasp":
            cached = {}
            def target_handles():
                handles = tuple(tuple(map(float, self.scout.handle_world(i))) for i in (0, 1))
                cached["handles"] = [np.asarray(p, float) for p in handles]
                return handles
            def require_handles(value):
                points = self._points(value, 2)
                expected = cached.get("handles")
                if expected is None or max(np.linalg.norm(points[i] - expected[i]) for i in (0, 1)) > 1e-6:
                    raise ValueError("use the exact current tuple returned by target_handles()")
                return points
            def move_both_above(value):
                points = require_handles(value)
                targets = [np.array([p[0], p[1], self.h + .29]) for p in points]
                return tuple(self.scout.dual_move_targets(targets, grip=1.0))
            def move_both_to(value):
                return tuple(self.scout.dual_move_targets(require_handles(value), grip=1.0, max_steps=220))
            def close_both():
                for arm in self.env.arms:
                    arm._grasp_enabled = False
                for _ in range(24):
                    self.env.hold_arms({"_L": 0.0, "_R": 0.0}, sub_steps=8)
                    self.env.record_frame("overview")
            def attach_both():
                if "handles" not in cached:
                    raise RuntimeError("target handles were not read")
                self._attach_both(cached["handles"])
            def settle(): self.scout.hold(12)
            values = locals()
        elif kind == "cooperative_move":
            phase = model["name"]
            if phase == "dual_lift_tray": target = np.array([*self._tray_xy0, self.p["carry_tray_base_z"]])
            elif phase == "dual_translate_tray": target = np.array([*self.p["shelf_xy"], self.p["carry_tray_base_z"]])
            elif phase == "dual_lower_tray": target = np.array([*self.p["shelf_xy"], self.p["shelf_top_z"]])
            else: raise ValueError(f"unknown cooperative waypoint {phase}")
            def target_tray_pose(): return tuple(map(float, target))
            def move_tray_to(value):
                point = self._points((value,), 1)[0]
                if np.linalg.norm(point - target) > 1e-6:
                    raise ValueError("use the bound target_tray_pose() without modification")
                self.scout.dual_translate_tray(point, phase)
            def settle(): self.scout.hold(12)
            values = locals()
        elif kind == "dual_release":
            def open_both():
                for _ in range(18):
                    self.env.hold_arms({"_L": 1.0, "_R": 1.0}, sub_steps=8)
                    self.env.record_frame("overview")
            def detach_both():
                if not all(a._grip_cmd >= .5 for a in self.env.arms):
                    raise RuntimeError("both grippers must be open before detaching")
                for i, arm in enumerate(self.env.arms):
                    arm._set_weld("tray", False)
                    self.scout.events.append({"event": "generated_api_dual_tray_weld_release", "arm": i,
                                              "sim_time": float(self.env.d.time)})
                if any(self.scout.tray_weld_active(i) for i in (0, 1)):
                    raise RuntimeError("tray weld remained active")
            def settle(): self.scout.hold(20)
            values = locals()
        elif kind == "dual_withdraw":
            def raise_both():
                targets = []
                for arm in self.env.arms:
                    ee = arm.get_ee_pose()[0]
                    targets.append(np.array([ee[0], ee[1], ee[2] + .13]))
                return tuple(self.scout.dual_move_targets(targets, grip=1.0))
            def settle(): self.scout.hold(12)
            values = locals()
        else:
            raise ValueError(f"unsupported program kind {kind}")
        return {name: values[name] for name in DOCS[kind]}

    def evidence_ok(self):
        return self.scout.goal_now()

    def stable_goal(self):
        return self.scout.stable(lambda: self.goal <= self.state() and self.evidence_ok())
