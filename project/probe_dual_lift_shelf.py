"""Reference baseline for cooperative dual-arm tray lifting and shelf placement.

The run first packs two free objects into the free tray using the established
reference controllers. Both arms then grasp rigid side handles, lift together,
translate, lower onto a fixed shelf, release, retreat vertically, and park.
No object qpos is written during execution and no arm is rehomed. The two tray
welds are explicit grasp assistance and are activated only after both hands have
physically approached the configured handles.
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from probe_feasible_curriculum import containment
from probe_free_tray_packing import FreeTrayScout

ROOT = Path(__file__).resolve().parent


class DualLiftShelfScout(FreeTrayScout):
    def handle_world(self, i):
        pos, quat = self.tray_pose()
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        side = 1.0 if i == 0 else -1.0
        local = np.array([0.0, side * self.p["lift_handle_y"], self.p["lift_handle_z"]])
        return pos + rot @ local

    def items_contained(self):
        pos, quat = self.tray_pose()
        checks = {}
        for i in (0, 1):
            checks[f"item{i}"] = containment(
                self.env.get_object_pose(f"item{i}"), (pos, quat),
                self.p["item_half_size"], self.p["tray_inner_half_size"],
                floor=self.p["tray_wall"], ceiling=self.p["tray_height"])
        return checks

    def tray_supported_by_shelf(self):
        pair = {"tray", "shelf"}
        for contact in self.env.d.contact:
            b1 = int(self.env.m.geom_bodyid[contact.geom1])
            b2 = int(self.env.m.geom_bodyid[contact.geom2])
            n1 = mujoco.mj_id2name(self.env.m, mujoco.mjtObj.mjOBJ_BODY, b1)
            n2 = mujoco.mj_id2name(self.env.m, mujoco.mjtObj.mjOBJ_BODY, b2)
            if {n1, n2} == pair and contact.dist <= 0.002:
                return True
        return False

    def tray_weld_active(self, i):
        wid = self.arms[i].obj_weld["tray"]
        return bool(wid >= 0 and self.env.d.eq_active[wid] == 1)

    def tray_footprint_on_shelf(self):
        pos, quat = self.tray_pose()
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        hx = self.p["tray_inner_half_size"][0] + self.p["tray_wall"]
        hy = max(self.p["tray_inner_half_size"][1] + self.p["tray_wall"],
                 self.p["lift_handle_y"] + 0.010)
        shelf = np.asarray(self.p["shelf_xy"], float)
        limit = np.asarray(self.p["shelf_half_size_xy"], float) - 0.004
        corners = [pos + rot @ np.array([sx * hx, sy * hy, 0.0])
                   for sx in (-1, 1) for sy in (-1, 1)]
        return all(np.all(np.abs(c[:2] - shelf) <= limit) for c in corners)

    def tray_on_shelf(self):
        pos, quat = self.tray_pose()
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        xy_error = float(np.linalg.norm(pos[:2] - np.asarray(self.p["shelf_xy"])))
        rotation_error = float(np.linalg.norm(rot.as_rotvec()))
        detached = not any(self.tray_weld_active(i) for i in (0, 1))
        hands_open = all(a._grip_cmd >= 0.5 for a in self.arms)
        return bool(
            xy_error < self.p["shelf_position_tolerance_m"]
            and abs(pos[2] - self.p["shelf_top_z"]) < 0.012
            and rotation_error < math.radians(self.p["shelf_tilt_tolerance_deg"])
            and detached and hands_open
            and self.tray_footprint_on_shelf()
            and self.tray_supported_by_shelf()
        )

    def goal_now(self):
        checks = self.items_contained()
        return self.tray_on_shelf() and all(v["inside"] for v in checks.values()) \
            and all(self.released(f"item{i}") for i in (0, 1))

    def dual_move_targets(self, targets, grip, max_steps=180, tol=0.012):
        targets = [np.asarray(t, float) for t in targets]
        saved = [a.bias_world.copy() for a in self.arms]
        for a in self.arms:
            a.bias_world = np.zeros(3)
        best = float("inf")
        try:
            for step in range(max_steps):
                errors = [targets[i] - self.arms[i].get_ee_pose()[0] for i in (0, 1)]
                worst = max(float(np.linalg.norm(e)) for e in errors)
                best = min(best, worst)
                if worst < tol:
                    self.hold(3)
                    break
                actions = {}
                for i, arm in enumerate(self.arms):
                    action = np.zeros(7)
                    action[:3] = np.clip(errors[i], -0.018, 0.018)
                    action[6] = grip
                    actions[arm.s] = action
                self.env.step_arms(actions, sub_steps=8)
                if step % 2 == 0:
                    self.env.record_frame("overview")
        finally:
            for a, bias in zip(self.arms, saved):
                a.bias_world = bias
        final = [float(np.linalg.norm(targets[i] - self.arms[i].get_ee_pose()[0])) for i in (0, 1)]
        if max(final) > 0.030:
            raise RuntimeError(f"a cooperative arm missed its target: residuals={final}; no state correction")
        return final

    def dual_translate_tray(self, target_base, phase):
        target_base = np.asarray(target_base, float)
        for _ in range(3):
            tray = self.tray_pose()[0]
            delta = target_base - tray
            targets = [a.get_ee_pose()[0] + delta for a in self.arms]
            residuals = self.dual_move_targets(targets, grip=0.0)
            if np.linalg.norm(self.tray_pose()[0] - target_base) < 0.018:
                break
        contained = self.items_contained()
        self.mark(phase, target_tray_base=target_base.tolist(), residuals_m=residuals,
                  tray_error_m=float(np.linalg.norm(self.tray_pose()[0] - target_base)),
                  containment={k: bool(v["inside"]) for k, v in contained.items()})
        if not all(v["inside"] for v in contained.values()):
            raise RuntimeError(f"an item left the tray during {phase}")

    def dual_grasp_handles(self):
        handles = [self.handle_world(i) for i in (0, 1)]
        high = [np.array([h[0], h[1], self.h + 0.29]) for h in handles]
        self.dual_move_targets(high, grip=1.0)
        self.dual_move_targets(handles, grip=1.0, max_steps=220)
        tcp_error = [float(np.linalg.norm(self.arms[i].get_ee_pose()[0] - handles[i])) for i in (0, 1)]
        finger_error = [float(np.linalg.norm(self.arms[i]._finger_mid() - handles[i])) for i in (0, 1)]
        if max(tcp_error) > 0.045 or max(finger_error) > 0.12:
            raise RuntimeError(f"hands did not reach tray handles: tcp={tcp_error}, finger={finger_error}")
        # Prevent the legacy nearest-object helper from attaching an already packed item.
        for a in self.arms:
            a._grasp_enabled = False
        for _ in range(24):
            self.env.hold_arms({"_L": 0.0, "_R": 0.0}, sub_steps=8)
            self.env.record_frame("overview")
        for i, a in enumerate(self.arms):
            a._set_weld("tray", True)
            self.events.append({"event": "explicit_dual_tray_grasp_weld", "arm": i,
                                "object": "tray", "sim_time": float(self.env.d.time),
                                "handle_world": handles[i].tolist(),
                                "finger_error_m": finger_error[i]})
        self.hold(12)
        if not all(a.is_holding("tray") for a in self.arms):
            raise RuntimeError("both tray grasp-assist welds did not become active")
        self.mark("dual_grasped", tcp_error_m=tcp_error, finger_error_m=finger_error)

    def dual_release_and_withdraw(self):
        for _ in range(18):
            self.env.hold_arms({"_L": 1.0, "_R": 1.0}, sub_steps=8)
            self.env.record_frame("overview")
        for i, a in enumerate(self.arms):
            a._set_weld("tray", False)
            self.events.append({"event": "dual_tray_weld_release", "arm": i,
                                "object": "tray", "sim_time": float(self.env.d.time)})
        self.hold(16)
        if any(self.tray_weld_active(i) for i in (0, 1)):
            raise RuntimeError("a tray equality weld remained active after release")
        if not all(a._grip_cmd >= 0.5 for a in self.arms):
            raise RuntimeError("a gripper remained closed after tray release")
        raised = []
        for a in self.arms:
            ee = a.get_ee_pose()[0]
            raised.append(np.array([ee[0], ee[1], ee[2] + 0.13]))
        self.dual_move_targets(raised, grip=1.0)
        parks = [np.array([self.config["robots"][0]["base_xy"][0], 0.28, self.h + 0.32]),
                 np.array([self.config["robots"][1]["base_xy"][0], -0.28, self.h + 0.32])]
        self.dual_move_targets(parks, grip=1.0, max_steps=220)
        self.mark("released_withdrawn_parked")

    def packing(self):
        initial_shelf_goal = self.goal_now()
        if initial_shelf_goal:
            raise ValueError("invalid trial: shelf transport goal already true")
        super().packing()
        packed_ok = bool(self.result["success"])
        self.result["packing_success_before_lift"] = packed_ok
        self.result["success"] = False
        if not packed_ok:
            raise RuntimeError("packing prerequisite failed")
        pre = self.items_contained()
        if not all(v["inside"] for v in pre.values()):
            raise RuntimeError("items are not contained before cooperative lifting")
        self.mark("packing_complete_before_lift", containment={k: bool(v["inside"]) for k, v in pre.items()})

        self.dual_grasp_handles()
        start = self.tray_pose()[0].copy()
        self.dual_translate_tray([start[0], start[1], self.p["carry_tray_base_z"]], "dual_lifted")
        self.dual_translate_tray([*self.p["shelf_xy"], self.p["carry_tray_base_z"]], "dual_translated")
        self.dual_translate_tray([*self.p["shelf_xy"], self.p["shelf_top_z"]], "dual_lowered")
        self.dual_release_and_withdraw()

        self.result["success"] = self.stable(self.goal_now)
        pos, quat = self.tray_pose()
        checks = self.items_contained()
        self.result.update(
            final_tray_position=pos.tolist(), final_tray_quaternion_wxyz=quat.tolist(),
            shelf_xy_error_mm=float(np.linalg.norm(pos[:2] - self.p["shelf_xy"]) * 1000),
            shelf_height_error_mm=float(abs(pos[2] - self.p["shelf_top_z"]) * 1000),
            final_containment=checks, shelf_contact=self.tray_supported_by_shelf(),
            scope="cooperative dual-side tray lift and shelf placement; oracle + two grasp welds; no object snap or arm rehome")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "scenes/packing_dual_lift_shelf_storage_v3.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    scout = DualLiftShelfScout(json.loads(args.config.read_text()), "packing", args.output, args.seed)
    scout.run()


if __name__ == "__main__":
    main()
