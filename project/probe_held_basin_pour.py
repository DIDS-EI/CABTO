"""Reference: left arm pours a sphere while right arm lifts a basin to catch it.

The basin is a free rigid body with a high side handle. Both source cup and basin
use declared grasp-assist welds after normal approach/close. No object qpos,
snap, arm rehome, or success-state write is used during execution.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from probe_feasible_curriculum import Curriculum

ROOT = Path(__file__).resolve().parent


class HeldBasinPour(Curriculum):
    def __init__(self, config, out, seed=0):
        super().__init__(config, "pour", out, seed)
        self.b = self.p["held_basin"]
        self.pour_events = []
        self.result["source_hashes"][str(Path(__file__))] = __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()
        if self.result["initial_penetrations"]:
            raise ValueError("held-basin scene starts with task-body penetration")

    def basin_local(self, point):
        pos, quat = self.env.get_object_pose("basin")
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        local = rot.T @ (np.asarray(point, float) - pos)
        local[0] += self.b["basin_handle_offset"]
        local[2] += self.b["basin_handle_height"]
        return local

    def sphere_in_source(self):
        rot = self.env.d.xmat[self.env.obj_bid["canL"]].reshape(3, 3)
        local = rot.T @ (self.pos("ballL") - self.pos("canL"))
        local[0] += self.p["handle_offset"]
        local[2] += self.p["handle_height"]
        r = self.p["ball_radius"]
        return bool(np.linalg.norm(local[:2]) + r < self.p["can_inner_radius"] + .001
                    and local[2] - r >= self.p["can_floor"] - .001
                    and local[2] + r < self.p["can_height"] + .002)

    def sphere_in_basin(self):
        local = self.basin_local(self.pos("ballL"))
        r = self.p["ball_radius"]
        return bool(np.linalg.norm(local[:2]) + r < self.b["basin_inner_radius"] - .003
                    and local[2] - r >= self.b["basin_floor"] - .001
                    and local[2] + r < self.b["basin_height"] + .002)

    def basin_center(self):
        pos, quat = self.env.get_object_pose("basin")
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        return pos + rot @ np.array([-self.b["basin_handle_offset"], 0.0,
                                      -self.b["basin_handle_height"]])

    def basin_ready(self):
        center = self.basin_center()
        quat = self.env.get_object_pose("basin")[1]
        angle = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).magnitude()
        return bool(self.arms[1].is_holding("basin")
                    and np.linalg.norm(center[:2] - self.b["receive_center_xy"]) < .025
                    and abs(center[2] - self.b["receive_bottom_z"]) < .025
                    and angle < math.radians(10))

    def pour(self):
        if not self.p.get("held_basin"):
            return super().pour()
        self.result["task"] = "held_basin_pour"
        self.result["initial_goal_false"] = self.sphere_in_source() and not self.sphere_in_basin()
        if not self.result["initial_goal_false"]:
            raise ValueError("invalid initial state: sphere must start only in source cup")

        self.pick(1, "basin", self.pos("basin")[2] + .015)
        basin_handle_xy = np.asarray(self.b["receive_center_xy"]) + np.array([self.b["basin_handle_offset"], 0.0])
        self.transport(1, "basin", basin_handle_xy, z=self.b["receiver_transport_z"])
        self.hold(12)
        self.mark("basin_held_ready", ready=self.basin_ready(), basin_center=self.basin_center().tolist())
        if not self.basin_ready():
            raise RuntimeError("receiver arm did not hold basin at catch pose")

        self.pick(0, "canL", self.pos("canL")[2] + .015)
        source_handle_xy = np.asarray(self.b["receive_center_xy"]) + np.asarray(self.b["source_handle_offset_xy"])
        self.transport(0, "canL", source_handle_xy, z=self.b["source_transport_z"])
        # The loaded receiving arm sags while the pouring arm traverses. Re-align
        # the held basin from its measured pose; this commands the arm only.
        self.transport(1, "basin", basin_handle_xy, z=self.b["receiver_transport_z"])
        self.hold(8)
        before = self.sphere_in_source()
        event = {"before_in_source": before, "before_in_basin": self.sphere_in_basin(),
                 "basin_ready_before_tip": self.basin_ready(),
                 "source_position": self.pos("canL").tolist(),
                 "basin_center": self.basin_center().tolist(),
                 "sphere_position": self.pos("ballL").tolist()}
        self.pour_events.append(event);self.result["pour_events"]=self.pour_events
        self.mark("pre_tip_evidence",event=event)
        if not before or event["before_in_basin"] or not event["basin_ready_before_tip"]:
            raise RuntimeError("invalid pre-tip evidence")
        ok, predicted = self.skills[0].tip_pour_joint(
            axis=self.p["tilt_axis"], total_angle=self.p["tilt_radians"], n=90,
            hold=110, can_name="canL")
        q = self.env.get_object_pose("canL")[1]
        normal=Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0,0,1])
        event.update(predicted_tilt_ok=bool(ok), predicted_axis=float(predicted),
                     actual_source_normal=normal.tolist(), actual_tipped_past_horizontal=bool(normal[2]<0),
                     after_in_source=self.sphere_in_source(), after_in_basin=self.sphere_in_basin(),
                     basin_still_held=self.arms[1].is_holding("basin"))
        self.mark("source_tipped_into_held_basin", event=event,
                  sphere_local_in_basin=self.basin_local(self.pos("ballL")).tolist())
        if event["after_in_source"] or not event["actual_tipped_past_horizontal"] \
                or not event["after_in_basin"] or not event["basin_still_held"]:
            raise RuntimeError("sphere transfer into the held basin lacks physical transition evidence")

        def goal():
            return self.sphere_in_basin() and self.basin_ready() and bool(self.pour_events and self.pour_events[-1].get("after_in_basin"))
        self.result["success"] = self.stable(goal)
        self.result.update(pour_events=self.pour_events, final_basin_center=self.basin_center().tolist(),
                           final_sphere_local=self.basin_local(self.pos("ballL")).tolist(),
                           scope="one sphere proxy poured into a basin continuously held by the other arm; oracle+weld assisted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "scenes/held_basin_pour_v1.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    HeldBasinPour(json.loads(args.config.read_text()), args.output, args.seed).run()


if __name__ == "__main__":
    main()
