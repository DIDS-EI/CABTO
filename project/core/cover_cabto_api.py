"""Real fine-grained motion/perception APIs; no pick/place macro, no no-op adapters.
The caller, never the generated program, owns independent symbolic observation.
"""
from copy import deepcopy
import inspect
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from core.single_arm_rebuilt import RebuiltSingleArm
from core.cover_cabto_grounding import PAIRS, CP, parse_json, save

API_DOCS = {
    "pick": {
        "grasp_point": "api.grasp_point() -> (x,y,z). Read-only payload grasp-center estimate in world metres. Backend is selected externally (oracle or Qwen). No movement.",
        "open_gripper": "api.open_gripper() -> None. Command gripper open and wait; does NOT move arm.",
        "approach": "api.approach(point) -> None. Move OPEN end effector above the point at safe transport height. One xyz tuple argument. Does NOT descend or close.",
        "descend": "api.descend(point) -> None. Move OPEN end effector vertically to this grasp xyz. Requires already above it. Does NOT close or lift.",
        "close_gripper": "api.close_gripper() -> None. Close at CURRENT location and wait. Requires simultaneous two-finger force. Does NOT approach, descend or lift.",
        "lift": "api.lift() -> None. Raise CLOSED end effector vertically to transport height. Does NOT close. The grasped payload must move upward.",
    },
    "place": {
        "destination_point": "api.destination_point() -> (x,y,z). Read-only desired PAYLOAD ORIGIN at its matching support; never payload's own location. No movement.",
        "carry": "api.carry(point) -> None. Move CLOSED gripper carrying payload horizontally above target; does NOT lower, open or retreat. One xyz tuple.",
        "lower": "api.lower(point) -> None. Lower CLOSED gripper so payload origin reaches target xyz. Does NOT release.",
        "open_gripper": "api.open_gripper() -> None. Command fingers open and settle at CURRENT location. Does NOT retreat. Held payload must already have been lowered to its support.",
        "withdraw": "api.withdraw() -> None. Move OPEN gripper vertically away to safe height and settle. Does NOT release the payload itself.",
    },
    "home": {
        "return_home": "api.return_home() -> None. Interpolate the EMPTY OPEN arm through a safe waypoint to its HOME joint posture and settle; refuses holding. Does not grasp/place objects.",
    },
}


def project(env, point, cam, size):
    ci = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    rot = env.d.cam_xmat[ci].reshape(3, 3)
    local = rot.T @ (np.asarray(point) - env.d.cam_xpos[ci])
    f = (size / 2) / np.tan(np.deg2rad(env.m.cam_fovy[ci]) / 2)
    if local[2] >= 0: raise ValueError("Point behind camera")
    return np.array([size/2 + f*local[0]/(-local[2]), size/2 - f*local[1]/(-local[2])])


def intersect_plane(env, uv, cam, size, z):
    ci = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    f = (size / 2) / np.tan(np.deg2rad(env.m.cam_fovy[ci]) / 2)
    direction = env.d.cam_xmat[ci].reshape(3, 3) @ np.array([(uv[0]-size/2)/f, -(uv[1]-size/2)/f, -1.])
    origin = env.d.cam_xpos[ci].copy()
    if abs(direction[2]) < 1e-8: raise ValueError("Ray parallel to calibrated plane")
    t = (z - origin[2])/direction[2]
    if t <= 0: raise ValueError("Plane behind camera")
    return origin + t*direction


class Perception:
    def __init__(self, runtime, backend, out, client=None):
        self.r = runtime; self.backend = backend; self.out = Path(out); self.client = client; self.records = []
        self.static_cache = {}

    def prepare_static_supports(self, names):
        """Observe immutable supports BEFORE holding objects can occlude them.
        Stores only the Qwen-derived xy, never the scene's ground-truth xy.
        """
        for name in names:
            z = self.r.config["table_top"] + self.r.config["support_surface_z"][name]
            self.static_cache[name] = self.locate(name, z, "support_floor_center").copy()

    @staticmethod
    def parse_pixel(call, maximum):
        parsed = parse_json(call["raw"])
        field = next((k for k in ("point", "point_2d") if k in parsed), None)
        if field is None: raise ValueError("No point or point_2d field")
        xy = parsed[field]
        if not isinstance(xy, list) or len(xy) != 2: raise ValueError("Malformed Qwen point")
        xy = np.asarray(xy, float)
        if not np.isfinite(xy).all() or (xy < 0).any() or (xy > maximum).any():
            raise ValueError("Qwen point outside configured coordinate scale")
        return xy, field

    def locate(self, name, z, kind):
        # Known object grasp / support height prior is explicit, shared by oracle
        # and Qwen trials. Qwen xy comes ONLY from image rays, never GT correction.
        if name in self.static_cache and kind == "support_floor_center":
            value = self.static_cache[name].copy()
            self.records.append({"backend": self.backend, "object": name, "point_kind": kind,
                                 "cache": "Qwen_pre_motion_static_support", "xyz": value.tolist(), "fallback": False})
            return value
        true_xy = self.r.pos(name)[:2]
        record = {"backend": self.backend, "object": name, "point_kind": kind, "height_prior_m": float(z),
                  "height_source": ("oracle_initial_object_z_plus_frozen_grasp_offset" if kind == "grasp_center" else "frozen_scene_support_surface_z"), "fallback": False}
        if self.backend == "oracle":
            p = np.r_[true_xy, z]; record["xyz"] = p.tolist(); self.records.append(record); return p
        if self.client is None: raise ValueError("Qwen backend requires an actual loaded visual model")
        desc = {"shrimp": "orange shrimp at the front of the table", "apple": "red apple at the front of the table", "potato": "brown potato at the front of the table", "bowl": "center of the inside floor of the white bowl", "board": "center of the wooden chopping board", "pan": "center of the inside floor of the black cooking pan"}[name]
        estimates = []
        record["views"] = []
        index = len(self.records)
        self.records.append(record)
        try:
            for cam in ("overview", "front"):
                image = self.r.env.render_cam(cam); size = image.shape[0]
                path = self.out / f"point_{index:03d}_{name}_{cam}.png"
                path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(image).save(path)
                absolute = "Qwen2.5-VL" in self.client.model_id
                contract = "absolute_pixel" if absolute else "normalized_0_1000"
                text = (f"Point to the {desc}. Output the pixel coordinate." if absolute else
                        f"Locate the {desc}. Return ONLY a JSON object {{\"point\":[x,y]}}. Coordinates MUST be normalized from 0 to 1000, x left to right, y top to bottom. Select the center, not the robot.")
                call = self.client.call(text, [path], max_tokens=180)
                view = {"camera":cam,"image_size":size,"call":call,"coordinate_contract":contract,
                        "contract_source":"fixed model-family convention, checked with current-scene diagnostic; no per-point GT choice"}
                record["views"].append(view)
                xy, field = self.parse_pixel(call, size if absolute else 1000)
                view["parsed_field"] = field
                uv = xy if absolute else xy * size / 1000
                if self.backend == "qwen_zoom":
                    if not absolute: raise ValueError("Zoom protocol requires calibrated Qwen2.5-VL pixel convention")
                    view["coarse_uv"] = uv.tolist()
                    # Crop centered on Qwen prediction, not ground truth. Border
                    # shifting retains a square ROI; world target is not clamped.
                    side = min(192, size)
                    left = int(np.clip(round(uv[0]-side/2), 0, size-side))
                    top = int(np.clip(round(uv[1]-side/2), 0, size-side))
                    zoom = Image.fromarray(image).crop((left,top,left+side,top+side)).resize((576,576))
                    crop_path = path.with_name(path.stem + "_crop.png"); zoom.save(crop_path)
                    query = f"Point to the {desc}. Output the pixel coordinate."
                    refined = self.client.call(query, [crop_path], max_tokens=180)
                    view["refine_call"] = refined; view["crop_bounds"] = [left,top,side,side]; view["crop_output_size"] = 576
                    local_uv, _ = self.parse_pixel(refined, 576)
                    uv = np.array([left,top]) + local_uv * side/576
                estimate = intersect_plane(self.r.env, uv, cam, size, z)
                gt_pixel = project(self.r.env, np.r_[true_xy, z], cam, size)
                view.update({"uv": uv.tolist(), "xyz": estimate.tolist(),
                        "evaluation_only_gt_uv": gt_pixel.tolist(), "evaluation_only_pixel_error": float(np.linalg.norm(uv-gt_pixel)),
                        "evaluation_time": "before any movement"})
                estimates.append(estimate)
                draw_image = Image.fromarray(image); draw = ImageDraw.Draw(draw_image)
                u, v = uv; draw.line((u-10,v,u+10,v), fill=(255,90,90), width=3); draw.line((u,v-10,u,v+10), fill=(255,90,90), width=3)
                u, v = gt_pixel; draw.ellipse((u-7,v-7,u+7,v+7), outline=(80,230,170), width=2)
                draw_image.save(path.with_name(path.stem + "_overlay.png"))
            disagreement = float(np.linalg.norm(estimates[0][:2]-estimates[1][:2]))
            record["view_disagreement_m"] = disagreement
            if disagreement > .025: raise ValueError(f"Cross-view xy disagreement {disagreement:.3f}m exceeds 0.025m; no oracle fallback")
            p = np.mean(estimates, axis=0)
            if not (.05 < p[0] < .9 and -.36 < p[1] < .36): raise ValueError("Perceived point outside safe workspace; reject, do not clamp")
            record["xyz"] = p.tolist(); record["evaluation_only_xy_error_m"] = float(np.linalg.norm(p[:2]-true_xy))
            return p
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            save(self.out / "perception.json", self.records)


class FineCover:
    def __init__(self, config, seed=0, render=False, perception="oracle", out=".", client=None):
        self.r = RebuiltSingleArm(config, seed=seed, render=render)
        self.perception = Perception(self.r, perception, out, client)
        self.binding = None

    def state(self):
        r = self.r; s = set()
        if r.released(): s.add("empty()")
        for obj, support in PAIRS:
            lifted = r.pos(obj)[2] > r.initial[obj][2] + .075
            if r.held(obj) and lifted:
                s.add(f"holding({obj})")
            elif r.assess_one((obj,support))["ok"]:
                s.add(f"at_target({obj})")
            elif np.linalg.norm(r.pos(obj)-r.initial[obj]) < .009:
                s.add(f"at_source({obj})")
        if r.assess()["final_home"]: s.add("home_completed()")
        return s

    def bind(self, model):
        self.binding = deepcopy(model)
        r = self.r
        self.obj = model["args"].get("object"); self.support = model["args"].get("support")
        if model["name"] == "pick":
            r.target = self.obj; r.current_support = "table"
        elif model["name"] == "place":
            if not r.held(self.obj): raise RuntimeError("Place precondition not physically established")
            r.current_support = self.support
        return {name: getattr(self, name) for name in API_DOCS[model["name"]]}

    def point_check(self, p):
        p = np.asarray(p, float)
        if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError("Expected finite xyz tuple")
        if not (.05 < p[0] < .9 and -.36 < p[1] < .36 and .39 < p[2] < .85): raise ValueError("Waypoint outside scene workspace")
        return p

    def grasp_point(self):
        z = self.r.initial[self.obj][2] + self.r.config["grip_offset_z"][self.obj]
        return tuple(float(v) for v in self.perception.locate(self.obj, z, "grasp_center"))

    def destination_point(self):
        r = self.r
        surface_z = r.config["table_top"] + r.config["support_surface_z"][self.support]
        p = self.perception.locate(self.support, surface_z, "support_floor_center")
        p[2] = r.config["table_top"] + r.config["placement_origin_z"][self.obj]
        return tuple(float(v) for v in p)

    def open_gripper(self):
        r = self.r
        if self.binding["name"] == "pick":
            r.mark("approach_" + self.obj); r.wait(.16, 1.)
        else:
            r.transport_monitor = False; r.mark("release_" + self.obj)
            r.wait(.16,1.); r.arm.release_all(); r.wait(.22,1.)
            if not r.released(): raise RuntimeError("Gripper not released")

    def approach(self, point):
        r = self.r; p = self.point_check(point)
        r.mark("approach_" + self.obj)
        r.move([*p[:2],r.c["transport_z"]],1.)

    def descend(self, point):
        r = self.r; p = self.point_check(point)
        if np.linalg.norm(r.ee()[:2]-p[:2]) > .015: raise RuntimeError("Must approach above point before descent")
        r.mark("descend_" + self.obj); r.move(p,1.,r.c["lower_speed_m_s"])

    def close_gripper(self):
        r = self.r; r.mark("grasp_" + self.obj)
        r.wait(.35,0.)
        if min(r.force_now.values()) < .05: raise RuntimeError(f"No simultaneous two-finger force at closure: {r.force_now}")
        if np.linalg.norm(r.ee()-r.grip_point(self.obj)) > .02: raise RuntimeError("Grasp geometry inconsistent")
        if r.active_welds(): raise RuntimeError("Strict scene must never activate a weld")
        r.events.append({"event":"bilateral_grasp_contact","object":self.obj,"normal_forces_N":dict(r.force_now),"time":float(r.env.d.time)})
        r.wait(.10,0.)
        r.grasp_relative = r.env.d.xmat[r.arm.hand_bid].reshape(3,3).T @ (r.pos(self.obj)-r.ee())
        r.transport_monitor = True; r.current_contact_gap = 0.

    def lift(self):
        r = self.r
        if r.arm._grip_cmd >= .5: raise RuntimeError("Lift requires closed gripper; close_gripper was omitted")
        r.mark("lift_" + self.obj); r.move([*r.ee()[:2],r.c["transport_z"]],0.,r.c["lift_speed_m_s"])
        if r.pos(self.obj)[2] < r.initial[self.obj][2]+.075: raise RuntimeError("Payload did not lift")

    def carry(self, point):
        r = self.r; p = self.point_check(point)
        if r.arm._grip_cmd >= .5: raise RuntimeError("Carry requires closed gripper")
        r.mark("carry_" + self.obj); p = p.copy(); p[2] = r.pos(self.obj)[2]
        r.move_object(self.obj,p)

    def lower(self, point):
        r = self.r; p = self.point_check(point)
        if np.linalg.norm(r.pos(self.obj)[:2]-p[:2]) > .015: raise RuntimeError("Must carry above destination before lowering")
        r.mark("lower_" + self.obj); r.move_object(self.obj,p,r.c["lower_speed_m_s"])

    def withdraw(self):
        r = self.r
        if not r.released(): raise RuntimeError("Withdraw requires explicit release first")
        r.mark("withdraw_" + self.obj); r.move([*r.ee()[:2],r.c["transport_z"]],1.,r.c["lift_speed_m_s"])
        r.target = None; r.wait(.10,1.)
        if r.assess_one((self.obj,self.support))["ok"] and (self.obj,self.support) not in r.completed:
            r.completed.append((self.obj,self.support))

    def return_home(self):
        self.r.home()

    def final_check(self, goal):
        r = self.r; metrics = r.assess()
        reached = set(goal) <= self.state()
        # Partial tasks must leave non-goal objects at source, not falsely demand
        # all three target predicates. Mechanical constraints remain identical.
        mechanical = all([metrics["released"],metrics["final_home"],metrics["completed_targets_preserved"],
                          metrics["grasp_verified"],metrics["motion_verified"],not metrics["forbidden_contacts"]])
        return {"success": bool(reached and mechanical and r.stability_ok), "goal_reached": reached,
                "mechanical_ok": bool(mechanical), "metrics": metrics, "observed_state": sorted(self.state())}

    def close(self): self.r.close()


def transition_check(model, before, after, exception=None):
    pre, add, delete = (set(model[k]) for k in ("pre","add","del"))
    predicted = (set(before) | add) - delete
    violations = {"missing_pre": sorted(pre-set(before)), "missing_add": sorted(add-set(after)),
                  "remaining_del": sorted(delete & set(after)), "missing_preserved_pre": sorted((pre-delete)-set(after)),
                  "unexpected_added": sorted(set(after)-predicted), "unexpected_removed": sorted(predicted-set(after))}
    return {"valid_trial": pre <= set(before), "paper_line20": (pre|add)-delete <= set(after),
            "strict_CP_transition_match": set(after)==predicted, "goal_ok": add <= set(after),
            "effect_ok": exception is None and not any(violations.values()), "violations": violations}
