"""Collision-audited airborne handover with contact-gated grasp assistance.

The task uses one rigid baton with two spatially separated handles.  Both arms
approach the physical handles; a declared grasp weld is activated only after the
corresponding closed gripper is geometrically close and has made handle contact.
No object pose, object velocity, success flag, or robot qpos is written during
execution.  The weld remains explicit assistance rather than a strict grasp.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT.parent / "CABTO/exp4_bt_tasks/stage2_scripted"
sys.path.insert(0, str(STAGE2))
from arm_skills import ArmSkills
from exp4_env import Exp4Env, HOME_QPOS
import scene_dual_common as dc
from probe_five_scene_preflight import camera_params, geom_corners


DOCS = {
    "donor_pick": {"pick_donor": "api.pick_donor(): zero arguments, returns None. Required exact policy body: api.pick_donor(). Call once; never assign its return or call it again."},
    "present": {"move_to_handover_pose": "api.move_to_handover_pose(): zero arguments. It moves the donor-held baton to the configured airborne handover pose and holds there. Call this one method exactly once."},
    "receive": {"grasp_opposite_handle": "api.grasp_opposite_handle(): zero arguments. It performs the complete receiver-side approach, physical handle contact check, gripper close, and checked assisted attachment while the donor still holds. Call this method exactly once."},
    "donor_release": {"finalize_transfer": "api.finalize_transfer(): zero arguments. It checks measured receiver hold, releases the donor, and withdraws the empty donor hand. Call this one method exactly once; do not add conditions, loops, observations, variables, or other calls."},
    "receiver_place": {"place_baton": "api.place_baton(): zero arguments, returns None. Required exact policy body: api.place_baton(). Call once; never assign its return or call it again."},
    "controlled_home": {"return_arm_home": "api.return_arm_home(): zero arguments. It moves the already-empty bound arm through its outer staging pose and performs the actuator interpolation to HOME. Call this one method exactly once."},
}


def action(name, kind, arm, pre, add, delete, description):
    return {"name": name, "args": {"arm": str(arm), "obj": "baton", "dest": name},
            "program_kind": kind, "pre": list(pre), "add": list(add),
            "del": list(delete), "description": description}


def domain():
    safe = ["collision_free()"]
    return [
        action("pick_baton_by_donor", "donor_pick", 0,
               ["empty(0)", "empty(1)", "at_source(baton)"] + safe,
               ["donor_holding(baton)", "baton_airborne()"],
               ["empty(0)", "at_source(baton)", "home(0)"],
               "Pick the baton by its donor-side handle and lift it without moving the receiver."),
        action("present_baton_in_air", "present", 0,
               ["donor_holding(baton)", "empty(1)", "baton_airborne()"] + safe,
               ["at_handover_pose(baton)"], [],
               "The only allowed implementation is one call to api.move_to_handover_pose()."),
        action("receiver_grasps_opposite_handle", "receive", 1,
               ["donor_holding(baton)", "empty(1)", "at_handover_pose(baton)"] + safe,
               ["dual_holding(baton)", "receiver_contact_verified()"],
               ["empty(1)", "home(1)"],
               "Close the receiver on the opposite physical handle while the donor still holds."),
        action("donor_releases_after_receiver", "donor_release", 0,
               ["dual_holding(baton)", "receiver_contact_verified()", "baton_airborne()"] + safe,
               ["receiver_holding(baton)", "empty(0)", "donor_withdrawn()", "handover_verified()"],
               ["dual_holding(baton)", "donor_holding(baton)"],
               "The only allowed implementation is one call to api.finalize_transfer(); it already verifies receiver hold before releasing and withdrawing the donor."),
        action("home_donor_arm", "controlled_home", 0,
               ["receiver_holding(baton)", "empty(0)", "donor_withdrawn()", "handover_verified()"] + safe,
               ["home(0)"], [],
               "The only allowed implementation is one call to api.return_arm_home()."),
        action("receiver_places_baton", "receiver_place", 1,
               ["receiver_holding(baton)", "home(0)", "handover_verified()"] + safe,
               ["at_destination(baton)", "empty(1)", "receiver_withdrawn()"],
               ["receiver_holding(baton)", "baton_airborne()", "at_handover_pose(baton)"],
               "Carry the baton to the destination, place and release it, then withdraw."),
        action("home_receiver_arm", "controlled_home", 1,
               ["at_destination(baton)", "empty(1)", "receiver_withdrawn()", "home(0)", "handover_verified()"] + safe,
               ["home(1)"], [],
               "The only allowed implementation is one call to api.return_arm_home()."),
    ]


def goal():
    return {"at_destination(baton)", "empty(0)", "empty(1)", "home(0)", "home(1)",
            "handover_verified()", "collision_free()"}


def _vec(values):
    return " ".join(str(float(v)) for v in values)


def build_scene(config):
    p = config["handover"]; h = config["table"]["top_z"]
    visual = config.get("visual_style", {}); tea_style = visual.get("profile") == "green_tea_handover"
    half = np.asarray(p["object_half_size"], float)
    hh = np.asarray(p["handle_half_size"], float)
    oy = float(p["handle_offset_y"]); hz = float(p["handle_height"])
    xy = p["start_center_xy"]
    z = h + half[2] + .002
    # The central rail rests flat. Two elevated handle pads are rigidly joined to
    # it and keep the two downward grippers separated during overlap.
    core_mat = "tea_green" if tea_style else "mat_yellow"
    handle_mat = "tea_green_dark" if tea_style else "mat_green"
    other_handle_mat = "tea_green_dark" if tea_style else "mat_blue"
    geoms = [
        f'<geom name="baton_core" type="box" size="{_vec(half)}" material="{core_mat}" mass="0.09" friction="0.9 0.02 0.001"/>',
        f'<geom name="baton_handle_L" type="box" size="{_vec(hh)}" pos="0 {oy} {hz}" material="{handle_mat}" mass="0.02" friction="1.2 0.02 0.001"/>',
        f'<geom name="baton_handle_R" type="box" size="{_vec(hh)}" pos="0 {-oy} {hz}" material="{other_handle_mat}" mass="0.02" friction="1.2 0.02 0.001"/>',
        f'<geom name="baton_stem_L" type="box" size="0.012 0.012 {hz/2}" pos="0 {oy} {hz/2}" material="{handle_mat}" mass="0.01"/>',
        f'<geom name="baton_stem_R" type="box" size="0.012 0.012 {hz/2}" pos="0 {-oy} {hz/2}" material="{other_handle_mat}" mass="0.01"/>',
    ]
    if tea_style:
        # Packaging artwork is collisionless and massless.  The calibrated core,
        # handles and stems above remain the complete physical handover object.
        geoms += [
            f'<geom name="tea_front_panel" type="box" size="{half[0]-.001} {half[1]-.010} .0007" pos="0 0 {half[2]+.0007}" material="tea_green_light" mass="0" contype="0" conaffinity="0"/>',
            f'<geom name="tea_center_band" type="box" size="{half[0]-.0005} .020 .0009" pos="0 0 {half[2]+.0015}" material="tea_cream" mass="0" contype="0" conaffinity="0"/>',
            f'<geom name="tea_badge" type="cylinder" size=".014 .0010" pos="0 0 {half[2]+.0025}" material="tea_badge_red" mass="0" contype="0" conaffinity="0"/>',
            f'<geom name="tea_leaf_L" type="ellipsoid" size=".010 .022 .0008" pos="-.012 .058 {half[2]+.0020}" quat=".9238795 0 0 .3826834" material="tea_leaf" mass="0" contype="0" conaffinity="0"/>',
            f'<geom name="tea_leaf_R" type="ellipsoid" size=".010 .022 .0008" pos=".012 -.058 {half[2]+.0020}" quat=".9238795 0 0 -.3826834" material="tea_leaf" mass="0" contype="0" conaffinity="0"/>',
        ]
    dx, dy = p["destination_center_xy"]
    displays = []
    if tea_style:
        layout=visual.get("display_layout",{});yaw=math.radians(layout.get("yaw_deg",0.0));quat=_vec([math.cos(yaw/2),0,0,math.sin(yaw/2)])
        for index,(cx,cy) in enumerate(layout.get("centers_xy",[])):
            name=f"tea_display_{index}";cz=h+half[2]+.002
            # Every display copy has the same outer core, two elevated handles,
            # stems and package artwork as the movable task object. It is a fixed,
            # collisionless replica, so it cannot alter handover physics.
            copy_geoms=[
                f'<geom name="{name}_core" type="box" size="{_vec(half)}" material="tea_green" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_handle_L" type="box" size="{_vec(hh)}" pos="0 {oy} {hz}" material="tea_green_dark" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_handle_R" type="box" size="{_vec(hh)}" pos="0 {-oy} {hz}" material="tea_green_dark" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_stem_L" type="box" size="0.012 0.012 {hz/2}" pos="0 {oy} {hz/2}" material="tea_green_dark" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_stem_R" type="box" size="0.012 0.012 {hz/2}" pos="0 {-oy} {hz/2}" material="tea_green_dark" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_front_panel" type="box" size="{half[0]-.001} {half[1]-.010} .0007" pos="0 0 {half[2]+.0007}" material="tea_green_light" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_center_band" type="box" size="{half[0]-.0005} .020 .0009" pos="0 0 {half[2]+.0015}" material="tea_cream" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_badge" type="cylinder" size=".014 .0010" pos="0 0 {half[2]+.0025}" material="tea_badge_red" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_leaf_L" type="ellipsoid" size=".010 .022 .0008" pos="-.012 .058 {half[2]+.0020}" quat=".9238795 0 0 .3826834" material="tea_leaf" contype="0" conaffinity="0"/>',
                f'<geom name="{name}_leaf_R" type="ellipsoid" size=".010 .022 .0008" pos=".012 -.058 {half[2]+.0020}" quat=".9238795 0 0 -.3826834" material="tea_leaf" contype="0" conaffinity="0"/>',
            ]
            displays.append(f'<body name="{name}" pos="{_vec([cx,cy,cz])}" quat="{quat}">{"".join(copy_geoms)}</body>')
    obj = (f'<body name="baton" pos="{_vec([xy[0], xy[1], z])}">'
           f'<freejoint name="baton_free"/>{"".join(geoms)}</body>'
           f'<body name="handover_destination" pos="{_vec([dx, dy, h+.0005])}">'
           f'<geom type="box" size="{half[0]+.012} {half[1]+.012} .0005" rgba="0.18 0.72 0.28 .40" contype="0" conaffinity="0"/></body>'
           + "".join(displays))
    welds = ('<weld name="grasp_weld_L_baton" body1="hand_L" body2="baton" active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>'
             '<weld name="grasp_weld_R_baton" body1="hand_R" body2="baton" active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>')
    root = ET.fromstring(dc.make_dual_xml("smooth_air_handover", obj, welds))
    if tea_style:
        asset = root.find("asset")
        for name, rgba in (("tea_green","0.34 0.66 0.12 1"),("tea_green_dark","0.12 0.38 0.08 1"),
                           ("tea_green_light","0.64 0.83 0.30 1"),("tea_leaf","0.13 0.52 0.10 1"),
                           ("tea_cream","0.92 0.87 0.62 1"),("tea_badge_red","0.62 0.12 0.08 1"),
                           ("tea_table","0.72 0.74 0.75 1"),("tea_table_edge","0.30 0.32 0.34 1")):
            ET.SubElement(asset,"material",{"name":name,"rgba":rgba,"specular":"0.16","shininess":"0.20"})
        sky=asset.find("texture[@type='skybox']")
        if sky is not None:sky.set("rgb1","0.35 0.70 0.86");sky.set("rgb2","0.23 0.53 0.72")
        grid=asset.find("texture[@name='grid']")
        if grid is not None:grid.set("rgb1","0.28 0.61 0.78");grid.set("rgb2","0.34 0.69 0.84")
    table = root.find("./worldbody/body[@name='table']")
    t = config["table"]; table.set("pos", _vec([*t["center_xy"], h-t["thickness"]/2]))
    table_geom=table.find("geom");table_geom.set("size", _vec([t["full_size_xy"][0]/2, t["full_size_xy"][1]/2, t["thickness"]/2]))
    if tea_style:
        table_geom.set("material","tea_table")
        tx,ty=np.asarray(t["full_size_xy"])/2
        ET.SubElement(table,"site",{"name":"tea_table_front_edge","type":"box","size":_vec([tx,.008,.018]),"pos":_vec([0,-ty+.008,-.010]),"material":"tea_table_edge"})
        ET.SubElement(table,"site",{"name":"tea_table_back_edge","type":"box","size":_vec([tx,.008,.018]),"pos":_vec([0,ty-.008,-.010]),"material":"tea_table_edge"})
        world=root.find("worldbody")
        for xline in np.arange(-2.5,2.6,.5):
            ET.SubElement(world,"geom",{"name":f"tea_grid_x_{xline}","type":"box","size":"0.004 2.5 0.0006","pos":_vec([xline,0,.001]),"rgba":"0.92 0.98 1 0.82","contype":"0","conaffinity":"0"})
        for yline in np.arange(-2.5,2.6,.5):
            ET.SubElement(world,"geom",{"name":f"tea_grid_y_{yline}","type":"box","size":"2.5 0.004 0.0006","pos":_vec([0,yline,.001]),"rgba":"0.92 0.98 1 0.82","contype":"0","conaffinity":"0"})
        target=np.asarray(visual["camera_target"],float)
        for name,key,fovy in (("overview","camera_overview","44"),("front","camera_front","48")):
            cam=root.find(f"./worldbody/camera[@name='{name}']");pos=np.asarray(visual[key],float)
            cam.set("pos",_vec(pos));cam.set("xyaxes",_vec(camera_params(pos,target)));cam.set("fovy",fovy)
    for robot in config["robots"]:
        body = root.find(f"./worldbody/body[@name='link0{robot['suffix']}']")
        body.set("pos", _vec([*robot["base_xy"], h+robot["base_above_table"]]))
        yaw = robot["yaw"]; body.set("quat", _vec([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
    return ET.tostring(root, encoding="unicode")


def _body_name(model, geom_id):
    bid = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body#{bid}"


def _geom_name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom#{geom_id}"


def _arm_side(name):
    if name.endswith("_L"): return "L"
    if name.endswith("_R"): return "R"
    return None


class HandoverAudit:
    def __init__(self, runtime):
        self.runtime = runtime; self.phase = "initial"; self.rows = []; self.allowed = []
        self.original = mujoco.mj_step
    def install(self):
        def audited(model, data, *args, **kwargs):
            out = self.original(model, data, *args, **kwargs); self.sample(); return out
        mujoco.mj_step = audited
    def restore(self):
        mujoco.mj_step = self.original
    def sample(self):
        m, d = self.runtime.env.m, self.runtime.env.d
        for c in d.contact:
            ba, bb = _body_name(m, c.geom1), _body_name(m, c.geom2)
            ga, gb = _geom_name(m, c.geom1), _geom_name(m, c.geom2)
            sa, sb = _arm_side(ba), _arm_side(bb)
            side = sa or sb
            arm_body = ba if sa else (bb if sb else None)
            other_body = bb if sa else (ba if sb else None)
            other_geom = gb if sa else (ga if sb else None)
            correct_pair = side in ("L", "R") and arm_body in (f"left_finger_{side}", f"right_finger_{side}") \
                and other_body == "baton" and other_geom == f"baton_handle_{side}"
            phases = {"L": {"pick_donor", "present_handover", "receive_baton", "donor_release"},
                      "R": {"receive_baton", "donor_release", "home_donor_arm", "receiver_place"}}
            depth_ok = float(c.dist) >= -float(self.runtime.p["allowed_handle_penetration_m"])
            allowed_handle = bool(correct_pair and self.phase in phases.get(side, set()) and depth_ok)
            if allowed_handle:
                self.allowed.append({"time": float(d.time), "phase": self.phase, "side": side,
                                     "finger_body": arm_body, "handle_geom": other_geom,
                                     "distance_m": float(c.dist)})
                continue
            kind = None
            if sa and sb and sa != sb: kind = "inter_arm"
            elif correct_pair and not depth_ok: kind = "excessive_handle_penetration"
            elif correct_pair: kind = "handle_contact_wrong_phase"
            elif side and other_body == "baton": kind = "arm_baton_wrong_part"
            elif side and other_body == "table" and not arm_body.startswith("link0"): kind = "arm_table"
            if kind:
                self.rows.append({"time": float(d.time), "phase": self.phase, "kind": kind,
                                  "body_a": ba, "body_b": bb, "geom_a": ga, "geom_b": gb,
                                  "distance_m": float(c.dist)})
    def recent_handle_contacts(self, side, since):
        return [r for r in self.allowed if r["side"] == side and r["time"] >= since]
    def summary(self):
        groups = {}
        for r in self.rows:
            key = (r["phase"], r["kind"], *sorted((r["body_a"], r["body_b"])))
            g = groups.setdefault(key, {"phase": key[0], "kind": key[1], "body_pair": list(key[2:]),
                                        "step_samples": 0, "min_distance_m": 0.0})
            g["step_samples"] += 1; g["min_distance_m"] = min(g["min_distance_m"], r["distance_m"])
        return sorted(groups.values(), key=lambda x: (x["phase"], x["kind"], x["body_pair"]))


class SmoothHandoverRuntime:
    def __init__(self, config, out, seed=0, render=True):
        del seed
        self.config = deepcopy(config); self.p = self.config["handover"]; self.h = self.config["table"]["top_z"]
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        arms = [(r["suffix"], r["yaw"]) for r in self.config["robots"]]
        self.env = Exp4Env(build_scene(self.config), arms=arms, obj_names=["baton"], render=render, img_size=560)
        self.env.reset()
        if render:
            self.env.start_record(); self.env.record_frame("overview"); self.env.save_png(str(self.out/"before.png"), "overview")
        self.skills = [ArmSkills(self.env, a, recorder=(lambda: self.env.record_frame("overview")) if render else None,
                                 rec_every=2, block_half=self.p["object_half_size"][2]) for a in self.env.arms]
        for arm in self.env.arms:
            arm._grasp_enabled = False
        self.initial_pos, self.initial_quat = self.env.get_object_pose("baton")
        self.events = []; self.home_dirty = [False, False]; self.home_events = set(); self.goal = goal()
        self.audit = HandoverAudit(self); self.audit.install(); self.audit.sample(); self._audit_restored = False
        self._log_welds()

    def _log_welds(self):
        for i, arm in enumerate(self.env.arms):
            original = arm._set_weld
            def logged(bound, name, active, _original=original, _i=i):
                self.events.append({"event": "grasp_weld", "arm": _i, "active": bool(active),
                                    "object": name, "time": float(self.env.d.time),
                                    "object_position": self.pos().tolist(),
                                    "finger_mid": bound._finger_mid().tolist()})
                return _original(name, active)
            arm._set_weld = types.MethodType(logged, arm)

    def pos(self): return self.env.get_object_pose("baton")[0]
    def quat(self): return self.env.get_object_pose("baton")[1]
    def weld_constraint_active(self, i):
        arm = self.env.arms[i]; wid = arm.obj_weld["baton"]
        return bool(wid >= 0 and self.env.d.eq_active[wid] == 1)
    def weld_active(self, i):
        return bool(self.weld_constraint_active(i) and self.env.arms[i]._grip_cmd < .5)
    def released(self): return not any(self.weld_constraint_active(i) for i in (0, 1))
    def collision_free(self):
        tol = float(self.p["forbidden_penetration_tolerance_m"])
        return not any(r["distance_m"] < -tol for r in self.audit.rows)
    def handle_world(self, i):
        rot = self.env.d.xmat[self.env.obj_bid["baton"]].reshape(3, 3)
        local = np.array([0.0, self.p["handle_offset_y"] * (1 if i == 0 else -1), self.p["handle_height"]])
        return self.pos() + rot @ local
    def _finger_target(self, i, handle):
        arm = self.env.arms[i]
        desired_mid = np.asarray(handle, float) + [0.0, 0.0, self.p["finger_mid_above_handle_m"]]
        return desired_mid + (arm.get_ee_pose()[0] - arm._finger_mid())
    def _move(self, i, target, grip, tol=.010, max_steps=180, ki=.2):
        err = self.skills[i].move_to(target, grip=grip, tol=tol, max_steps=max_steps, ki=ki)
        if err > .04: raise RuntimeError(f"arm {i} residual {err:.4f}m")
        return err
    def _move_object_slow(self, i, target_origin, step=.008, max_steps=420):
        arm = self.env.arms[i]; sk = self.skills[i]; target = np.asarray(target_origin, float)
        saved = arm.bias_world.copy(); arm.bias_world = np.zeros(3)
        try:
            for k in range(max_steps):
                error = target - self.pos()
                if np.linalg.norm(error) < .008:
                    break
                action = np.zeros(7); action[:3] = np.clip(error, -step, step); action[6] = 0.0
                self.env.step_arms({arm.s: action}, sub_steps=8)
                if k % 2 == 0: sk._tick()
        finally:
            arm.bias_world = saved
        residual = float(np.linalg.norm(target - self.pos()))
        if residual > .025: raise RuntimeError(f"baton transport residual {residual:.4f}m")
        return residual
    def _close_and_attach(self, i):
        side = ("L", "R")[i]; arm = self.env.arms[i]
        since = float(self.env.d.time)
        for _ in range(30):
            self.env.hold_arms({arm.s: 0.0}, sub_steps=8); self.skills[i]._tick()
        handle = self.handle_world(i); distance = float(np.linalg.norm(arm._finger_mid() - handle))
        contacts = self.audit.recent_handle_contacts(side, since)
        if distance > self.p["attachment_distance_m"]:
            raise RuntimeError(f"{side} handle distance {distance:.4f}m exceeds gate")
        if len(contacts) < int(self.p["required_contact_samples"]):
            raise RuntimeError(f"{side} handle contact not observed before assisted attachment")
        arm._set_weld("baton", True)
        self.events.append({"event": "contact_gated_attach", "arm": i, "time": float(self.env.d.time),
                            "finger_handle_distance_m": distance, "contact_samples": len(contacts),
                            "mode": "assisted_grasp_weld"})
        for _ in range(12): self.env.hold_arms({arm.s: 0.0}, sub_steps=8); self.skills[i]._tick()
        if not self.weld_active(i): raise RuntimeError(f"arm {i} weld did not activate")

    def pick_donor(self):
        self.audit.phase = "pick_donor"; self.home_dirty[0] = True
        handle = self.handle_world(0); target = self._finger_target(0, handle)
        self._move(0, target + [0, 0, self.p["approach_clearance"]], 1.0, ki=.2)
        self._move(0, target, 1.0, tol=.007, ki=.15)
        self._close_and_attach(0)
        self._move_object_slow(0, [self.pos()[0], self.pos()[1], self.h + .20], step=.006)
    def present_handover(self):
        self.audit.phase = "present_handover"
        if not self.weld_active(0): raise RuntimeError("donor does not hold baton")
        self._move_object_slow(0, self.p["meet_center_xyz"], step=.006)
        for _ in range(18): self.env.hold_arms({"_L": 0.0, "_R": 1.0}, sub_steps=8); self.skills[0]._tick()
    def receive_baton(self):
        self.audit.phase = "receive_baton"; self.home_dirty[1] = True
        if not self.weld_active(0): raise RuntimeError("receiver approach requires donor hold")
        handle = self.handle_world(1); target = self._finger_target(1, handle)
        self._move(1, target + [0, 0, self.p["approach_clearance"]], 1.0, ki=.0)
        self._move(1, target, 1.0, tol=.008, max_steps=220, ki=.0)
        self._close_and_attach(1)
        if not (self.weld_active(0) and self.weld_active(1)):
            raise RuntimeError("handover did not establish measured dual hold")
        self.events.append({"event": "dual_hold_verified", "time": float(self.env.d.time),
                            "object_z": float(self.pos()[2])})
    def release_donor(self):
        self.audit.phase = "donor_release"
        if not self.weld_active(1): raise RuntimeError("receiver must hold before donor release")
        arm = self.env.arms[0]
        for _ in range(12): self.env.hold_arms({"_L": 1.0, "_R": 0.0}, sub_steps=8); self.skills[0]._tick()
        arm._set_weld("baton", False)
        self.events.append({"event": "donor_release_after_receiver", "time": float(self.env.d.time),
                            "receiver_active": self.weld_active(1), "object_z": float(self.pos()[2])})
        if not self.weld_active(1): raise RuntimeError("receiver lost baton at donor release")
        self._move(0, self.p["donor_safe_xyz"], 1.0, max_steps=220, ki=.2)
        if not self.weld_active(1): raise RuntimeError("receiver lost baton during donor withdrawal")
        self.events.append({"event": "donor_withdrawn", "time": float(self.env.d.time)})
    def placement_geometry(self):
        p, q = self.env.get_object_pose("baton"); bottom = float(geom_corners(self.env.m, self.env.d, "baton")[:,2].min())
        rotation_error = 2 * math.acos(float(np.clip(abs(np.dot(q, self.initial_quat)), -1, 1)))
        xy_error = float(np.linalg.norm(p[:2] - np.asarray(self.p["destination_center_xy"])))
        ok = bool(xy_error < self.p["position_tolerance_m"] and abs(bottom-self.h) < .012
                  and rotation_error < math.radians(self.p["rotation_tolerance_deg"]) and self.released())
        return {"placed": ok, "xy_error_m": xy_error, "bottom_gap_m": bottom-self.h,
                "rotation_error_deg": math.degrees(rotation_error), "released": self.released(),
                "origin": p.tolist()}
    def place_baton(self):
        self.audit.phase = "receiver_place"
        if not self.weld_active(1) or self.weld_active(0): raise RuntimeError("place requires receiver-only hold")
        dst = np.array([*self.p["destination_center_xy"], self.h + self.p["object_half_size"][2] + .003])
        command_xy = dst[:2] + np.asarray(self.p.get("placement_preoffset_xy", [0.0, 0.0]), float)
        lane = np.array([dst[0], dst[1], self.h + .24])
        self._move_object_slow(1, lane, step=.006)
        self._move_object_slow(1, [dst[0], dst[1], self.h + .20], step=.006)
        sk = self.skills[1]; sk.record_grab_offset("baton")
        sk.descend_place_tracked(command_xy, self.h, "baton", gap=.003,
                                 max_steps=320, xy_ok=.012, z_step=.018,
                                 suppress_feedforward=True)
        self.events.append({"event": "pre_release_pose", "time": float(self.env.d.time),
                            "object_position": self.pos().tolist()})
        for _ in range(8): self.env.hold_arms({"_R": 1.0}, sub_steps=4); self.skills[1]._tick()
        self.env.arms[1]._set_weld("baton", False)
        for _ in range(18): self.env.hold_arms({"_R": 1.0}, sub_steps=8); self.skills[1]._tick()
        self.events.append({"event": "post_release_pose", "time": float(self.env.d.time),
                            "object_position": self.pos().tolist()})
        if not self.released(): raise RuntimeError("baton weld remained active after placement")
        sk.retreat(dz=.10)
        self._move(1, self.p["receiver_safe_xyz"], 1.0, max_steps=220, ki=.10)
        self.events.append({"event": "receiver_withdrawn", "time": float(self.env.d.time)})
        if not self.placement_geometry()["placed"]:
            raise RuntimeError(f"baton did not settle at destination: {self.placement_geometry()}")
    def handover_verified(self):
        dual = next((e for e in self.events if e["event"] == "dual_hold_verified"), None)
        release = next((e for e in self.events if e["event"] == "donor_release_after_receiver"), None)
        return bool(dual and release and dual["time"] < release["time"] and release["receiver_active"]
                    and min(dual["object_z"], release["object_z"]) > self.h + .10)
    def _controlled_home(self, i):
        if self.weld_active(i): raise RuntimeError("controlled home requires empty arm")
        arm = self.env.arms[i]; sk = self.skills[i]
        self._move(i, self.p[("donor_safe_xyz", "receiver_safe_xyz")[i]], 1.0, max_steps=220, ki=.2)
        start = arm.arm_target.copy(); steps = int(self.p["controlled_home_steps"])
        self.events.append({"event": "controlled_actuator_home", "arm": i, "time": float(self.env.d.time),
                            "q_start": start.tolist(), "q_goal": HOME_QPOS.tolist(), "steps": steps})
        for k in range(1, steps+1):
            self.env.step_joint(arm, start + (HOME_QPOS-start)*(k/steps), grip=1.0, sub_steps=8); sk._tick()
        for _ in range(12): self.env.step_joint(arm, HOME_QPOS, grip=1.0, sub_steps=8); sk._tick()
        self.home_events.add(i)
        if not self.home(i): raise RuntimeError(f"arm {i} did not reach controlled HOME")
    def home(self, i):
        arm = self.env.arms[i]; q = np.array([self.env.d.qpos[a] for a in arm.arm_qadr])
        return bool(i in self.home_events and not self.weld_constraint_active(i) and arm._grip_cmd >= .5
                    and np.linalg.norm(q-HOME_QPOS) < .10)
    def at_source(self):
        return bool(self.released() and np.linalg.norm(self.pos()[:2]-self.initial_pos[:2]) < .018
                    and abs(self.pos()[2]-self.initial_pos[2]) < .012)
    def at_handover(self):
        return bool(self.weld_active(0) and np.linalg.norm(self.pos()-np.asarray(self.p["meet_center_xyz"])) < .030)
    def state(self):
        s = set()
        if self.weld_active(0): s.add("donor_holding(baton)")
        else: s.add("empty(0)")
        if self.weld_active(1): s.add("receiver_holding(baton)")
        else: s.add("empty(1)")
        if self.weld_active(0) and self.weld_active(1): s.add("dual_holding(baton)")
        if self.pos()[2] > self.h + .10 and not self.released(): s.add("baton_airborne()")
        if self.at_source(): s.add("at_source(baton)")
        if self.at_handover(): s.add("at_handover_pose(baton)")
        if self.placement_geometry()["placed"]: s.add("at_destination(baton)")
        if any(e["event"] == "contact_gated_attach" and e["arm"] == 1 for e in self.events): s.add("receiver_contact_verified()")
        if any(e["event"] == "donor_withdrawn" for e in self.events): s.add("donor_withdrawn()")
        if self.placement_geometry()["placed"] and any(e["event"] == "receiver_withdrawn" for e in self.events): s.add("receiver_withdrawn()")
        if self.handover_verified(): s.add("handover_verified()")
        if self.collision_free(): s.add("collision_free()")
        for i in (0, 1):
            if self.home(i): s.add(f"home({i})")
        return s
    def api(self, model):
        kind = model["program_kind"]; self.audit.phase = model["name"]
        values = {"pick_donor": self.pick_donor, "move_to_handover_pose": self.present_handover,
                  "grasp_opposite_handle": self.receive_baton, "finalize_transfer": self.release_donor,
                  "place_baton": self.place_baton,
                  "return_arm_home": lambda: self._controlled_home(int(model["args"]["arm"]))}
        return {name: values[name] for name in DOCS[kind]}
    def stable_goal(self):
        steps = max(1, int(self.p["stability_seconds"] / (self.env.m.opt.timestep * 16)))
        stable = True
        for _ in range(steps):
            self.env.hold_arms({"_L": 1.0, "_R": 1.0}, sub_steps=16)
            if self.env.render_enabled: self.env.record_frame("overview")
            stable = stable and self.goal <= self.state()
        return bool(stable)
    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        if self.env.render_enabled:
            self.env.save_png(str(path/"final.png"), "overview")
            self.env.save_png(str(path/"final_front.png"), "front")
            self.env.save_video(str(path/"rollout.mp4"), fps=30)
        (path/"collision_audit.json").write_text(json.dumps({"collision_free": self.collision_free(),
             "forbidden_groups": self.audit.summary(), "forbidden_samples": self.audit.rows,
             "allowed_handle_contacts": self.audit.allowed}, indent=2))
    def close(self):
        if not self._audit_restored:
            self.audit.restore(); self._audit_restored = True
        if self.env.renderer: self.env.renderer.close()
