"""Handle-free carton variant. No hidden handle geometry; direct box-face contacts.

The legacy contact-gated weld remains explicit assistance. This module changes
geometry and grasp targets, not object state or success flags. The old scene and
controller remain available separately.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import types
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from core.smooth_handover_bridge import (
    ROOT, ArmSkills, DOCS, Exp4Env, SmoothHandoverRuntime, HandoverAudit,
    _body_name, _geom_name, _arm_side, _vec, build_scene as legacy_scene, domain, goal,
)


def build_box_scene(config):
    c = deepcopy(config)
    # Do not generate old handle-shaped display replicas for this variant.
    c["visual_style"]["display_layout"] = {"centers_xy": []}
    # Legacy scene scaffold only; these temporary geoms are deleted before compile.
    c["handover"].update(handle_half_size=[.014,.028,.014], handle_offset_y=.142, handle_height=.055)
    root = ET.fromstring(legacy_scene(c))
    root.set("model", "handleless_green_tea_handover")
    root.find("compiler").set("meshdir", str(ROOT / "mujoco_workspace/franka_articulated/assets/panda_assets"))
    world = root.find("worldbody")
    body = world.find("body[@name='baton']")
    for g in list(body.findall("geom")):
        body.remove(g)
    p = c["handover"]
    half = np.asarray(p["object_half_size"], float)
    ET.SubElement(body, "geom", name="tea_box_core", type="box", size=_vec(half),
                  material="tea_green", mass=str(p["object_mass"]), friction="1.2 0.02 0.001",
                  solref="0.005 1", solimp="0.99 0.99 0.001", priority="1")
    # Artwork always follows the box faces (not graspable protrusions).
    asset = root.find("asset")
    texture = ROOT / c["visual_style"]["texture_file"]
    if texture.is_file():
        ET.SubElement(asset, "texture", name="tea_print", type="2d", file=str(texture))
        ET.SubElement(asset, "material", name="tea_print", texture="tea_print", texuniform="false", specular="0.08", shininess="0.1")
        # Front/back quad meshes with explicit UVs; zero mass/collision.
        for side in (-1, 1):
            xx = side * (half[0] + .00008)
            verts = [[xx, -half[1], -half[2]], [xx, half[1], -half[2]], [xx, half[1], half[2]], [xx, -half[1], half[2]]]
            ET.SubElement(asset, "mesh", name=f"tea_print_{side}", vertex=_vec(np.array(verts).ravel()),
                          face="0 1 2 0 2 3", texcoord="0 1 1 1 1 0 0 0", inertia="shell")
            ET.SubElement(body, "geom", name=f"tea_box_print_{side}", type="mesh", mesh=f"tea_print_{side}",
                          material="tea_print", mass="0", contype="0", conaffinity="0")
    else:
        ET.SubElement(body, "geom", name="tea_box_cream_label", type="box", size=_vec([.0001, half[1]*.40, half[2]*.60]),
                      pos=_vec([half[0]+.0001, 0, 0]), material="tea_cream", mass="0", contype="0", conaffinity="0")
    # Fixed inert display props: same complete shape/artwork as the task carton.
    for i, xy in enumerate(c["visual_style"].get("box_display_centers_xy", [])):
        disp = ET.SubElement(world, "body", name=f"tea_display_{i}", pos=_vec([*xy, c["table"]["top_z"]+half[2]]))
        for g in body.findall("geom"):
            copy = deepcopy(g)
            copy.set("name", f"tea_display_{i}_{g.get('name')}")
            copy.set("mass", "0")
            is_core = g.get("name") == "tea_box_core"
            copy.set("contype", "1" if is_core else "0"); copy.set("conaffinity", "1" if is_core else "0")
            disp.append(copy)
    # Neutral studio illumination and solid blue floor with the existing white grid.
    root.find("visual/headlight").set("diffuse", ".5 .5 .5")
    root.find("visual/headlight").set("ambient", ".35 .35 .35")
    for light in world.findall("light"):
        light.set("diffuse", ".9 .9 .9" if light.get("name")=="key" else ".4 .4 .4")
    table_mat=asset.find("material[@name='tea_table']");table_mat.set("rgba", ".50 .51 .52 1")
    # Rebuild the legs underneath the configured tabletop, not the legacy origin.
    for g in list(world.findall("geom")):
        if g.get("name") is None:world.remove(g)
    t=c["table"];hh=(t["top_z"]-t["thickness"])/2
    for sx in (-1,1):
        for sy in (-1,1):
            xy=np.array(t["center_xy"])+np.array([sx,sy])*(np.array(t["full_size_xy"])/2-.06)
            ET.SubElement(world,"geom",name=f"studio_leg_{sx}_{sy}",type="box",size=_vec([.025,.025,hh]),pos=_vec([*xy,hh]),material="tea_table_edge")
    floor=asset.find("texture[@name='grid']")
    floor.set("rgb1", ".30 .62 .77"); floor.set("rgb2", ".30 .62 .77")
    floor_mat=asset.find("material[@name='grid']");floor_mat.set("reflectance", "0");floor_mat.set("specular", "0")
    # Hide robot debug TCP sites, not collision geometry.
    for robot in world.findall("body"):
        if robot.get("name", "").startswith("link0"):
            for site in robot.iter("site"):site.set("rgba", "0 0 0 0")
    if p.get("robot_gravity_compensation", True):
        for rb in world.findall("body"):
            if rb.get("name", "").startswith("link0"):
                for b in rb.iter("body"):b.set("gravcomp", "1")
    return ET.tostring(root, encoding="unicode")


class BoxFaceAudit(HandoverAudit):
    """Only designated fingers touching their designated box-face zone are allowed."""
    def sample(self):
        r = self.runtime; m, d = r.env.m, r.env.d
        r.sample_motion()
        for index, contact in enumerate(d.contact):
            ba, bb = _body_name(m, contact.geom1), _body_name(m, contact.geom2)
            ga, gb = _geom_name(m, contact.geom1), _geom_name(m, contact.geom2)
            sa, sb = _arm_side(ba), _arm_side(bb)
            side = sa or sb
            arm_body = ba if sa else bb if sb else None
            other_body = bb if sa else ba if sb else None
            other_geom = gb if sa else ga if sb else None
            prop_pair = (ba.startswith("tea_display_") or bb.startswith("tea_display_")) and (side or "baton" in (ba,bb))
            if prop_pair:
                self.rows.append({"time":float(d.time),"phase":self.phase,"kind":"unexpected_display_contact",
                                  "body_a":ba,"body_b":bb,"geom_a":ga,"geom_b":gb,"distance_m":float(contact.dist)})
            if not side:
                continue
            local = d.xmat[r.env.obj_bid["baton"]].reshape(3, 3).T @ (contact.pos-r.pos())
            i = 0 if side == "L" else 1
            target = r.p["grasp_local_points"][i]
            normal_local=d.xmat[r.env.obj_bid["baton"]].reshape(3,3).T@contact.frame[:3]
            face = abs(abs(local[0])-r.p["object_half_size"][0]) < .004 and abs(normal_local[0])>.9
            zone = abs(local[1]-target[1]) < r.p["contact_zone_half_y"]
            correct = arm_body in (f"left_finger_{side}", f"right_finger_{side}") and other_body == "baton" and other_geom == "tea_box_core" and face and zone
            phases = {"L": {"pick_donor", "present_handover", "receive_baton", "donor_release"},
                      "R": {"receive_baton", "donor_release", "home_donor_arm", "receiver_place"}}
            if correct and self.phase in phases[side] and contact.dist >= -r.p["allowed_handle_penetration_m"]:
                force = np.zeros(6); mujoco.mj_contactForce(m, d, index, force)
                self.allowed.append({"time": float(d.time), "phase": self.phase, "side": side,
                                     "finger_body": arm_body, "handle_geom": other_geom,
                                     "local_point": local.tolist(), "normal_force_n": float(force[0]),
                                     "distance_m": float(contact.dist)})
                continue
            kind = None
            if sa and sb and sa != sb: kind = "inter_arm"
            elif other_body == "baton": kind = "arm_box_wrong_part_or_excess_depth"
            elif other_body == "table" and not arm_body.startswith("link0"): kind = "arm_table"
            if kind:
                self.rows.append({"time": float(d.time), "phase": self.phase, "kind": kind,
                                  "body_a": ba, "body_b": bb, "geom_a": ga, "geom_b": gb,
                                  "distance_m": float(contact.dist)})


class HandlelessTeaRuntime(SmoothHandoverRuntime):
    def __init__(self, config, out, seed=0, render=True):
        self.config = deepcopy(config); self.p = self.config["handover"]; self.h = config["table"]["top_z"]
        self.seed = seed
        rng = np.random.default_rng(seed)
        jitter = self.config["evaluation"]["layout_jitter_m"] if seed else 0.0
        self.p["start_center_xy"] = (np.asarray(self.p["start_center_xy"])+rng.uniform(-jitter, jitter, 2)).tolist()
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        xml = build_box_scene(self.config)
        (self.out/"scene.xml").write_text(xml)
        (self.out/"effective_config.json").write_text(json.dumps(self.config, indent=2))
        self.env = Exp4Env(xml, arms=[(a["suffix"], a["yaw"]) for a in self.config["robots"]], obj_names=["baton"], render=render, img_size=720)
        self.env.reset()
        self.skills = [ArmSkills(self.env, a, recorder=(lambda:self.env.record_frame("overview")) if render else None,
                                 rec_every=2, block_half=self.p["object_half_size"][2]) for a in self.env.arms]
        for arm in self.env.arms: arm._grasp_enabled = False
        self.initial_pos, self.initial_quat = self.env.get_object_pose("baton")
        self.events=[]; self.home_dirty=[False,False]; self.home_events=set(); self.goal=goal()
        self.frame_times=[];self.trace=[];self.next_frame=float(self.env.d.time);self.next_trace=self.next_frame
        self.max_actual_speed=0.0;self.peak_tracking=0.0
        self.env.start_record()
        self.env.record_frame=lambda *a,**kw: None  # uniformly sampled below, never per primitive
        self.audit=BoxFaceAudit(self); self.audit.install(); self.audit.sample(); self._audit_restored=False
        self._log_welds()
        if render:self.env.save_png(str(self.out/"before.png"), "overview")

    def sample_motion(self):
        d=self.env.d
        ids=[j for a in self.env.arms for j in a.arm_qadr]
        self.max_actual_speed=max(self.max_actual_speed,float(np.max(abs(d.qvel[ids]))))
        if d.time>=self.next_trace-1e-9:
            self.trace.append({"t":float(d.time),"box":self.pos().tolist(),"q":d.qpos[ids].tolist(),"phase":self.audit.phase})
            self.next_trace+=.02
        if d.time>=self.next_frame-1e-9:
            if self.env.render_enabled:
                self.env._rec_frames.append(self.env.render_cam("overview"))
            self.frame_times.append(float(d.time));self.next_frame+=1/30

    def placement_geometry(self):
        base=super().placement_geometry()
        bid=self.env.obj_bid["baton"];m,d=self.env.m,self.env.d
        support=False
        for j,c in enumerate(d.contact):
            bodies={_body_name(m,c.geom1),_body_name(m,c.geom2)}
            if bodies=={"baton","table"} and abs(c.frame[2])>.8 and c.dist<.0005:
                force=np.zeros(6);mujoco.mj_contactForce(m,d,j,force)
                if force[0]>.01:support=True
        base["table_support_contact"]=support
        base["placed"]=bool(base["placed"] and support)
        return base

    def handle_world(self, i):
        # Compatibility method name only: there are NO handle geoms or hidden pads.
        rot=self.env.d.xmat[self.env.obj_bid["baton"]].reshape(3,3)
        return self.pos()+rot@np.asarray(self.p["grasp_local_points"][i])

    def _move(self, i, target, grip, **kwargs):
        arm=self.env.arms[i];start=arm.get_ee_pose()[0];target=np.asarray(target,float)
        dt=.02;duration=max(.8,1.875*np.linalg.norm(target-start)/.12)
        count=math.ceil(duration/dt)
        for k in range(1,count+1):
            t=k/count;u=10*t**3-15*t**4+6*t**5
            q=arm.solve_ik(start+u*(target-start),arm.grasp_quat,local_only=True,w_rot=1.0,iters=60)
            q=arm.arm_target+np.clip(q-arm.arm_target,-.65*dt,.65*dt)
            self.env.step_joint(arm,q,grip,sub_steps=10)
            self.skills[i]._tick()
        for _ in range(12):self.env.step_joint(arm,arm.arm_target,grip,sub_steps=10);self.skills[i]._tick()
        error=float(np.linalg.norm(target-arm.get_ee_pose()[0]));self.peak_tracking=max(error,self.peak_tracking)
        if error>.010:raise RuntimeError(f"Cartesian segment {i} missed by {error:.4f}m")
        return error

    def _move_object_slow(self, i, target_origin, **kwargs):
        target=np.asarray(target_origin,float)
        for _ in range(3):
            self._move(i,self.env.arms[i].get_ee_pose()[0]+target-self.pos(),0.0)
            if np.linalg.norm(target-self.pos())<.004:break
        residual=float(np.linalg.norm(target-self.pos()))
        if residual>.009:raise RuntimeError(f"Held box missed target by {residual:.4f}m")
        return residual

    def _close_and_attach(self, i):
        side=("L","R")[i]; arm=self.env.arms[i]; since=float(self.env.d.time)
        for _ in range(30):
            self.env.hold_arms({arm.s:0.0}, sub_steps=8); self.skills[i]._tick()
        contacts=self.audit.recent_handle_contacts(side,since)
        simultaneous={}
        for row in contacts:
            if row["normal_force_n"]>.03 and row["time"]>=float(self.env.d.time)-.10:
                simultaneous.setdefault(row["time"],set()).add(row["finger_body"])
        bilateral=sum(len(fingers)==2 for fingers in simultaneous.values())
        current_fingers=simultaneous.get(float(self.env.d.time),set())
        ee_distance=float(np.linalg.norm(arm.get_ee_pose()[0]-self.handle_world(i)))
        if bilateral<2 or len(current_fingers)!=2 or ee_distance>.030:
            raise RuntimeError(f"{side} direct carton grasp rejected: bilateral_steps={bilateral}, TCP_error={ee_distance:.4f}, contacts={len(contacts)}")
        if not self.collision_free():
            raise RuntimeError(f"{side} grasp collision: {self.audit.summary()}")
        before=self.pos().copy()
        arm._set_weld("baton", True)
        jump=float(np.linalg.norm(self.pos()-before))
        self.events.append({"event":"contact_gated_attach","arm":i,"time":float(self.env.d.time),
                            "contact_surface":"tea_box_core","bilateral_contact_steps":bilateral,
                            "tcp_grasp_error_m":ee_distance,"mode":"assisted_grasp_weld","handles_exist":False,
                            "final_window_s":.10,"activation_pose_jump_m":jump,"both_fingers_contact_at_activation":True})
        for _ in range(12):self.env.hold_arms({arm.s:0.0},sub_steps=8);self.skills[i]._tick()

    def _controlled_home(self, i):
        self.audit.phase="home_donor_arm" if i==0 else "home_receiver_arm"
        return super()._controlled_home(i)

    def stable_goal(self):
        self.audit.phase="stability"
        return super().stable_goal()

    def api(self, model):
        values={"donor_pick":self.pick_donor,"present":self.present_handover,"receive":self.receive_baton,
                "donor_release":self.release_donor,"receiver_place":self.place_baton,
                "controlled_home":lambda:self._controlled_home(int(model["args"]["arm"]))}
        method=next(iter(DOCS[model["program_kind"]]))
        return {method:values[model["program_kind"]]}

    def save(self,path):
        # Full contact logs retained in compact JSON, not millions of indented lines.
        path=Path(path)
        if self.env.render_enabled:
            self.env.save_png(str(path/"final.png"),"overview")
            self.env.save_png(str(path/"final_front.png"),"front")
            self.env.save_video(str(path/"rollout.mp4"),fps=30)
        (path/"collision_audit.json").write_text(json.dumps({"forbidden":self.audit.rows,"allowed_box_face":self.audit.allowed},separators=(",",":")))
        (path/"motion_trace.json").write_text(json.dumps(self.trace,separators=(",",":")))


def box_domain():
    models=deepcopy(domain())
    for m in models:
        m["name"]=m["name"].replace("pick_baton_by_donor","pick_tea_box").replace("receiver_grasps_opposite_handle","receiver_grasps_box_face")
        m["description"]=m["description"].replace("donor-side handle","donor-side box faces").replace("opposite physical handle","opposite box-face zone")
    return models
