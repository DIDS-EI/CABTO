"""Three-pair Cover: grounded STRIPS, plain-data API bindings and independent checks.

No object snap. Oracle object pose, grasp weld and explicit arm rehome remain
assisted features. Existing low-level controllers are reused, not task templates.
"""
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"stage2_scripted"))
import numpy as np
from exp4_env import Exp4Env
from arm_skills import ArmSkills
import scene_cover as sc

PICK_DOCS={
 "prepare_empty":"api.prepare_empty() -> None: with EMPTY hand only, move arm to initial configuration (explicit assisted joint rehome). Call before approaching a new lid.",
 "payload_pose":"api.payload_pose() -> tuple: returns current lid grasp-point (x,y,z) in world metres; z is already the calibrated TCP grasp height, not an object top. Takes no arguments.",
 "approach":"api.approach(p) -> None: open gripper, move high above grasp-point tuple p. Does not descend or close.",
 "align":"api.align(p) -> None: refine x/y alignment above the same tuple p, keeping hand open.",
 "descend":"api.descend(p) -> None: descend open gripper to the grasp height p[2], keeping p[:2]. Takes one tuple.",
 "close":"api.close(p) -> None: close gripper while tracking p[:2]. Does not lift. Takes one tuple.",
 "lift":"api.lift() -> None: raise closed hand to the transport height. No arguments.",
 "measure_offset":"api.measure_offset() -> None: measure current held lid offset relative to hand for placement calibration; call after grasp and lift. No arguments."
}
PLACE_DOCS={
 "target_pose":"api.target_pose() -> tuple: read-only TOP point (x,y,z) of the action destination base. No arguments. Tuples, not dictionaries.",
 "move_above":"api.move_above(p) -> tuple: transport closed hand above destination tuple p; return TCP xy. Does not release.",
 "align_target":"api.align_target(p) -> tuple: refine alignment above destination tuple p; return TCP xy.",
 "lower":"api.lower(p) -> float: tracked descent to TOP surface of destination tuple p. Closed hand; no release. Feedforward suppressed only during descent.",
 "release":"api.release() -> None: open fingers and release at CURRENT pose. Does not transport or lower. No arguments.",
 "retreat":"api.retreat() -> None: retreat vertically with empty gripper after release. No arguments.",
 "settle":"api.settle() -> None: wait with open gripper. Use after release, never while holding a lid. No arguments."
}


CANONICAL_PICK_DOCS={
 "prepare_empty":PICK_DOCS["prepare_empty"],
 "target_pose":"api.target_pose() returns a tuple (x,y,z) for the lid HANDLE grasp point. No arguments. The z is the TCP grasp height already.",
 "approach":"api.approach(xyz) moves the OPEN hand above xyz. One tuple argument.",
 "align_xy":"api.align_xy(xyz) aligns OPEN hand horizontally above xyz. One tuple argument.",
 "descend_to":"api.descend_to(z, lock_xy) moves OPEN hand down to z; lock_xy is a tuple (x,y). Two arguments.",
 "grasp":"api.grasp(xy) CLOSES fingers to grasp the handle at current height. One tuple (x,y) argument. Does not lift.",
 "lift":"api.lift() raises CLOSED hand and held lid to transport height. No arguments.",
 "record_grab_offset":"api.record_grab_offset() measures held lid offset for placement; use AFTER grasp and lift. No arguments."
}
CANONICAL_PLACE_DOCS={
 "target_pose":PLACE_DOCS["target_pose"],"move_above":PLACE_DOCS["move_above"],
 "align_target":PLACE_DOCS["align_target"],"lower":PLACE_DOCS["lower"],
 "place_release":"api.place_release() opens fingers and releases at current pose. No movement. No arguments.",
 "retreat":PLACE_DOCS["retreat"],"settle":PLACE_DOCS["settle"]}


def models():
    result=[]
    for k,*_ in sc.PAIRS:
        lid,base="lid_"+k,"base_"+k
        result += [dict(name="pick_lid",args={"lid":lid,"base":base},
            pre=["hand_empty()",f"at_source({lid})"],add=[f"holding({lid})"],
            **{"del":["hand_empty()",f"at_source({lid})"]}),
            dict(name="place_lid",args={"lid":lid,"base":base},
            pre=[f"holding({lid})",f"clear({base})"],add=[f"covered({lid},{base})","hand_empty()"],
            **{"del":[f"holding({lid})",f"clear({base})"]})]
    return result


def effect_report(model,before,after):
    pre,add,delete=(set(model[k]) for k in ("pre","add","del"))
    violations={"missing_pre":sorted(pre-before),"missing_add":sorted(add-after),
                "remaining_del":sorted(delete&after),"missing_preserved_pre":sorted((pre-delete)-after)}
    return not any(violations.values()),violations


class CoverRuntime:
    def __init__(self,render=True):
        # Camera-only candidate changes; retain reference task coordinates.
        sys.path.insert(0,str(ROOT.parents[1]/"edps"))
        from probe_five_scene_preflight import candidate_xml
        self.env=Exp4Env(candidate_xml(sc.build_scene_xml(),"cover"),[("",0.0)],
                       [f"lid_{k}" for k,*_ in sc.PAIRS],render=render,img_size=560)
        self.arm=self.env.arms[0]
        self.initial={k:v.copy() for k,v in self.env._init_obj_qpos.items()}
        self.events=[]
        self.reset()

    def reset(self,seed=0):
        rng=np.random.default_rng(seed)
        for name,q in self.initial.items():
            pose=q.copy()
            if seed:pose[:2]+=rng.uniform(-.003,.003,2)
            self.env._init_obj_qpos[name]=pose
        self.env.reset();self.arm.grasp_z_tol=.10
        self.env.start_record();self.env.record_frame("overview")
        self.events=[]
        self.skills=ArmSkills(self.env,self.arm,recorder=lambda:self.env.record_frame("overview"),
                              rec_every=2,block_half=sc.LID_HH)

    def geometry(self):
        rows={}
        for k,xy,*_ in sc.PAIRS:
            p,q=self.env.get_object_pose("lid_"+k)
            off=float(np.linalg.norm(p[:2]-xy));gap=float(p[2]-sc.LID_HH-sc.BASE_TOP)
            tilt=float(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
            rows[k]={"xy_m":off,"gap_m":gap,"tilt_deg":math.degrees(tilt),
                     "covered":off<.012 and abs(gap)<.010 and tilt<math.radians(10)}
        return rows

    def state(self):
        state=set();g=self.geometry();any_held=False
        for k,*_ in sc.PAIRS:
            lid,base="lid_"+k,"base_"+k
            attached=self.arm.is_holding(lid)
            p,_=self.env.get_object_pose(lid)
            holding=attached and p[2]>.40+sc.LID_HH+.07
            if attached:any_held=True
            if holding:state.add(f"holding({lid})")
            if not attached and abs(p[2]-(.40+sc.LID_HH))<.012:state.add(f"at_source({lid})")
            if g[k]["covered"]:state.add(f"covered({lid},{base})")
            else:state.add(f"clear({base})")
        if not any_held:state.add("hand_empty()")
        return state

    @property
    def goal(self):return {f"covered(lid_{k},base_{k})" for k,*_ in sc.PAIRS}

    def api(self,model):
        lid,base=model["args"]["lid"],model["args"]["base"]
        sk=self.skills
        def prepare_empty():
            if any(self.arm.is_holding(n) for n in self.env.obj_names):raise RuntimeError("prepare requires empty hand")
            self.events.append({"event":"assisted_arm_rehome","sim_time":float(self.env.d.time)})
            sk.rehome(settle=10)
        def payload_pose():
            p=self.env.get_object_pose(lid)[0]
            return (float(p[0]),float(p[1]),float(p[2]+sc.LID_HH+2*sc.HANDLE_HH-.30*(2*sc.HANDLE_HH)))
        def target_pose():
            p=self.env.get_body_pos(base)
            return (float(p[0]),float(p[1]),float(p[2]+2*sc.BASE_HH))
        def measure_offset():sk.record_grab_offset(lid)
        def move_above(p):return sk.move_above(p,high_z=.72).tolist()
        def align_target(p):return sk.refine_above(p[:2],high_z=.72,obj_name=lid).tolist()
        def lower(p):return sk.descend_place_tracked(p[:2],p[2],obj_name=lid,gap=.003,max_steps=300,suppress_feedforward=True)
        return {"prepare_empty":prepare_empty,"payload_pose":payload_pose,
                "target_pose":payload_pose if model["name"]=="pick_lid" else target_pose,
                "align_xy":lambda xyz:sk.align_xy(xyz),
                "descend_to":lambda z,lock_xy:sk.descend_to(z,lock_xy=lock_xy),
                "grasp":lambda xy:sk.grasp(xy,n=28),"record_grab_offset":measure_offset,"place_release":sk.place_release,
                "approach":lambda p:sk.approach(p),"align":lambda p:sk.align_xy(p),
                "descend":lambda p:sk.descend_to(p[2],lock_xy=p[:2]),
                "close":lambda p:sk.grasp(p[:2],n=28),"lift":lambda:sk.lift(top=.72),
                "measure_offset":measure_offset,"move_above":move_above,"align_target":align_target,
                "lower":lower,"release":sk.place_release,"retreat":lambda:sk.retreat(dz=.16),
                "settle":lambda:sk.hold(grip=1.0,n=20)}

    def stable_goal(self):
        stable=True
        for _ in range(32):
            self.env.hold_arms({"":1.0},sub_steps=16);self.env.record_frame("overview")
            stable=stable and self.goal<=self.state()
        return stable

    def close(self):
        if self.env.renderer:self.env.renderer.close()
