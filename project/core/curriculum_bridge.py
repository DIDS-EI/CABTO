"""STRIPS / plain-data APIs for the FROZEN feasible dual-task curriculum.

No task macros, object writes or rehome are exposed to generated programs.
Oracle geometry, legacy calibrated controllers and grasp welds are explicit.
"""
from copy import deepcopy
from pathlib import Path
import sys
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/"CABTO/exp4_bt_tasks/stage3_cabto"))
from probe_feasible_curriculum import Curriculum, containment
from cover_bridge import effect_report

PICK_DOCS={
 "target_pose":"api.target_pose() -> tuple (x,y,z): read current bound OBJECT grasp point. z already is calibrated TCP grasp height. No arguments.",
 "approach":"api.approach(xyz): move OPEN hand above the object. One XYZ tuple. Does not descend or grasp.",
 "align_xy":"api.align_xy(xyz): align OPEN hand horizontally above one XYZ tuple, without descending or grasping.",
 "descend_to":"api.descend_to(z, lock_xy): move OPEN hand to height z and (x,y) tuple lock_xy. Two arguments.",
 "grasp":"api.grasp(xy): CLOSE fingers at current height tracking tuple (x,y). Does not lift.",
 "lift":"api.lift(): lift CLOSED hand to transport height. No arguments.",
 "record_grab_offset":"api.record_grab_offset(): measure held object's relative pose for subsequent placement. No arguments. Call after lifting."
}
PLACE_DOCS={
 "target_pose":"api.target_pose() -> tuple (x,y,z): read the bound DESTINATION surface point, not the held object. No arguments.",
 "move_above":"api.move_above(p): transport CLOSED hand above destination XYZ tuple p. Does not release or lower.",
 "align_target":"api.align_target(p): align held object horizontally over destination tuple p with closed fingers.",
 "lower":"api.lower(p): lower CLOSED hand towards destination surface tuple p, following the actual held object. Does not release.",
 "place_release":"api.place_release(): OPEN fingers, releasing at CURRENT pose. No movement. No arguments.",
 "retreat":"api.retreat(): move OPEN hand vertically up away from the placement surface. No arguments.",
 "settle":"api.settle(): wait without changing grip or position. No arguments."
}
PARK_DOCS={
 "target_pose":"api.target_pose() -> tuple (x,y,z): safe parking pose outside shared work zone. No arguments.",
 "move_to":"api.move_to(p): move EMPTY OPEN hand to given XYZ tuple p. No arguments beyond one tuple.",
 "settle":"api.settle(): wait preserving existing gripper commands. No arguments."
}
POUR_DOCS={
 "target_pose":"api.target_pose() -> tuple (x,y,z): calibrated vessel-handle pose above bound RECEIVER with outlet offset. No arguments.",
 "move_above":"api.move_above(p): transport CLOSED hand with vessel above receiver tuple p. Does NOT pour.",
 "align_target":"api.align_target(p): refine held vessel position over receiver tuple p. Does NOT pour.",
 "tip_pour":"api.tip_pour(): rotate CLOSED hand by the configured pouring angle and hold. No arguments. It only rotates; it never moves contents or sets success.",
 "settle":"api.settle(): wait with grip commands preserved. No arguments. Do not release the vessel."
}
DOCS={"pick":PICK_DOCS,"place":PLACE_DOCS,"park":PARK_DOCS,"pour":POUR_DOCS}
DESCRIPTIONS={
 "pick":"Pick up the bound object. Initially the hand is empty. Finish holding the object in the air; measure its grasp offset. Reading a point and approaching it alone do NOT close fingers or lift.",
 "place":"Place the object ALREADY HELD in the bound arm onto the bound destination surface. Finish with object resting there, gripper OPEN and hand retracted vertically. Do not grasp another object.",
 "park":"The arm is EMPTY. Move it to the bound parking pose to clear the shared workspace.",
 "pour":"The bound arm ALREADY HOLDS a vessel containing a sphere. Move and align the held vessel over the target receiver, then tilt to pour out the sphere and wait. Finish still holding the vessel tilted; do NOT release or return it."
}

def op(name,kind,arm,obj,dest,pre,add,delete):
    return {"name":name,"args":{"arm":str(arm),"obj":obj,"dest":dest},"program_kind":kind,
            "pre":pre,"add":add,"del":delete,"description":DESCRIPTIONS[kind]}

def relay_objects(config=None):
    p=(config or {}).get("relay",{})
    return [{"name":"box0","start_xy":p.get("start_xy",[.29,.18]),"destination_xy":p.get("destination_xy",[.30,-.18])}]+p.get("extra_objects",[])


def domain(task,config=None):
    def hold(i,o):return f"holding({i},{o})"
    def empty(i):return f"empty({i})"
    def park(i):return op("park","park",i,"none","park",[empty(i)],[f"parked({i})"],[])
    if task=="relay":
        rows=relay_objects(config);out=[];multi=len(rows)>1
        for row in rows:
            o=row["name"]
            out.extend([op("pick_source","pick",0,o,"source",[empty(0),f"at_source({o})"]+(["station_free()"] if multi else []),[hold(0,o)],[empty(0),f"at_source({o})","parked(0)"]),
              op("place_station","place",0,o,"station",[hold(0,o)]+(["station_free()","parked(1)"] if multi else []),[empty(0),f"at_station({o})"],[hold(0,o)]+(["station_free()"] if multi else [])),
              op("pick_station","pick",1,o,"station",[empty(1),f"at_station({o})","parked(0)"],[hold(1,o)]+(["station_free()"] if multi else []),[empty(1),f"at_station({o})","parked(1)"]),
              op("place_destination","place",1,o,"destination",[hold(1,o)],[empty(1),f"at_destination({o})"],[hold(1,o)])])
        if not multi:return [out[0],out[1],park(0),out[2],out[3],park(1)]
        return out+[park(0),park(1)]
    out=[]
    for i in (0,1):
        o=f"item{i}" if task=="packing" else "can"+("L","R")[i]
        src=f"at_source({o})";inside=f"content_in_source({i})"
        out.append(op("pick_source","pick",i,o,"source",[empty(i),src]+([inside] if task=="pour" else []),[hold(i,o)],[empty(i),src,f"parked({i})"]))
        if task=="packing":
            out.append(op("place_tray","place",i,o,"tray",[hold(i,o),f"slot_free({i})"],[empty(i),f"packed({i})"],[hold(i,o),f"slot_free({i})"]))
            out.append(park(i))
        else:
            out.append(op("pour_receiver","pour",i,o,"cup"+("L","R")[i],[hold(i,o),inside],[f"filled({i})"],[inside]))
    return out

def goals(task,config=None):
    if task=="relay":return {f"at_destination({r['name']})" for r in relay_objects(config)}|{"parked(0)","parked(1)"}
    if task=="packing":return {"packed(0)","packed(1)","parked(0)","parked(1)"}
    return {"filled(0)","filled(1)"}

class Runtime:
    def __init__(self,config,task,out,seed=0):
        self.scout=Curriculum(deepcopy(config),task,out,seed)
        self.config=config;self.task=task;self.env=self.scout.env;self.p=config[task];self.h=config["table"]["top_z"]
        self.initial={o:self.scout.pos(o).copy() for o in self.env.obj_names}
        self.goal=goals(task,config);self.pour_events=[]
    def park_xyz(self,i):
        xy=self.p["donor_park_xy"] if self.task=="relay" and i==0 else [self.config["robots"][i]["base_xy"][0],.28 if i==0 else -.28]
        return np.array([*xy,self.h+.32])
    def filled(self,i):
        s=("L","R")[i];b=self.scout.pos("ball"+s);c=self.env.get_body_pos("cup"+s);r=self.p["ball_radius"]
        return bool(np.linalg.norm(b[:2]-c[:2])+r<self.p["cup_inner_radius"]-.002 and
                    b[2]-r>=self.h+self.p["cup_floor"]-.001 and b[2]+r<self.h+self.p["cup_height"])
    def content_in_source(self,i):
        s=("L","R")[i];can="can"+s;ball="ball"+s
        rot=self.env.d.xmat[self.env.obj_bid[can]].reshape(3,3)
        local=rot.T@(self.scout.pos(ball)-self.scout.pos(can));local[0]+=self.p["handle_offset"];local[2]+=self.p["handle_height"]
        r=self.p["ball_radius"]
        return bool(np.linalg.norm(local[:2])+r<self.p["can_inner_radius"]+.001 and
                    local[2]-r>=self.p["can_floor"]-.001 and local[2]+r<self.p["can_height"]+.002)
    def packed(self,i):
        return containment(self.env.get_object_pose(f"item{i}"),(self.env.get_body_pos("tray"),np.array([1.,0,0,0])),
                           self.p["item_half_size"],self.p["tray_inner_half_size"],self.p["tray_wall"],self.p["tray_height"])["inside"] and self.scout.released(f"item{i}")
    def slot_free(self,i):
        center=self.env.get_body_pos("tray")[:2]+self.p["slots_xy"][i]
        half=np.asarray(self.p["item_half_size"])
        for j in (0,1):
            o=f"item{j}";p=self.scout.pos(o)
            if self.scout.released(o) and np.all(np.abs(p[:2]-center)<2*half[:2]+.002) and abs(p[2]-(self.h+self.p["tray_wall"]+half[2]))<.020:
                return False
        return True
    def initial_task_ready(self):
        if self.task=="relay":return all(f"at_source({r['name']})" in self.state() and f"at_destination({r['name']})" not in self.state() for r in relay_objects(self.config))
        if self.task=="packing":return not any(self.packed(i) for i in (0,1))
        return all(self.content_in_source(i) and not self.filled(i) for i in (0,1))
    def state(self):
        state=set()
        for i,a in enumerate(self.env.arms):
            attached=[o for o in self.env.obj_names if a.is_holding(o)]
            if not attached:
                state.add(f"empty({i})")
                # parked means TCP outside the central corridor, not exact point
                # tracking. Holding joint commands can relax >2cm after arrival;
                # movement farther out of the corridor must not revoke clearance.
                ee=a.get_ee_pose()[0];side=1.0 if i==0 else -1.0
                if side*ee[1]>=.20 and ee[2]>=self.h+.20 and abs(ee[0]-.30)<.10:
                    state.add(f"parked({i})")
            for o in attached:
                if self.scout.pos(o)[2]>self.initial[o][2]+.070:state.add(f"holding({i},{o})")
        objects=[r["name"] for r in relay_objects(self.config)] if self.task=="relay" else (["item0","item1"] if self.task=="packing" else ["canL","canR"])
        for o in objects:
            now=self.scout.pos(o)
            if self.scout.released(o) and np.linalg.norm(now[:2]-self.initial[o][:2])<.020 and abs(now[2]-self.initial[o][2])<.008:
                state.add(f"at_source({o})")
        if self.task=="relay":
            for r in relay_objects(self.config):
                for kind,xy in (("station",self.p["station_xy"]),("destination",r["destination_xy"])):
                    if self.scout.near_table(r["name"],xy,self.p["object_half_size"][2]):state.add(f"at_{kind}({r['name']})")
            # Shared station availability is physical occupancy, not a cached flag.
            if all(np.linalg.norm(self.scout.pos(o)[:2]-self.p["station_xy"])>.060 or self.scout.pos(o)[2]>self.h+.12 for o in objects):
                state.add("station_free()")
        elif self.task=="packing":
            for i in (0,1):
                if self.packed(i):state.add(f"packed({i})")
                if self.slot_free(i):state.add(f"slot_free({i})")
        else:
            for i in (0,1):
                if self.content_in_source(i):state.add(f"content_in_source({i})")
                if self.filled(i):state.add(f"filled({i})")
        return state
    def api(self,m):
        i=int(m["args"]["arm"]);o=m["args"]["obj"];kind=m["program_kind"]
        sk=self.scout.skills[i];a=self.env.arms[i]
        def target_pose():
            if kind=="pick":p=self.scout.pos(o).copy();p[2]+=.015
            elif kind=="park":p=self.park_xyz(i)
            elif kind=="pour":
                xy=self.env.get_body_pos(m["args"]["dest"])[:2]+self.p["can_offset_xy"];p=np.array([*xy,self.h+.29])
            elif self.task=="relay":
                xy=self.p["station_xy"] if m["args"]["dest"]=="station" else next(r["destination_xy"] for r in relay_objects(self.config) if r["name"]==o)
                p=np.array([*xy,self.h])
            else:p=np.array([*(self.env.get_body_pos("tray")[:2]+self.p["slots_xy"][i]),self.h+self.p["tray_wall"]])
            return tuple(map(float,p))
        def go(p,grip):
            error=sk.move_to(p,grip=grip,tol=.007,max_steps=140,ki=.5)
            if error>.04:raise RuntimeError(f"pose residual {error:.4f}m; no state correction")
            return float(error)
        def coordinates(p,min_size=2):
            if not isinstance(p,(list,tuple)) or len(p)<min_size or not all(type(x) in (int,float) and math.isfinite(x) for x in p):
                raise ValueError(f"Expected coordinate tuple of >= {min_size} finite numbers; got {p!r}. Use target_pose(), not record_grab_offset() which returns None.")
        def approach(xyz):coordinates(xyz,3);return go([xyz[0],xyz[1],self.h+.24],1.0)
        def align_xy(xyz):return approach(xyz)
        def descend_to(z,lock_xy):coordinates(lock_xy);return go([lock_xy[0],lock_xy[1],z],1.0)
        def grasp(xy):
            coordinates(xy)
            # Legacy place_release disables assistance until rehome. Restore only
            # this already-declared assisted mode for a new grasp, WITHOUT rehome,
            # directly activating a weld, or changing any object pose.
            if not getattr(a,"_grasp_enabled",True):
                self.scout.events.append({"event":"reenable_declared_grasp_assistance","arm":i,"sim_time":float(self.env.d.time)})
                a._grasp_enabled=True
            sk.grasp(xy[:2],n=28)
        def lift():sk.lift(top=self.h+.30)
        def record_grab_offset():sk.record_grab_offset(o)
        def move_above(p):
            now=self.scout.pos(o);ee=sk.ee();ee[:2]+=np.asarray(p[:2])-now[:2];ee[2]=p[2] if kind=="pour" else self.h+.30
            return go(ee,0.0)
        def align_target(p):
            return tuple(map(float,sk.refine_above(np.asarray(p[:2]),high_z=p[2] if kind=="pour" else self.h+.30,obj_name=o)))
        def lower(p):
            sk.record_grab_offset(o)
            return sk.descend_place_tracked(np.asarray(p[:2]),p[2],o,gap=.003,suppress_feedforward=True)
        def place_release():sk.place_release()
        def retreat():p=sk.ee();return go([p[0],p[1],self.h+.30],1.0)
        def settle():self.scout.hold(16)
        def move_to(p):
            if any(a.is_holding(n) for n in self.env.obj_names):raise RuntimeError("park move requires empty hand")
            return go(p,1.0)
        def tip_pour():
            before=self.content_in_source(i);already=self.filled(i)
            event={"arm":i,"before_content_in_source":before,"before_receiver_filled":already}
            self.pour_events.append(event)
            if not before or already:raise RuntimeError("content not in source immediately before pouring")
            sk.tip_pour_joint(axis=self.p["tilt_axis"],total_angle=self.p["tilt_radians"],n=90,hold=100,can_name=o)
            event["after_receiver_filled"]=self.filled(i)
        all_api=locals()
        return {name:all_api[name] for name in DOCS[kind]}
    def task_evidence_ok(self):
        if self.task!="pour":return True
        arms={e["arm"] for e in self.pour_events if e["before_content_in_source"] and
              not e["before_receiver_filled"] and e.get("after_receiver_filled")}
        return arms=={0,1} and all(self.env.arms[i].is_holding("can"+("L","R")[i]) for i in (0,1))
    def stable_goal(self):
        return self.scout.stable(lambda:self.goal<=self.state() and self.task_evidence_ok())
    def save(self,path):
        path.mkdir(parents=True,exist_ok=True)
        self.env.save_png(str(path/"final.png"));self.env.save_png(str(path/"final_front.png"),"front")
        self.env.save_video(str(path/"rollout.mp4"),fps=30)
    def close(self):
        if self.env.renderer:self.env.renderer.close()
