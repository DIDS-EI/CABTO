"""Sequential dual-task reference scouts without object pose snapping or success flags.

Still grasp-weld assisted and oracle-guided. These are not generated policies and
not strict contact-grasp proofs. Stops at measured failures; never repairs results
by writing object qpos. All target coordinates are read from the compiled scene.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import types
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"CABTO"/"exp4_bt_tasks"/"stage2_scripted"))
from exp4_env import Exp4Env
from arm_skills import ArmSkills
import scene_handover as sh
import scene_pour as sp
import scene_storage as ss
from probe_five_scene_preflight import candidate_xml

def containment(item_pose, container_pose, half, inner, floor=.008, ceiling=.060):
    ip,iq=item_pose;cp,cq=container_pose
    ir=Rotation.from_quat(np.asarray(iq)[[1,2,3,0]]).as_matrix()
    cr=Rotation.from_quat(np.asarray(cq)[[1,2,3,0]]).as_matrix()
    corners=np.asarray(list(itertools.product((-1,1),repeat=3)))*half
    local=((corners@ir.T+ip)-cp)@cr
    lo,hi=local.min(0),local.max(0)
    inside=bool(np.all(lo[:2]>=-np.asarray(inner)) and np.all(hi[:2]<=inner)
                and lo[2]>=floor-.002 and hi[2]<=ceiling+.003)
    return {"inside":inside,"bounds_in_container":[lo.tolist(),hi.tolist()]}

class Scout:
    def __init__(self,task,out,pour_offset=-.065):
        self.task=task;self.out=out;self.pour_offset=pour_offset;out.mkdir(parents=True,exist_ok=True)
        module={"handover":sh,"pour":sp,"storage":ss}[task]
        names={"handover":["box0","box1","box2"],"pour":["canL","canR","ballL","ballR"],
               "storage":["item_g1","item_g2","carton"]}[task]
        self.env=Exp4Env(candidate_xml(module.build_scene_xml(),task),[("_L",-math.pi/2),("_R",math.pi/2)],names,render=True,img_size=560)
        self.env.reset();self.env.start_record();self.env.record_frame("overview")
        self.events=[];self.steps=[];self.env.save_png(str(out/"before.png"))
        self.arms=self.env.arms
        self.skills=[ArmSkills(self.env,a,recorder=lambda:self.env.record_frame("overview"),rec_every=2,
                       block_half={"handover":sh.BOX_HALF[2],"pour":sp.CAN_HH,"storage":ss.ITEM_HALF[2]}[task]) for a in self.arms]
        for a in self.arms:
            a.grasp_z_tol=.09
            original=a._set_weld
            def logged(arm,name,active,_original=original):
                self.events.append({"arm":arm.s,"object":name,"active":bool(active),
                    "object_position":self.pos(name).tolist(),"finger_mid":arm._finger_mid().tolist(),
                    "sim_time":float(self.env.d.time),"trigger":"legacy_grasp_assist_or_release"})
                return _original(name,active)
            a._set_weld=types.MethodType(logged,a)
        self.result={"task":task,"source":"handwritten_diagnostic_reference","perception":"oracle",
                     "control":"grasp_weld_assisted","object_snap":False,"arm_rehome_during_execution":False,
                     "steps":self.steps,"assistance_events":self.events,"success":False,
                     "scope":"single object handover / bidirectional sphere pour / packing only", "failure":None}

    def pos(self,name):return self.env.get_object_pose(name)[0]
    def mark(self,phase,**fields):
        row={"phase":phase,"sim_time":float(self.env.d.time),"positions":{n:self.pos(n).tolist() for n in self.env.obj_names},
             "quaternions_wxyz":{n:self.env.get_object_pose(n)[1].tolist() for n in self.env.obj_names},
             "hand_poses":{a.s:a.get_ee_pose()[0].tolist() for a in self.arms},**fields}
        self.env.save_png(str(self.out/f"phase_{len(self.steps)}_{phase}.png"),"front")
        self.steps.append(row);print(self.task,phase,fields,flush=True)
    def go(self,i,xyz,grip):
        error=self.skills[i].move_to(xyz,grip=grip,tol=.007,max_steps=140,ki=.5)
        self.mark("move",arm=i,target=list(map(float,xyz)),error_m=error)
        if error>.04:raise RuntimeError(f"arm {i} target residual {error:.3f}m exceeds 4cm; do not continue blindly")
    def pick(self,i,name,z):
        sk=self.skills[i];p=self.pos(name).copy()
        self.go(i,[p[0],p[1],.64],1.0);self.go(i,[p[0],p[1],z],1.0)
        sk.grasp(self.pos(name)[:2],n=28)
        if not self.arms[i].is_holding(name):raise RuntimeError(f"{name} not held after closing")
        sk.lift(top=.70);sk.record_grab_offset(name)
        if not self.arms[i].is_holding(name) or self.pos(name)[2]<.49:raise RuntimeError(f"{name} failed lift")
        self.mark("picked",object=name,arm=i)
    def transport(self,i,name,xy,z=.70):
        sk=self.skills[i];p=self.pos(name);ee=sk.ee();target=ee.copy();target[:2]+=np.asarray(xy)-p[:2];target[2]=z
        self.go(i,target,0.0)
        sk.refine_above(np.asarray(xy),high_z=z,obj_name=name)
    def place(self,i,name,xy,top):
        sk=self.skills[i]
        self.transport(i,name,xy)
        sk.record_grab_offset(name)
        sk.descend_place_tracked(np.asarray(xy),top,name,gap=.003,suppress_feedforward=True)
        sk.place_release();sk.retreat(dz=.15)
        self.mark("placed",object=name,arm=i)
    def hold(self,n=32):
        for _ in range(n):
            grips={a.s:a._grip_cmd for a in self.arms};self.env.hold_arms(grips,sub_steps=16);self.env.record_frame("overview")

    def handover(self):
        name="box0";self.pick(0,name,self.pos(name)[2]+.005)
        self.transport(0,name,[.30,0.0],z=.70)
        bp=self.pos(name).copy();aR=self.arms[1];skR=self.skills[1]
        offset=aR.get_ee_pose()[0]-aR._finger_mid()
        # Position the receiving hand around the actual object, not vice versa.
        target=bp+offset
        self.go(1,[target[0],target[1]-.10,target[2]+.06],1.0)
        self.go(1,target,1.0)
        skR.grasp(self.pos(name)[:2],n=8)
        right_holds=aR.is_holding(name);self.mark("receiver_close",right_holds=right_holds,
                    receiver_distance_m=float(np.linalg.norm(aR._finger_mid()-self.pos(name))))
        if not right_holds:raise RuntimeError("receiving grasp not established; left hand not released")
        # Normal open command releases donor only after measured receiver hold.
        self.arms[0]._grip_cmd=1.0;self.hold(16)
        self.go(0,[.30,.25,.74],1.0)
        if not aR.is_holding(name):raise RuntimeError("receiver lost payload after donor release")
        skR.record_grab_offset(name);self.place(1,name,sh.PLACE_XYS[0],.40);self.hold()
        p=self.pos(name);off=float(np.linalg.norm(p[:2]-sh.PLACE_XYS[0]))
        self.result["single_object_success"]=bool(off<.02 and abs(p[2]-(.4+sh.BOX_HALF[2]))<.012)
        self.result["success"]=False # Three-object task not yet tested by this single-object scout.
        self.result["remaining"]="repeat for all three objects and verify genuine contact transfer; weld hold alone is not proof"

    def pour(self):
        start={n:self.pos(n).copy() for n in ("canL","canR")};inside=[]
        for i,can,ball,cup in ((0,"canL","ballL","cupR"),(1,"canR","ballR","cupL")):
            self.pick(i,can,self.pos(can)[2]+sp.CAN_HH*.4)
            sk=self.skills[i];c=self.env.get_body_pos(cup);target=c[:2]+np.array([self.pour_offset,0.])
            self.transport(i,can,target,z=.61)
            ok,pred=sk.tip_pour_joint(axis="ny",total_angle=math.radians(120),n=70,grip=0.0,hold=100,can_name=can)
            q=self.env.get_object_pose(can)[1]
            self.mark("tilted",object=can,predicted_tilt_ok=bool(ok),predicted_axis=float(pred),
                      actual_can_normal=Rotation.from_quat(q[[1,2,3,0]]).apply([0,0,1]).tolist())
            sk.untip_joint(grip=0.0) # no retreat/open before putting can back
            self.place(i,can,start[can][:2],.404);self.hold()
            p=self.pos(ball);radial=float(np.linalg.norm(p[:2]-c[:2]))
            valid=radial+sp.CONTENT_R<sp.CUP_R_IN+.001 and p[2]-sp.CONTENT_R>.40+.006 and p[2]+sp.CONTENT_R<sp.CUP_TOP+.003
            inside.append(bool(valid));self.mark("content_check",ball=ball,cup=cup,fully_inside=bool(valid),radial_m=radial)
            if not valid:raise RuntimeError(f"{ball} not fully inside {cup}; no snapping fallback")
        self.result["success"]=all(inside);self.result["scope"]="two-direction sphere transfer, not liquid dynamics"

    def storage(self):
        carton=self.env.get_body_pos("carton");pack=[]
        for i,name,dy in ((0,"item_g1",.026),(1,"item_g2",-.026)):
            self.arms[i]._grasp_enabled=True
            self.pick(i,name,self.pos(name)[2]+.015)
            current=self.pos("carton");xy=current[:2]+np.array([0.,dy])
            self.place(i,name,xy,float(current[2]+ss.CARTON_WALL))
            self.hold();item=self.pos(name);c=self.pos("carton");relative=item-c
            ok=abs(relative[0])+ss.ITEM_HALF[0]<ss.CARTON_IN[0] and abs(relative[1])+ss.ITEM_HALF[1]<ss.CARTON_IN[1] and 0<relative[2]<2*ss.CARTON_H
            pack.append(bool(ok));self.mark("packing_check",object=name,inside=bool(ok),relative=relative.tolist())
            if not ok:raise RuntimeError(f"{name} not stably inside carton; no snap/pin fallback")
        checks={};stable=True
        for _ in range(32):
            self.hold(1)
            checks={name:containment(self.env.get_object_pose(name),self.env.get_object_pose("carton"),
                    ss.ITEM_HALF,ss.CARTON_IN) for name in ("item_g1","item_g2")}
            stable=stable and all(r["inside"] for r in checks.values())
        self.result["final_containment"]=checks;self.result["packing_stable_seconds"]=1.024
        self.result["packing_success"]=bool(all(pack) and stable);self.result["success"]=False
        self.result["remaining"]="true two-arm carton lifting and transport to shelf not yet implemented; single-arm lift is not a substitute"

    def run(self):
        try:getattr(self,self.task)()
        except Exception as e:self.result["failure"]={"type":type(e).__name__,"message":str(e),"attribution":"unresolved_controller_or_geometry"}
        finally:
            self.result["final_positions"]={n:self.pos(n).tolist() for n in self.env.obj_names}
            self.env.save_png(str(self.out/"after.png"));self.env.save_png(str(self.out/"after_front.png"),"front")
            self.env.save_video(str(self.out/"rollout.mp4"),fps=30)
            (self.out/"result.json").write_text(json.dumps(self.result,indent=2,ensure_ascii=False))
            self.env.renderer.close()
            print("SCOUT_FINAL",self.task,self.result["success"],self.result["failure"],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("task",choices=("handover","pour","storage"));ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--pour-offset",type=float,default=-.065,help="x offset of upright can relative to cup; diagnostic parameter")
    a=ap.parse_args();Scout(a.task,a.output,a.pour_offset).run()
if __name__=="__main__":main()
