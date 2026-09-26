"""Frozen Blocks task: transferred genuine Cover model schemas and programs,
real fine-grained actuator calls, Qwen RGB localization, strict grasp, no weld.
"""
from copy import deepcopy
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from core.cover_cabto_api import FineCover,Perception,intersect_plane,API_DOCS
from core.cover_visual_localizer import install_camera
from core.cover_cabto_grounding import save

PAIRS=[("yellow","green"),("red","blue")]

class BlocksPerception:
    def __init__(self,r,out,client):
        self.r=r;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.client=client;self.records=[]
        self.camera_change=install_camera(r.env);save(self.out/"camera.json",self.camera_change);self.cache={}
    def prepare_supports(self):
        for name in ("green","blue"):
            z=self.r.config["table_top"]+3*self.r.config["cube_half_size"]+.003
            self.cache[name]=self.locate(name,z).copy()
    def locate(self,name,z):
        if name in self.cache:
            p=self.cache[name].copy();p[2]=z
            self.records.append({"object":name,"source":"initial_pre_occlusion_Qwen_RGB_support_cache","xyz":p.tolist(),"fallback":False})
            return p
        image=self.r.env.render_cam("front").copy();size=image.shape[0];p=self.out/f"{len(self.records):03d}_{name}.png";Image.fromarray(image).save(p)
        call=self.client.call(f"Point to the {name} cube. Output the pixel coordinate.",[p],max_tokens=180)
        row={"object":name,"call":call,"height_prior":z,"source":"Qwen point + RGB top face + calibrated known plane","fallback":False};self.records.append(row)
        try:
            point,field=Perception.parse_pixel(call,size);u,v=point;radius=56
            x0,y0=max(0,int(u)-radius),max(0,int(v)-radius);x1,y1=min(size,int(u)+radius+1),min(size,int(v)+radius+1)
            hsv=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_RGB2HSV);h,s,w=[hsv[:,:,i] for i in range(3)]
            mask={"yellow":(h>17)&(h<40),"green":(h>38)&(h<95),"blue":(h>90)&(h<135),"red":(h<12)|(h>167)}[name]&(s>85)&(w>55)
            n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
            candidates=[(np.linalg.norm(centers[i]+[x0,y0]-point),i) for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>80]
            candidates.sort()
            if not candidates or candidates[0][0]>32:raise ValueError("No matching cube RGB region near Qwen point")
            if len(candidates)>1 and candidates[1][0]-candidates[0][0]<5:raise ValueError("Ambiguous cube regions")
            component=(labels==candidates[0][1]).astype(np.uint8)
            if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():raise ValueError("Cube ROI clipped")
            # Visible top-face xy center; physical cube half-size is a disclosed height prior.
            contour=max(cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
            rectangle=cv2.minAreaRect(contour);uv=np.array(rectangle[0])+[x0,y0]
            top_z=self.r.config["table_top"]+2*self.r.config["cube_half_size"]
            xyz=intersect_plane(self.r.env,uv,"front",size,top_z);xyz[2]=z
            row.update(coarse_uv=point.tolist(),uv=uv.tolist(),xyz=xyz.tolist(),top_plane_z=top_z,roi=[x0,y0,x1,y1])
            draw=Image.fromarray(image);d=ImageDraw.Draw(draw)
            for pt,c in [(point,(255,195,90)),(uv,(75,225,175))]:
                x,y=pt;d.line((x-7,y,x+7,y),fill=c,width=2);d.line((x,y-7,x,y+7),fill=c,width=2)
            draw.save(p.with_name(p.stem+"_overlay.png"));return xyz
        except Exception as exc:row["error"]=str(exc);raise
        finally:save(self.out/"perception.json",self.records)

class FineBlocks(FineCover):
    def __init__(self,config,seed=0,render=True,perception="oracle",out=".",client=None):
        super().__init__(config,seed,render,perception,out,client)
        if perception!="oracle":self.perception=BlocksPerception(self.r,out,client)
        self.backend=perception;self.env=self.r.env;self.goal={"at_target(yellow)","at_target(red)","empty()","home_completed()"}
    def state(self):
        r=self.r;s=set()
        if r.released():s.add("empty()")
        for o,support in PAIRS:
            if r.held(o) and r.pos(o)[2]>r.initial[o][2]+.075:s.add(f"holding({o})")
            elif r.assess_one((o,support))["ok"]:s.add(f"at_target({o})")
            elif np.linalg.norm(r.pos(o)-r.initial[o])<.009:s.add(f"at_source({o})")
        if r.assess()["final_home"]:s.add("home_completed()")
        return s
    def api(self,m):return self.bind(m)
    def grasp_point(self):
        if self.backend=="oracle":return tuple(map(float,self.r.grip_point(self.obj)))
        z=self.r.config["table_top"]+self.r.config["cube_half_size"]+.004
        return tuple(map(float,self.perception.locate(self.obj,z)))
    def destination_point(self):
        r=self.r
        if self.backend=="oracle":p=r.pos(self.support)+[0,0,2*r.config["cube_half_size"]+.003]
        else:p=self.perception.locate(self.support,r.config["table_top"]+3*r.config["cube_half_size"]+.003)
        return tuple(map(float,p))
    def stable_goal(self):
        self.r.mark("final_stability");ok=True
        for _ in range(round(self.r.c["stable_seconds"]/(self.r.dt*self.r.steps))):
            self.r.wait(self.r.dt*self.r.steps,1.);ok=ok and self.goal<=self.state()
        return ok and self.final_check(self.goal)["success"]
    def save(self,path):
        p=Path(path);self.env.save_png(str(p/"final.png"));self.env.save_png(str(p/"final_front.png"),"front");self.env.save_video(str(p/"rollout.mp4"),fps=30)
        save(p/"metrics.json",self.r.assess());save(p/"perception.json",self.perception.records)

def task_set():
    initial={"empty()","at_source(yellow)","at_source(red)"}
    return [{"id":name,"initial":sorted(initial),"goal":sorted({"empty()","home_completed()"}|{f"at_target({o})" for o in objects})}
            for name,objects in [("B1",["yellow"]),("B2",["red"]),("B3",["yellow","red"])]]

def transferred_models(package):
    path=Path(package)/"outputs/cover_cabto_fix/proposal_final_v2/models.json";d=json.loads(path.read_text());schemas={r["name"]:r for r in d["schemas"]};models=[]
    for o,s in PAIRS:
        for kind in ("pick","place"):
            models.append({"name":kind,"args":{"object":o,"support":s},"program_kind":kind,
                           **{k:[a.replace("OBJECT",o) for a in schemas[kind][k]] for k in ("pre","add","del")},
                           "description":("Grasp the bound cube from table and leave it held in air" if kind=="pick" else "Stack held cube on bound support cube, release and withdraw. Do not return home.")})
    models.append({**schemas["home"],"args":{},"program_kind":"home","description":"Return empty arm home without moving cubes"})
    return models,path
