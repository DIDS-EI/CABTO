"""New, task-local single-arm scenes and time-parametrized control.

Only Panda assets/kinematics and actuator plumbing are reused. No old Cover or
Blocks task controllers, default positions, repeated rehome, or PI delta loop.
Initialization may reset; execution only changes actuator targets and explicitly
reported grasp constraints. The evaluator reads physics independently.
"""
from copy import deepcopy
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / "CABTO/exp4_bt_tasks/stage2_scripted"
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(ROOT.parent / "CABTO/exp4_bt_tasks/scenes"))
from exp4_env import Exp4Env, HOME_QPOS
from panda_body import panda_asset_block, panda_default_block, panda_body, panda_tendon_equality_actuator

COLORS = {"red": ".82 .035 .05 1", "blue": ".035 .08 .85 1", "yellow": ".96 .84 .035 1", "green": ".13 .74 .07 1"}


def vec(x): return " ".join(f"{float(v):.8f}" for v in x)

def ease(t): return t*t*t*(10.0 + t*(-15.0 + 6.0*t))

def camera(name, eye, target):
    eye, target = np.array(eye), np.array(target)
    z = eye-target; z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return f'<camera name="{name}" pos="{vec(eye)}" xyaxes="{vec(np.r_[x,y])}" fovy="43"/>'


def ring_mesh(name, levels, n=64):
    vertices=[]
    for radius,z in levels:
        vertices += [[radius*math.cos(2*math.pi*j/n),radius*math.sin(2*math.pi*j/n),z] for j in range(n)]
    faces=[]
    for k in range(len(levels)-1):
        for j in range(n):
            a=k*n+j;b=k*n+(j+1)%n;c=b+n;d=a+n
            faces.extend([[a,b,c],[a,c,d]])
    return f'<mesh name="{name}" vertex="{vec(np.asarray(vertices).ravel())}" face="{" ".join(str(v) for f in faces for v in f)}"/>'


def scene(config, seed=0):
    c=deepcopy(config);h=c["table_top"];rng=np.random.default_rng(seed)
    def xy(v): return np.asarray(v)+ (rng.uniform(-c["jitter_m"], c["jitter_m"], 2) if seed else 0)
    names=[];objects=[];welds=[];meshes=[]
    def free(name, pos, geoms, graspable=True):
        names.append(name)
        objects.append(f'<body name="{name}" pos="{vec(pos)}"><freejoint name="{name}_free"/>{geoms}</body>')
        if graspable:
            welds.append(f'<weld name="grasp_weld_{name}" body1="hand" body2="{name}" active="false" solref=".006 1" solimp=".95 .99 .001"/>')
    if c["task"]=="blocks":
        r=c["cube_half_size"]
        for name,p in c["blocks"].items():
            free(name,[*xy(p),h+r+.001],f'<geom name="{name}_cube" type="box" size="{r} {r} {r}" rgba="{COLORS[name]}" mass=".05" friction="1 .02 .001"/>')
    else:
        # Kitchen sorting scene: upright bowl, chopping board and pan in the back
        # row; shrimp, apple and potato in the front row.  All three foods are
        # free bodies and task targets.  Supports are fixed and cannot be moved
        # by a lucky collision.
        supports={name:np.asarray(pos,float) for name,pos in c["supports"].items()}
        bowl=[]
        bowl.append('<geom name="bowl_floor" type="cylinder" size=".070 .004" pos="0 0 .004" material="white" friction="1 .02 .002"/>')
        for j in range(32):
            a=2*math.pi*j/32
            bowl.append(f'<geom name="bowl_wall_{j}" type="box" size=".004 .007 .025" pos="{.069*math.cos(a)} {.069*math.sin(a)} .029" quat="{math.cos(a/2)} 0 0 {math.sin(a/2)}" material="white" friction="1 .02 .002"/>')
        objects.append(f'<body name="bowl" pos="{supports["bowl"][0]} {supports["bowl"][1]} {h}">{"".join(bowl)}</body>')
        objects.append(f'<body name="board" pos="{supports["board"][0]} {supports["board"][1]} {h+.012}"><geom name="board_top" type="box" size=".080 .060 .012" material="wood" friction="1.2 .02 .002"/><geom type="cylinder" size=".010 .013" pos="-.061 0 .001" quat=".7071068 .7071068 0 0" rgba=".28 .16 .07 1"/></body>')
        pan=['<geom name="pan_floor" type="cylinder" size=".073 .005" pos="0 0 .005" material="dark" friction="1 .02 .002"/>']
        for j in range(32):
            a=2*math.pi*j/32
            pan.append(f'<geom name="pan_wall_{j}" type="box" size=".004 .0075 .018" pos="{.071*math.cos(a)} {.071*math.sin(a)} .021" quat="{math.cos(a/2)} 0 0 {math.sin(a/2)}" material="dark" friction="1 .02 .002"/>')
        pan.append('<geom name="pan_handle" type="box" size=".012 .060 .009" pos="0 .126 .020" material="dark"/>')
        objects.append(f'<body name="pan" pos="{supports["pan"][0]} {supports["pan"][1]} {h}">{"".join(pan)}</body>')

        items={name:xy(pos) for name,pos in c["items"].items()}
        shrimp_geoms=['<geom name="shrimp_body" type="ellipsoid" size=".022 .014 .010" pos="0 0 .011" rgba=".96 .42 .18 1" mass=".012" condim="6" friction="2 .03 .003"/>',
                      '<geom name="shrimp_curve" type="ellipsoid" size=".013 .009 .008" pos="-.020 .006 .010" quat=".9239 0 0 .3827" rgba="1 .51 .23 1" mass=".003" condim="6" friction="2 .03 .003"/>',
                      '<geom name="shrimp_tail_left" type="ellipsoid" size=".010 .004 .004" pos="-.033 .013 .010" quat=".9239 0 0 .3827" rgba="1 .66 .34 1" mass=".001"/>',
                      '<geom name="shrimp_tail_right" type="ellipsoid" size=".010 .004 .004" pos="-.036 .004 .010" quat=".9239 0 0 -.3827" rgba="1 .66 .34 1" mass=".001"/>']
        free("shrimp",[*items["shrimp"],h+.001],"".join(shrimp_geoms))
        ar=.025;az=.021
        free("apple",[*items["apple"],h+az],f'<geom name="apple_flesh" type="ellipsoid" size="{ar} {ar} {az}" rgba=".72 .018 .028 1" mass="0" contype="0" conaffinity="0"/><geom name="apple_grasp_core" type="cylinder" size=".020 .018" rgba=".70 .016 .026 1" mass=".026" condim="6" friction="2 .03 .003"/><geom name="apple_flat_base" type="cylinder" size=".015 .002" pos="0 0 {-az+.002}" rgba=".69 .015 .024 1" mass=".002" condim="6" friction="2 .03 .003"/><geom name="apple_stem" type="cylinder" size=".002 .007" pos="0 0 {az}" rgba=".19 .09 .025 1" mass=".0001"/>')
        free("potato",[*items["potato"],h+.017],'<geom name="potato_skin" type="ellipsoid" size=".026 .021 .017" rgba=".79 .58 .25 1" mass="0" contype="0" conaffinity="0"/><geom name="potato_grasp_core" type="box" size=".021 .017 .015" rgba=".79 .58 .25 1" mass=".025" condim="6" friction="2 .03 .003"/><geom type="sphere" size=".0015" pos=".012 .012 .014" rgba=".30 .17 .06 1" mass=".0001"/><geom type="sphere" size=".0015" pos="-.010 -.013 .010" rgba=".30 .17 .06 1" mass=".0001"/>')
    assets=ROOT/"mujoco_workspace/franka_articulated/assets/panda_assets"
    tendon,equality,actuator=panda_tendon_equality_actuator()
    robot=ET.fromstring(panda_body(base_pos=c["robot_base"],pedestal=False))
    if c["control"]["gravity_compensation"]:
        for body in robot.iter("body"): body.set("gravcomp","1")
    # Omit bright diagnostic TCP sphere from rendered task.
    robot.find(".//site[@name='tcp']").set("rgba","0 0 0 0")
    floor=['<geom name="floor" type="plane" size="4 4 .1" rgba=".28 .59 .71 1"/>']
    for a in np.arange(-3,3.1,.5):
        floor.append(f'<geom type="box" size=".003 3 .0005" pos="{a} 0 .001" rgba=".88 .97 1 1" contype="0" conaffinity="0"/>')
        floor.append(f'<geom type="box" size="3 .003 .0005" pos="0 {a} .001" rgba=".88 .97 1 1" contype="0" conaffinity="0"/>')
    view_sign=-1 if c["task"]=="cover" else 1
    xml=f'''<mujoco model="paper_style_{c['task']}"><compiler angle="radian" meshdir="{assets}" autolimits="true"/>
<option timestep="{c['control']['dt']}" integrator="implicitfast" gravity="0 0 -9.81"/>
<visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/><headlight ambient=".32 .32 .32" diffuse=".6 .6 .6"/></visual>
<default><geom solref=".006 1" solimp=".95 .99 .001" friction=".9 .02 .001"/>{panda_default_block()}</default>
<asset><texture type="skybox" builtin="gradient" rgb1=".32 .61 .72" rgb2=".32 .61 .72" width="512" height="512"/>{panda_asset_block()}{''.join(meshes)}
<material name="white" rgba=".96 .96 .95 1"/><material name="off_white" rgba=".87 .90 .91 1"/><material name="black" rgba=".19 .21 .23 1"/><material name="green" rgba="0 .8 0 1"/><material name="light_blue" rgba=".12 .65 .83 1"/>
<material name="steel" rgba=".80 .84 .87 1" specular=".7" shininess=".6"/><material name="dark" rgba=".06 .07 .08 1" specular=".45"/><material name="wood" rgba=".83 .68 .44 1"/>
</asset><worldbody><light pos=".2 -1.0 2.5" dir="0 .2 -1" diffuse=".8 .8 .8"/><light pos="-1 1 2" diffuse=".45 .45 .45"/>
{''.join(floor)}<body name="table" pos="{vec(c['table_center'])}"><geom type="box" size="{vec(c['table_half_size'])}" rgba=".21 .22 .23 1"/></body>
<body name="pedestal" pos="0 0 .18"><geom type="cylinder" size=".085 .18" rgba=".20 .23 .26 1"/></body>
{ET.tostring(robot,encoding='unicode')}{''.join(objects)}
{camera('overview',[1.25,1.32*view_sign,1.55],[.43,0,.55])}{camera('front',[1.3,.9*view_sign,1.16],[.47,0,.50])}
</worldbody><tendon>{tendon}</tendon><equality>{equality}{''.join(welds)}</equality><actuator>{actuator}</actuator></mujoco>'''
    return xml,names


class RebuiltSingleArm:
    def __init__(self, config, seed=0, render=False):
        self.config=deepcopy(config);self.c=config["control"];self.task=config["task"]
        self.xml,names=scene(config,seed)
        self.env=Exp4Env(self.xml,[("",0.)],names,render=render,img_size=720)
        self.env.reset();self.arm=self.env.arms[0];self.arm._grasp_enabled=False
        self.dt=self.env.m.opt.timestep;self.steps=int(self.c["sample_steps"])
        self.phase="initial";self.target=None;self.current_support="table";self.events=[];self.trace=[];self.contacts={};self.contact_counts={}
        self.next_frame=float(self.env.d.time);self.frame_times=[];self.key_frames={}
        self.qprev=np.array(self.arm.arm_target);self.max_command_speed=0.;self.max_actual_speed=0.
        self.max_tracking=0.;self.preserved=True;self.completed=[];self.home_event=None
        self.transport_monitor=False;self.grasp_relative=None;self.max_grasp_slip=0.;self.current_contact_gap=0.;self.max_contact_gap=0.
        self.force_now={};self.strict_weld_violation=False;self.stability_ok=True
        self.robot_ids={i for i in range(self.env.m.nbody) if self.body(i).startswith("link") or self.body(i) in ("hand","left_finger","right_finger")}
        self.initial={n:self.env.get_object_pose(n)[0].copy() for n in names}
        supports={s for _,s in self.config["pairs"]}
        self.support_initial={s:self.pose(s)[0].copy() for s in supports}
        self.source_positions={**{n:p.copy() for n,p in self.initial.items()},**{n:p.copy() for n,p in self.support_initial.items()}}
        self.arm_ids=self.arm.arm_dofadr
        if render:self.env.start_record()
        self.sample()

    def body(self,i):return mujoco.mj_id2name(self.env.m,mujoco.mjtObj.mjOBJ_BODY,int(i)) or "world"
    def pose(self,n):
        if n in self.env.obj_bid:return self.env.get_object_pose(n)
        bid=mujoco.mj_name2id(self.env.m,mujoco.mjtObj.mjOBJ_BODY,n)
        if bid<0:raise KeyError(n)
        return self.env.d.xpos[bid].copy(),self.env.d.xquat[bid].copy()
    def pos(self,n):return self.pose(n)[0]
    def ee(self):return self.arm.get_ee_pose()[0]
    def active_welds(self):return [n for n,w in self.arm.obj_weld.items() if w>=0 and self.env.d.eq_active[w]]
    def released(self):return not self.active_welds() and self.arm._grip_cmd>=.5 and self.arm.get_gripper_width()>.065
    def held(self,n):
        if self.c["grasp_mode"]=="assisted":return n in self.active_welds()
        return self.target==n and self.arm._grip_cmd<.5 and np.linalg.norm(self.ee()-self.grip_point(n))<.025
    def grip_point(self,n):
        p,q=self.pose(n);r=self.env.d.xmat[self.env.obj_bid[n]].reshape(3,3)
        offset_z=self.config.get("grip_offset_z",{}).get(n,.004)
        return p+r@[0,0,offset_z]
    def mark(self,name):
        self.phase=name;self.events.append({"event":name,"time":float(self.env.d.time),"object_positions":{n:self.pos(n).tolist() for n in self.env.obj_names}})
        if self.env.render_enabled and name.startswith(("carry_","release_","controlled_home")):
            self.key_frames[name]=self.env.render_cam("overview").copy()
    def sample(self):
        m,d=self.env.m,self.env.d
        self.force_now={"left_finger":0.,"right_finger":0.}
        for ci,contact in enumerate(d.contact):
            a,b=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);ba,bb=self.body(a),self.body(b)
            ra,rb=a in self.robot_ids,b in self.robot_ids
            kind=None
            if ra!=rb:
                robot=ba if ra else bb;other=bb if ra else ba
                allowed=robot in ("left_finger","right_finger") and other==self.target
                if allowed:
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                    self.force_now[robot]+=max(0.,float(force[0]))
                    self.contact_counts[(self.target,robot)]=self.contact_counts.get((self.target,robot),0)+1
                    if contact.dist<-.002:kind="deep_grasp_contact"
                elif robot=="link0" and other in ("table","pedestal"):pass
                else:kind="robot_environment"
            elif ra and rb and a!=b and m.body_parentid[a]!=b and m.body_parentid[b]!=a:
                # Panda links 5/7 have built-in neighbours crossing a wrist shell.
                if {ba,bb} not in ({"link5","link7"},{"link5","hand"}):kind="robot_self"
            if not ra and not rb and ba!=bb:
                if self.target in (ba,bb):
                    other=bb if ba==self.target else ba
                    expected_support=(self.phase.startswith(("approach_","descend_","grasp_","lift_")) and other=="table") or (self.phase.startswith(("lower_","release_","withdraw_")) and other==self.current_support)
                    if not expected_support and contact.dist<-.0003:kind="object_obstacle"
            if kind:
                key=(self.phase,kind,*sorted([ba,bb]));row=self.contacts.setdefault(key,{"phase":self.phase,"kind":kind,"bodies":sorted([ba,bb]),"samples":0,"min_distance_m":0.})
                row["samples"]+=1;row["min_distance_m"]=min(row["min_distance_m"],float(contact.dist))
        if self.transport_monitor:
            rot=d.xmat[self.arm.hand_bid].reshape(3,3)
            relative=rot.T@(self.pos(self.target)-self.ee())
            self.max_grasp_slip=max(self.max_grasp_slip,float(np.linalg.norm(relative-self.grasp_relative)))
            if min(self.force_now.values())<.05:self.current_contact_gap+=self.dt
            else:self.current_contact_gap=0.
            self.max_contact_gap=max(self.max_contact_gap,self.current_contact_gap)
        if self.c["grasp_mode"]=="strict" and self.active_welds():self.strict_weld_violation=True
        for item in self.completed:
            if not self.assess_one(item)["ok"]:self.preserved=False
        self.max_actual_speed=max(self.max_actual_speed,float(np.max(np.abs(d.qvel[self.arm_ids]))))
        if d.time+1e-9>=self.next_frame:
            if self.env.render_enabled:
                from PIL import Image,ImageDraw
                image=Image.fromarray(self.env.render_cam("overview"));draw=ImageDraw.Draw(image)
                draw.rectangle((12,12,390,58),fill=(239,245,248))
                draw.text((24,20),f"{self.task.upper()} | {self.c['grasp_mode'].upper()} CONTACT CONTROL",fill=(24,49,60))
                draw.text((24,38),f"{self.phase} / simulation {d.time-.16:05.2f} s",fill=(42,71,83))
                self.env._rec_frames.append(np.asarray(image))
            self.frame_times.append(float(d.time));self.next_frame+=1/30
    def tick(self,q,grip):
        q=np.asarray(q);self.max_command_speed=max(self.max_command_speed,float(np.max(abs(q-self.qprev))/(self.steps*self.dt)))
        self.qprev=q.copy()
        for _ in range(self.steps):
            self.env.step_joint(self.arm,q,grip,sub_steps=1);self.sample()
        self.trace.append({"t":round(float(self.env.d.time),4),"phase":self.phase,"tcp":self.ee().tolist(),"q":self.env.d.qpos[self.arm.arm_qadr].tolist()})
    def wait(self,seconds,grip):
        for _ in range(math.ceil(seconds/(self.dt*self.steps))):self.tick(self.arm.arm_target.copy(),grip)
    def move(self,target,grip,speed=None):
        target=np.asarray(target,float);start=self.ee().copy();speed=speed or self.c["cartesian_speed_m_s"]
        duration=max(.5,1.875*np.linalg.norm(target-start)/speed);count=math.ceil(duration/(self.steps*self.dt))
        for k in range(1,count+1):
            xyz=start+ease(k/count)*(target-start)
            q=self.arm.solve_ik(xyz,self.arm.grasp_quat,local_only=True,w_rot=1.,iters=60)
            # Reference slew limit is physical actuator rate control, not qpos.
            maxdq=self.c["joint_speed_rad_s"]*self.steps*self.dt
            q=self.arm.arm_target+np.clip(q-self.arm.arm_target,-maxdq,maxdq)
            self.tick(q,grip)
        self.wait(self.c["settle_seconds"],grip)
        error=float(np.linalg.norm(target-self.ee()));self.max_tracking=max(error,self.max_tracking)
        if error>.008:raise RuntimeError(f"{self.phase}: TCP residual {error:.4f}m")
        return error
    def move_object(self,n,origin,speed=None):
        origin=np.asarray(origin);self.move(self.ee()+origin-self.pos(n),0.,speed)
        residual=float(np.linalg.norm(origin-self.pos(n)))
        if residual>.004:
            self.move(self.ee()+origin-self.pos(n),0.,speed)
        if np.linalg.norm(origin-self.pos(n))>.009:raise RuntimeError("held object tracking failed")
    def pick(self,n):
        self.target=n;self.mark("approach_"+n)
        gp=self.grip_point(n);safe=self.c["transport_z"]
        self.move([*gp[:2],safe],1.)
        self.mark("descend_"+n);self.move(gp,1.,self.c.get("lower_speed_m_s",.065))
        self.mark("grasp_"+n);before=dict(self.contact_counts);self.wait(.35,0.)
        evidence={finger:self.contact_counts.get((n,finger),0)-before.get((n,finger),0) for finger in ("left_finger","right_finger")}
        if not all(v>0 for v in evidence.values()) or min(self.force_now.values())<.05:raise RuntimeError(f"missing simultaneous two-finger force: {evidence}, {self.force_now}")
        self.events.append({"event":"bilateral_grasp_contact","object":n,"time":float(self.env.d.time),"normal_forces_N":dict(self.force_now)})
        if np.linalg.norm(self.ee()-self.grip_point(n))>.02:raise RuntimeError("grasp geometry gate failed")
        if self.c["grasp_mode"]=="assisted":
            old=self.pos(n).copy();self.arm._set_weld(n,True)
            self.events.append({"event":"contact_gated_weld","object":n,"contacts":evidence,"time":float(self.env.d.time),"activation_pose_jump_m":float(np.linalg.norm(self.pos(n)-old))})
        self.wait(.10,0.)
        self.grasp_relative=self.env.d.xmat[self.arm.hand_bid].reshape(3,3).T@(self.pos(n)-self.ee())
        self.transport_monitor=True;self.current_contact_gap=0.
        self.mark("lift_"+n);self.move([*self.ee()[:2],safe],0.,self.c.get("lift_speed_m_s",.09))
        if self.pos(n)[2]<self.initial[n][2]+.075:raise RuntimeError("object did not lift")
    def place(self,n,support):
        self.mark("carry_"+n);self.current_support=support
        if self.task=="cover":
            destination=np.r_[self.pos(support)[:2],self.config["table_top"]+self.config["placement_origin_z"][n]]
        else:
            destination=self.pos(support)+[0,0,2*self.config["cube_half_size"]+.003]
        high=destination.copy();high[2]=self.pos(n)[2]
        self.move_object(n,high)
        self.mark("lower_"+n);self.move_object(n,destination,self.c.get("lower_speed_m_s",.055))
        self.transport_monitor=False;self.mark("release_"+n)
        self.wait(.16,1.)
        self.arm.release_all();self.wait(.22,1.)
        if not self.released():raise RuntimeError("weld did not release")
        self.mark("withdraw_"+n);self.move([*self.ee()[:2],self.c["transport_z"]],1.,self.c.get("lift_speed_m_s",.09))
        self.target=None;self.wait(.10,1.)
        item=(n,support);check=self.assess_one(item)
        if not check["ok"]:raise RuntimeError(f"placement failed: {check}")
        self.completed.append(item)
    def supporting_contact(self,a,b,surface_z=None,surface_geom=None):
        for contact in self.env.d.contact:
            names={self.body(self.env.m.geom_bodyid[contact.geom1]),self.body(self.env.m.geom_bodyid[contact.geom2])}
            if names!={a,b} or contact.dist>.0005:continue
            geom_names={mujoco.mj_id2name(self.env.m,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom1)),mujoco.mj_id2name(self.env.m,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom2))}
            if surface_geom is not None and surface_geom not in geom_names:continue
            if surface_z is not None and abs(float(contact.pos[2])-surface_z)>.006:continue
            return True
        return False
    def assess_one(self,item):
        n,support=item;p,q=self.pose(n);tilt=math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))))
        if self.task=="cover":
            base,bq=self.pose(support);xy=float(np.linalg.norm(p[:2]-base[:2]))
            expected_z=self.config["table_top"]+self.config["placement_origin_z"][n]
            gap=float(p[2]-expected_z)
            base_error=float(np.linalg.norm(base-self.source_positions[support]))
            base_tilt=math.degrees(math.acos(float(np.clip(1-2*(bq[1]**2+bq[2]**2),-1,1))))
            tolerance=float(self.config["target_xy_tolerance_m"][n])
            inside=xy<tolerance and base_error<.001 and base_tilt<1
            if support=="bowl":inside=inside and p[2]<self.config["table_top"]+.052
            elif support=="pan":inside=inside and p[2]<self.config["table_top"]+.043
            surface_z=self.config["table_top"]+self.config["support_surface_z"][support]
            surface_geom=self.config["support_surface_geom"][support]
            support_ok=self.supporting_contact(n,support,surface_z,surface_geom)
            tilt_limit=35.
        else:
            base,bq=self.pose(support);xy=float(np.linalg.norm(p[:2]-base[:2]));gap=float(p[2]-base[2]-2*self.config["cube_half_size"])
            base_error=float(np.linalg.norm(base-self.source_positions[support]))
            base_tilt=math.degrees(math.acos(float(np.clip(1-2*(bq[1]**2+bq[2]**2),-1,1))))
            inside=xy<.007 and base_error<.004 and base_tilt<3
            support_ok=self.supporting_contact(n,support) and self.supporting_contact(support,"table")
            tilt_limit=8.
        gap_tolerance=.014 if self.task=="cover" else .008
        return {"ok":bool(inside and abs(gap)<gap_tolerance and tilt<tilt_limit and support_ok and n not in self.active_welds()),"xy_error_m":xy,"gap_m":gap,"tilt_deg":tilt,"support_contact":bool(support_ok),"support_upright":bool(base_tilt<1 if self.task=="cover" else base_tilt<3),"base_displacement_m":base_error,"origin":p.tolist()}
    def home(self):
        if not self.released():raise RuntimeError("HOME requires no active weld")
        self.mark("safe_return");self.move([.29,0.,.73],1.)
        self.mark("controlled_home");start=self.arm.arm_target.copy()
        count=max(80,math.ceil(1.875*np.max(abs(HOME_QPOS-start))/(self.c["joint_speed_rad_s"]*self.steps*self.dt)))
        for k in range(1,count+1):self.tick(start+ease(k/count)*(HOME_QPOS-start),1.)
        self.wait(.4,1.)
        self.home_event={"event":"actuated_home_complete","time":float(self.env.d.time),"steps":count,"q_goal":HOME_QPOS.tolist(),"q_actual":self.env.d.qpos[self.arm.arm_qadr].tolist()}
    def assess(self):
        items=self.config["pairs"]
        checks={n:self.assess_one((n,s)) for n,s in items}
        home_error=float(np.linalg.norm(self.env.d.qpos[self.arm.arm_qadr]-HOME_QPOS))
        violations=[v for v in self.contacts.values() if v["min_distance_m"]<-.0003]
        final_home=bool(self.home_event and home_error<.05 and self.released() and np.max(abs(self.env.d.qvel[self.arm_ids]))<.02)
        grip_ok=not self.strict_weld_violation and self.max_grasp_slip<.012 and (self.c["grasp_mode"]!="strict" or self.max_contact_gap<.1)
        hand_clearance={n:float(np.linalg.norm(self.ee()-self.pos(n))) for n,_ in items}
        released_clear=all(v>.08 for v in hand_clearance.values())
        motion_ok=self.max_tracking<.008 and self.max_actual_speed<=max(1.0,1.25*self.c["joint_speed_rad_s"])
        return {"objects":checks,"released":bool(self.released() and released_clear),"hand_clearance_m":hand_clearance,"final_home":final_home,"home_error_rad":home_error,"completed_targets_preserved":self.preserved,"forbidden_contacts":violations,
                "max_grasp_relative_slip_m":self.max_grasp_slip,"max_bilateral_contact_gap_s":self.max_contact_gap,"strict_weld_violation":self.strict_weld_violation,"grasp_verified":grip_ok,"motion_verified":motion_ok,
                "success":bool(all(x["ok"] for x in checks.values()) and final_home and released_clear and self.preserved and self.stability_ok and grip_ok and motion_ok and not violations)}
    def state(self):
        pairs=self.config["pairs"]
        state=set()
        if self.released():state.add("empty()")
        for n,s in pairs:
            if self.held(n):state.add(f"holding({n})")
            elif np.linalg.norm(self.pos(n)-self.initial[n])<.009:state.add(f"at_source({n})")
            if self.assess_one((n,s))["ok"]:state.add(f"at_target({n})")
        if self.assess()["final_home"]:state.add("home_completed()")
        return state
    def run(self):
        sys.path.insert(0,str(ROOT.parent/"CABTO/exp4_bt_tasks/stage3_cabto"))
        from formal_bt import ModelLibrary
        pairs=self.config["pairs"]
        domain=[]
        for i,(n,s) in enumerate(pairs):
            prior=[] if i==0 else [f"at_target({pairs[i-1][0]})"]
            domain += [{"name":"pick","args":{"object":n,"support":s},"pre":["empty()",f"at_source({n})"]+prior,"add":[f"holding({n})"],"del":["empty()",f"at_source({n})"]},
                       {"name":"place","args":{"object":n,"support":s},"pre":[f"holding({n})"],"add":[f"at_target({n})","empty()"],"del":[f"holding({n})"]}]
        target={f"at_target({n})" for n,s in pairs}|{"empty()","home_completed()"}
        domain.append({"name":"home","args":{},"pre":["empty()"]+[f"at_target({n})" for n,s in pairs],"add":["home_completed()"],"del":[]})
        bt=ModelLibrary(domain).build(self.state(),target);self.bt_json=bt.export();self.bt_dot=bt.export("dot")
        self.execution=[]
        for _ in range(len(domain)+1):
            before=self.state();status,model=bt.tick(before)
            if status!="running":break
            if model["name"]=="pick":self.pick(model["args"]["object"])
            elif model["name"]=="place":self.place(model["args"]["object"],model["args"]["support"])
            else:self.home()
            after=self.state();ok=set(model["add"])<=after and not set(model["del"])&after and (set(model["pre"])-set(model["del"]))<=after
            self.execution.append({"model":model,"before":sorted(before),"after":sorted(after),"effect_ok":ok})
            if not ok:raise RuntimeError(f"action effect failed: {model['name']}")
        self.mark("final_stability")
        for _ in range(math.ceil(self.c["stable_seconds"]/(self.dt*self.steps))):
            self.wait(self.dt*self.steps,1.);self.stability_ok=self.stability_ok and target<=self.state()
        result=self.assess();status,_=bt.tick(self.state());result["bt_status"]=status;result["success"]=result["success"] and status=="success"
        result["steps"]=self.execution
        return result
    def close(self):
        if self.env.renderer:self.env.renderer.close()
