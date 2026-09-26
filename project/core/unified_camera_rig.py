"""Camera-only MJCF augmentation shared by all five Panda scenes.
Wrist cameras have identical hand-local mounting transforms; no target tracking,
geom/physics edits, fake wrist images, or control target changes. Global camera
uses a table-frame workspace specification, not current object ground truth.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib,inspect,math,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

PROFILE_PATH=Path(__file__).resolve().parents[1]/"scenes/camera_rig_unified_v3.json"
PROFILE=json.loads(PROFILE_PATH.read_text())


def axes(eye,target,up=(0.,1.,0.)):
    eye=np.array(eye,float);target=np.array(target,float);up=np.array(up,float)
    z=eye-target;z/=np.linalg.norm(z);x=np.cross(up,z)
    if np.linalg.norm(x)<1e-9:raise ValueError("look-at up is parallel to viewing axis")
    x/=np.linalg.norm(x);y=np.cross(z,x)
    return np.column_stack((x,y,z))

def numbers(a):return " ".join(f"{v:.12g}" for v in np.asarray(a).ravel())

def table_frame(config):
    if "table_top" in config:return np.array(config["table_center"][:2],float),float(config["table_top"])
    return np.array(config["table"]["center_xy"],float),float(config["table"]["top_z"])

def augment_xml(xml,config,profile=None):
    p=deepcopy(profile or PROFILE);root=ET.fromstring(xml);world=root.find("worldbody")
    if world is None:raise ValueError("Missing worldbody")
    if any(c.get("name","").startswith("rig_") for c in root.iter("camera")):raise ValueError("Camera rig already installed")
    hands=[b for b in root.iter("body") if b.get("name") in ("hand","hand_L","hand_R")]
    if not hands:raise ValueError("No Panda hand bodies")
    mounts=[];wp=p["wrist"];R=axes(wp["position_hand_m"],wp["look_at_hand_m"])
    for body in hands:
        side={"hand":"single","hand_L":"left","hand_R":"right"}[body.get("name")];name="rig_wrist_"+side
        ET.SubElement(body,"camera",name=name,mode="fixed",pos=numbers(wp["position_hand_m"]),xyaxes=numbers(np.r_[R[:,0],R[:,1]]),fovy=str(wp["fovy_deg"]))
        mounts.append({"name":name,"parent_body":body.get("name"),"T_hand_camera":{"position":wp["position_hand_m"],"rotation":R.tolist()},"fovy_deg":wp["fovy_deg"]})
    center,h=table_frame(config);xy=center+np.array(p["global"]["offset_table_xy_m"]);gp=[*xy,h+p["global"]["height_above_table_m"]]
    ET.SubElement(world,"camera",name="rig_global",mode="fixed",pos=numbers(gp),quat="1 0 0 0",fovy=str(p["global"]["fovy_deg"]))
    names=[m["name"] for m in mounts]+["rig_global"]
    return ET.tostring(root,encoding="unicode"),{"profile":p,"cameras":names,"wrist_mounts":mounts,
        "global_position_world":gp,"table_center_xy":center.tolist(),"table_top_z":h,
        "old_cameras_preserved":[c.get("name") for c in ET.fromstring(xml).iter("camera")]}


def physics_comparison(reference,actual):
    names=("body_pos","body_quat","body_mass","body_inertia","body_gravcomp","body_parentid","jnt_pos","jnt_axis","jnt_range","qpos0",
           "geom_size","geom_pos","geom_quat","geom_friction","geom_contype","geom_conaffinity","geom_solref","geom_solimp","geom_bodyid",
           "actuator_ctrlrange","actuator_gainprm","actuator_biasprm","dof_damping","dof_frictionloss","eq_data","eq_active0")
    for key in ("nq","nv","nbody","ngeom","nu","neq"):
        if getattr(reference,key)!=getattr(actual,key):raise AssertionError(key)
    for key in names:np.testing.assert_array_equal(getattr(reference,key),getattr(actual,key),err_msg=key)
    np.testing.assert_array_equal(reference.opt.gravity,actual.opt.gravity)
    assert reference.opt.timestep==actual.opt.timestep
    return {"matched":True,"arrays":list(names),"old_ncam":reference.ncam,"new_ncam":actual.ncam,"only_sensor_configuration_changed":True}


@contextmanager
def install_for_construction(config,profile=None):
    """Scoped constructor adapter; restore even on failure, never modify source env.
    Execute one scene constructor in this context, no concurrent scene creation.
    """
    from exp4_env import Exp4Env
    original=Exp4Env.__init__;signature=inspect.signature(original)
    def construct(self,*args,**kwargs):
        bound=signature.bind(self,*args,**kwargs);bound.apply_defaults();xml=bound.arguments["xml"]
        augmented,meta=augment_xml(xml,config,profile)
        bound.arguments["xml"]=augmented
        p=profile or PROFILE;bound.arguments["img_size"]=p["width"]
        if p["width"]!=p["height"]:raise ValueError("Current render backend requires square images")
        ref=mujoco.MjModel.from_xml_string(xml)
        original(*bound.args,**bound.kwargs)
        meta["physics_validation"]=physics_comparison(ref,self.m)
        meta["source_xml_sha256"]=hashlib.sha256(xml.encode()).hexdigest();meta["rig_xml_sha256"]=hashlib.sha256(augmented.encode()).hexdigest()
        self.camera_rig_metadata=meta;self.camera_rig_xml=augmented;self.camera_rig_original_xml=xml
    Exp4Env.__init__=construct
    try:yield
    finally:Exp4Env.__init__=original


def calibration(env,name):
    m,d=env.m,env.d;i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,name)
    if i<0:raise ValueError(name)
    size=int(env.img_size);f=size/(2*math.tan(math.radians(float(m.cam_fovy[i]))/2));R=d.cam_xmat[i].reshape(3,3).copy();pos=d.cam_xpos[i].copy()
    parent=int(m.cam_bodyid[i]);bodyR=d.xmat[parent].reshape(3,3);bodypos=d.xpos[parent]
    return {"name":name,"time":float(d.time),"image_size":[size,size],"K":[[f,0,size/2],[0,f,size/2],[0,0,1]],
            "T_world_camera":{"position":pos.tolist(),"rotation":R.tolist()},"optical_axis_world":(-R[:,2]).tolist(),
            "parent_body":mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,parent),
            "measured_T_body_camera":{"position":(bodyR.T@(pos-bodypos)).tolist(),"rotation":(bodyR.T@R).tolist()},
            "fovy_deg":float(m.cam_fovy[i]),"frame_convention":PROFILE["optical_frame"]}


def project(cal,point):
    T=cal["T_world_camera"];R=np.array(T["rotation"]);v=R.T@(np.array(point)-T["position"])
    if -v[2]<=0:return None
    K=np.array(cal["K"]);return np.array([K[0,2]+K[0,0]*v[0]/(-v[2]),K[1,2]-K[1,1]*v[1]/(-v[2])])


def ray(cal,uv):
    K=np.array(cal["K"]);R=np.array(cal["T_world_camera"]["rotation"])
    v=R@np.array([(uv[0]-K[0,2])/K[0,0],-(uv[1]-K[1,2])/K[1,1],-1.]);v/=np.linalg.norm(v)
    return np.array(cal["T_world_camera"]["position"]),v


def intersect_plane(cal,uv,z):
    origin,v=ray(cal,uv)
    if abs(v[2])<1e-8:raise ValueError("Ray parallel to plane")
    t=(z-origin[2])/v[2]
    if t<=0:raise ValueError("Plane behind camera")
    return origin+t*v
