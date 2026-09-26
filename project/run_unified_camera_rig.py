"""Run immutable accepted policies with additional diagnostic wrist/global cameras.
Never uses new sensor images or evaluation ground truth as motion commands.
Records simultaneous images and time-varying calibration at action/approach frames.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from copy import deepcopy
import mujoco
import numpy as np
from PIL import Image
from core.unified_camera_rig import PROFILE,PROFILE_PATH,install_for_construction,calibration,project,ray,intersect_plane
from run_remaining_cabto import make,config_path,execute as execute_remaining,docs,plancheck
from core.cover_cabto_grounding import save,digest
from core.cover_cabto_api import FineCover
from formal_bt import ModelLibrary,action_id
from run_cover_cabto_method import execute as execute_cover

ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT.parent;OUTPUT=PACKAGE/"outputs/unified_cameras"


def load_inputs(task):
    if task=="cover":
        base=PACKAGE/"outputs/cover_cabto_fix"
        return base/"proposal_final_v2/models.json",base/"sampling_9b_complete/policies.json",ROOT/"scenes/cover_kitchen_sort_v2.json"
    base=PACKAGE/"outputs/remaining_cabto"/task
    if task=="blocks":return base/"proposal/models.json",base/"sampling/policies.json",config_path(task)
    return base/"refinement/models.json",base/"refinement/policies_for_revalidation.json",config_path(task)


def evaluation_targets(task,r,m):
    """GT targets ONLY for evaluation/visibility; never sent to any model/controller."""
    env=r.env if task!="cover" else r.r.env;out=[]
    def add(name,body,point,semantic):out.append({"name":name,"body":body,"xyz":np.asarray(point,float).tolist(),"semantic":semantic})
    if task=="storage":
        for i in (0,1):
            body=f"item{i}";p=r.scout.pos(body);rot=env.d.xmat[env.obj_bid[body]].reshape(3,3)
            add(body,body,p+rot@[0,0,r.p["item_half_size"][2]],"payload collision top center; decorative studs may lie above")
        p,q=r.tray_pose();R=env.d.xmat[env.obj_bid["tray"]].reshape(3,3)
        add("box_floor","tray",p+R@[0,0,r.p["tray_wall"]],"interior floor center")
        for i in (0,1):add(f"slot{i}","tray",p+R@np.r_[r.p["slots_xy"][i],r.p["tray_wall"]],"unmarked task slot; requires box pose and declared local offset")
        for i in (0,1):add(f"handle{i}","tray",r.scout.handle_world(i),"declared side-handle target")
        add("shelf","shelf",np.r_[r.p["shelf_xy"],r.p["shelf_top_z"]],"shelf support center")
    elif task=="pour":
        R=env.d.xmat[env.obj_bid["basin"]].reshape(3,3)
        add("basin_floor","basin",r.scout.basin_center()+R@[0,0,r.b["basin_floor"]],"basin floor center; evaluation-only true surface height")
        for name in ("canL","basin"):add(name,name,r.scout.pos(name),"body origin at handle; not object silhouette center")
    elif task=="handover":
        add("box_center","baton",r.pos(),"box body center; may be behind visible front face")
        for i in (0,1):add(f"grasp{i}","baton",r.handle_world(i),"designated box-face grasp zone; not silhouette center")
    else:
        base=r.r
        for name,support in base.config["pairs"]:
            add(name,name,base.grip_point(name),"grasp TCP prior, not visible surface center")
            add(support,support,base.pos(support),"support body origin, not top surface")
    return out


def capture(r,task,directory,index,event,m=None):
    env=r.r.env if task=="cover" else r.env
    directory.mkdir(parents=True,exist_ok=True)
    before_q=env.d.qpos.copy();before_v=env.d.qvel.copy();before_time=float(env.d.time)
    cams=["overview",*env.camera_rig_metadata["cameras"]];views=[]
    targets=evaluation_targets(task,r,m)
    for name in cams:
        cal=calibration(env,name);image=env.render_cam(name).copy();file=directory/f"{name}.png";Image.fromarray(image).save(file)
        errors=[];visibility=[]
        for target in targets:
            uv=project(cal,target["xyz"]);in_frame=uv is not None and bool(np.all(uv>=0) and np.all(uv<env.img_size))
            seen=False;distance=None;hitbody=None
            if in_frame:
                back=intersect_plane(cal,uv,target["xyz"][2]);errors.append(float(np.linalg.norm(back-target["xyz"])))
                origin,direction=ray(cal,uv);gid=np.array([-1],dtype=np.int32)
                distance=float(mujoco.mj_ray(env.m,env.d,origin,direction,None,1,-1,gid))
                if gid[0]>=0:
                    hitbody=mujoco.mj_id2name(env.m,mujoco.mjtObj.mjOBJ_BODY,int(env.m.geom_bodyid[gid[0]]))
                    seen=hitbody==target["body"]
            visibility.append({"target":target["name"],"pixel":None if uv is None else uv.tolist(),"in_frame":in_frame,
                  "ray_hits_target_body":seen,"first_hit_body":hitbody,"ray_distance_m":distance,
                  "warning":"GT geometric diagnostic; not a perception output, ray body hit is not precise grasp-point visibility"})
        views.append({"camera":name,"image":str(file),"image_sha256":digest(file),"calibration":cal,
                      "max_project_plane_roundtrip_m":max(errors,default=0),"evaluation_only_visibility":visibility})
    np.testing.assert_array_equal(before_q,env.d.qpos);np.testing.assert_array_equal(before_v,env.d.qvel);assert before_time==float(env.d.time)
    row={"index":index,"event":event,"action":m,"simulation_time":before_time,"images_simultaneous_no_step":True,
         "render_did_not_change_state":True,"camera_control_use":"diagnostic_only","views":views,"evaluation_only_targets":targets}
    save(directory/"capture.json",row);return row


def run(task,out,seed,initial_only=False,profile=None):
    mp,pp,cp=load_inputs(task);modeldata=json.loads(mp.read_text());policydata=json.loads(pp.read_text());config=json.loads(cp.read_text());out.mkdir(parents=True,exist_ok=True)
    model_by={action_id(m):m for m in modeldata["models"]};policies=policydata["accepted"]
    for key,v in policies.items():assert hashlib.sha256(v["source"].encode()).hexdigest()==v["sha256"]
    # Snapshot only files actually relied on; report builder edits never change this snapshot.
    sources=list((ROOT/"core").glob("*.py"))+[Path(__file__),ROOT/"run_remaining_cabto.py",ROOT/"run_cover_cabto_method.py",cp,PROFILE_PATH]
    sources+=list((PACKAGE/"CABTO/exp4_bt_tasks/stage2_scripted").glob("*.py"))+list((PACKAGE/"CABTO/exp4_bt_tasks/stage3_cabto").glob("*.py"))+list(ROOT.glob("probe_*.py"))
    source_hashes={str(p):digest(p) for p in sources};snap=out/"source_snapshot";snap.mkdir(exist_ok=True)
    for p in sources:
        target=snap/p.relative_to(PACKAGE);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    with install_for_construction(config,profile):
        if task=="cover":r=FineCover(config,seed,True,"oracle",out/"oracle_perception")
        else:r=make(task,out/"runtime",seed)
    env=r.r.env if task=="cover" else r.env
    (out/"scene_with_cameras.xml").write_text(env.camera_rig_xml)
    (out/"scene_reference.xml").write_text(env.camera_rig_original_xml)
    save(out/"rig.json",env.camera_rig_metadata)
    captures=[];result={"task":task,"seed":seed,"success":False,"steps":[],"source_hashes":source_hashes,
        "model_input":str(mp),"model_input_sha256":digest(mp),"policies_input":str(pp),"policies_input_sha256":digest(pp),
        "config_sha256":digest(cp),"camera_rig":env.camera_rig_metadata,"perception_used_for_motion":"oracle_only_diagnostic_run",
        "assistance":"strict_no_weld" if task in ("cover","blocks") else "declared_weld_assisted",
        "new_cameras_used_for_control":False}
    def take(event,m=None):
        i=len(captures);folder=out/"frames"/f"{i:03d}_{event}";captures.append(capture(r,task,folder,i,event,m))
    take("initial")
    if initial_only:
        result.update(success=None,mode="initial_camera_probe",captures=captures);save(out/"result.json",result);r.close();return result
    raw_api=r.bind if task=="cover" else r.api
    captured_per_action=set()
    # Preserve signatures so the existing whitelist/signature validation still applies.
    import functools
    def instrument_signature(m):
        api=raw_api(m);new={}
        for name,fn in api.items():
            @functools.wraps(fn)
            def wrapper(*a,_name=name,_fn=fn,**kw):
                answer=_fn(*a,**kw)
                groups={"approach":"approach","align_xy":"approach","move_both_above":"approach","move_above":"above_target","carry":"above_target","move_source":"above_target","align_receiver":"receiver_aligned"}
                group=groups.get(_name)
                if group and group not in captured_per_action:captured_per_action.add(group);take("after_"+_name,m)
                return answer
            new[name]=wrapper
        return new
    if task=="cover":r.bind=instrument_signature
    else:r.api=instrument_signature;r.require_full_frame=True
    goal=set(modeldata["P"][-1]["goal"] if task in ("cover","blocks") else modeldata["goal"])
    if task!="cover":r.goal=goal
    tree=ModelLibrary(list(model_by.values())).build(r.state(),goal)
    try:
        for step in range(40):
            status,m=tree.tick(r.state())
            if status!="running":break
            take(f"step{step:02d}_before",m);captured_per_action.clear();source=policies[action_id(m)]["source"]
            if task=="cover":rec=execute_cover(r,m,source)
            else:rec=execute_remaining(r,task,m,source,out/f"step{step:02d}")
            save(out/f"step{step:02d}.json",rec);result["steps"].append(rec)
            print("CAMERA_RUN",task,step,m["name"],rec["effect_ok"],rec.get("exception"),flush=True)
            if not rec["effect_ok"]:raise RuntimeError("Frozen control action failed")
        if task=="cover":
            r.r.mark("final_stability");r.r.wait(config["control"]["stable_seconds"],1.)
            result.update(r.final_check(goal));result["success"]=result["success"] and goal<=r.state()
        else:result["success"]=bool(goal<=r.state() and r.stable_goal())
        result["bt_status"]=tree.tick(r.state())[0];result["success"]=result["success"] and result["bt_status"]=="success"
    except Exception as exc:result.update(success=False,error=f"{type(exc).__name__}: {exc}")
    finally:
        take("final");result["captures"]=captures;result["final_state"]=sorted(r.state());result["sources_unchanged"]=all(digest(p)==h for p,h in source_hashes.items())
        save(out/"result.json",result)
        env.save_png(str(out/"final.png"),"overview")
        env.save_video(str(out/"rollout.mp4"),fps=30)
        r.close()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tasks",nargs="+",default=["storage","pour","handover","cover","blocks"]);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=0);ap.add_argument("--initial-only",action="store_true");a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):ap.error("Use new output to preserve camera trials")
    for task in a.tasks:
        result=run(task,a.output/task,a.seed,a.initial_only);print("CAMERA_RESULT",task,result["success"],result.get("error"),flush=True)

if __name__=="__main__":main()
