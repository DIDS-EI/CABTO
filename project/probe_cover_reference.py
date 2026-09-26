"""Reference-only Cover calibration after IK fixes; no LLM or generated-policy claim.

Runs existing three-pair reference without object snapping. Legacy grasp weld and
arm rehome remain explicitly assisted. Rechecks ALL lids at the final state and
for one further second instead of trusting per-step cached success.
"""
import argparse
import json
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parent
STAGE2=ROOT.parent/"CABTO"/"exp4_bt_tasks"/"stage2_scripted"
sys.path.insert(0,str(STAGE2))
import numpy as np
from exp4_env import Exp4Env, HOME_QPOS
from arm_skills import ArmSkills
import scene_cover as sc
from probe_five_scene_preflight import candidate_xml


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--zero-place-bias",action="store_true",help="paired test: zero calibrated feedforward only during tracked descent")
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # Legacy module creates an output dir at import using argv[1]. Keep it local.
    saved=sys.argv[:];sys.argv=["run_cover",str(args.output)]
    try: from run_cover import cover_one
    finally: sys.argv=saved
    env=Exp4Env(candidate_xml(sc.build_scene_xml(),"cover"),[("",0.0)],
                [f"lid_{k}" for k,*_ in sc.PAIRS],render=True,img_size=560)
    env.reset();env.start_record();env.record_frame("overview")
    arm=env.arms[0];arm.grasp_z_tol=.10
    sk=ArmSkills(env,arm,recorder=lambda:env.record_frame("overview"),rec_every=2,block_half=sc.LID_HH)
    original=sk.descend_place_tracked
    def tracked_configured(*a,**kw):
        kw["suppress_feedforward"]=args.zero_place_bias
        return original(*a,**kw)
    sk.descend_place_tracked=tracked_configured
    home_events=[]
    def controlled_home(settle=12,grip=1.0,reason="between_pairs",steps=80):
        if any(arm.is_holding(name) for name in env.obj_names):
            raise RuntimeError("controlled HOME requires empty hand")
        # place_release disables future assisted attachments. Re-enable only the
        # declared grasp assistance before a new pick; never write object/robot pose.
        if reason.startswith("before_"):arm._grasp_enabled=True
        monitor_goal=reason.startswith("final_")
        start=arm.arm_target.copy();event={"event":"controlled_actuator_home",
            "reason":reason,"time":float(env.d.time),"steps":steps,
            "q_start":start.tolist(),"q_goal":HOME_QPOS.tolist(),
            "goal_monitor_active":monitor_goal,"goal_preserved":True if monitor_goal else None}
        home_events.append(event)
        for k in range(1,steps+1):
            t=k/steps;alpha=t*t*(3.0-2.0*t)
            env.step_joint(arm,start+(HOME_QPOS-start)*alpha,grip=grip,sub_steps=8)
            if monitor_goal:event["goal_preserved"]=event["goal_preserved"] and all(v["tight_geometry"] for v in assess().values())
            if k%2==0:env.record_frame("overview")
        for _ in range(settle):
            env.step_joint(arm,HOME_QPOS,grip=grip,sub_steps=8)
            if monitor_goal:event["goal_preserved"]=event["goal_preserved"] and all(v["tight_geometry"] for v in assess().values())
            env.record_frame("overview")
        q=np.array([env.d.qpos[a] for a in arm.arm_qadr]);err=float(np.linalg.norm(q-HOME_QPOS));event["residual_rad"]=err
        if err>.10:raise RuntimeError(f"controlled HOME residual {err:.4f}rad")
        if monitor_goal and not event["goal_preserved"]:raise RuntimeError("cover goal was lost during controlled HOME")
        return err
    # cover_one calls pick_secure(..., rehome_first=True). Replace the legacy
    # qpos-writing rehome with actuator interpolation without changing its path.
    sk.rehome=lambda settle=12,grip=1.0:controlled_home(settle,grip,"before_next_lid")
    result={"task":"cover_three_pairs_smooth_complete","source":"handwritten_reference", "mode":"assisted",
            "zero_place_bias":args.zero_place_bias,"instant_rehome":False,
            "assistance":["grasp_weld","controlled_actuator_home","GT object tracking"],
            "object_snap":False,"per_pair":[],"all_goals_final":False}
    def assess():
        out={}
        for k,xy,*_ in sc.PAIRS:
            p,q=env.get_object_pose("lid_"+k)
            off=float(np.linalg.norm(p[:2]-xy));gap=float(p[2]-sc.LID_HH-sc.BASE_TOP)
            # quaternion wxyz: world z component of local lid normal
            normal_z=1-2*(q[1]*q[1]+q[2]*q[2]); tilt=float(math.acos(np.clip(normal_z,-1,1)))
            out[k]={"xy_error_m":off,"z_gap_m":gap,"tilt_deg":math.degrees(tilt),
                    "legacy_criterion":off<.04 and -.02<gap<.05,
                    "tight_geometry":off<.012 and abs(gap)<.010 and tilt<math.radians(10)}
        return out
    try:
        env.save_png(str(args.output/"before.png"))
        for k,xy,*_ in sc.PAIRS:
            ok,off,gap,p=cover_one(env,sk,arm,"lid_"+k,np.asarray(xy),sc.BASE_TOP)
            result["per_pair"].append({"pair":k,"legacy_step_ok":bool(ok),"xy_error_m":off,"z_gap_m":gap})
        result["pre_home_geometry"]=assess()
        home_error=controlled_home(reason="final_after_all_covers")
        stable={k:True for k,*_ in sc.PAIRS}
        for _ in range(32):
            env.hold_arms({"":1.0},sub_steps=16);env.record_frame("overview")
            for k,v in assess().items(): stable[k]=stable[k] and v["tight_geometry"]
        q=np.array([env.d.qpos[a] for a in arm.arm_qadr])
        final_home=bool(np.linalg.norm(q-HOME_QPOS)<.10 and arm._grip_cmd>=.5
                        and not any(arm.is_holding(n) for n in env.obj_names))
        result["final_geometry"]=assess();result["stable_seconds"]=1.024
        result["stable"]=stable;result["final_home"]=final_home
        result["home_error_rad"]=home_error;result["home_events"]=home_events
        result["final_arm_qpos"]=q.tolist()
        result["all_goals_final"]=all(v["legacy_criterion"] for v in result["final_geometry"].values())
        result["tight_stable_success"]=bool(all(stable.values()) and final_home)
        env.save_png(str(args.output/"after.png"));env.save_video(str(args.output/"rollout.mp4"),fps=30)
        print("COVER_REFERENCE",result["all_goals_final"],result["tight_stable_success"],result["final_geometry"],flush=True)
    finally:
        (args.output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        env.renderer.close()

if __name__=="__main__":main()
