"""Run a new physical handle-free tea-box task; never reuse old success records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"CABTO/exp4_bt_tasks/stage3_cabto"))
from core.handleless_tea_handover import HandlelessTeaRuntime, box_domain
from formal_bt import ModelLibrary
from cover_bridge import effect_report


def run(config, out, seed=0, render=True):
    out=Path(out)
    if (out/"result.json").exists():raise ValueError("Use a fresh output directory to preserve previous evidence")
    source_paths=[Path(__file__),ROOT/"core/handleless_tea_handover.py",ROOT/"core/smooth_handover_bridge.py",
                  ROOT.parent/"CABTO/exp4_bt_tasks/stage2_scripted/exp4_env.py",ROOT.parent/"CABTO/exp4_bt_tasks/stage2_scripted/arm_skills.py"]
    source_paths += [ROOT/config["visual_style"]["texture_file"],ROOT/"assets/tea_carton/make_texture.py"]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    r=HandlelessTeaRuntime(config,out,seed,render)
    result={"task":"handleless_tea_handover","seed":seed,"mode":"direct_box_face_contact_gated_weld_assisted",
            "program_source":"handwritten_reference_actions_formal_BT_not_new_LLM_generation",
            "handles_exist":False,"success":False,"steps":[],"source_hashes":hashes,
            "config_sha256":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
            "scene_sha256":hashlib.sha256((out/"scene.xml").read_bytes()).hexdigest()}
    try:
        bt=ModelLibrary(box_domain()).build(r.state(),r.goal)
        (out/"tree.json").write_text(bt.export())
        for k in range(10):
            status,m=bt.tick(r.state())
            if status!="running":break
            before=r.state()
            if not set(m["pre"])<=before:raise RuntimeError("Unsatisfied physical preconditions")
            next(iter(r.api(m).values()))()
            after=r.state();ok,diag=effect_report(m,before,after)
            ok=bool(ok and r.collision_free())
            result["steps"].append({"action":m["name"],"before":sorted(before),"after":sorted(after),"effect_ok":ok,"diagnostics":diag})
            if render:
                r.env.save_png(str(out/f"step{k:02d}.png"),"overview")
                if m["program_kind"]=="receive":r.env.save_png(str(out/"handover_closeup.png"),"front")
            print(m["name"],ok,flush=True)
            if not ok:break
        stable=r.stable_goal();status,_=bt.tick(r.state())
        result.update(success=bool(stable and status=="success" and all(s["effect_ok"] for s in result["steps"])),stable=stable,bt_status=status)
    except Exception as error:
        result["exception"]=f"{type(error).__name__}: {error}"
    finally:
        result.update(placement=r.placement_geometry(),final_state=sorted(r.state()),handover_verified=r.handover_verified(),
                      video_fps=30,video_sampling="uniform_simulation_time",video_frames=len(r.frame_times),
                      robot_gravity_compensation=config["handover"].get("robot_gravity_compensation",True),
                      max_joint_speed_rad_s=r.max_actual_speed,max_segment_error_m=r.peak_tracking,
                      events=r.events,collision_free=r.collision_free(),forbidden_groups=r.audit.summary(),
                      initial_position=r.initial_pos.tolist(),final_position=r.pos().tolist(),sim_seconds=float(r.env.d.time),
                      source_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()))
        r.save(out);r.close()
        (out/"result.json").write_text(json.dumps(result,indent=2))
    print("FINAL",result["success"],result.get("exception"),flush=True)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,default=ROOT/"scenes/handover_tea_box_v3.json")
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seed",type=int,default=0);ap.add_argument("--no-render",action="store_true")
    a=ap.parse_args();r=run(json.loads(a.config.read_text()),a.output,a.seed,not a.no_render)
    raise SystemExit(0 if r["success"] else 1)
