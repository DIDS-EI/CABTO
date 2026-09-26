"""Run the rewritten paper-style Cover or Blocks task, without LLM downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from core.single_arm_rebuilt import RebuiltSingleArm

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task",choices=["cover","blocks"])
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--no-render",action="store_true")
    ap.add_argument("--strict",action="store_true",help="No grasp weld assistance; keep contact-only failures visible")
    a=ap.parse_args()
    default_config=ROOT/("scenes/cover_kitchen_sort_v2.json" if a.task=="cover" else "scenes/blocks_rebuilt_v1.json")
    config=json.loads((a.config or default_config).read_text())
    if a.strict:config["control"]["grasp_mode"]="strict"
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"config.json").write_text(json.dumps(config,indent=2))
    result={"task":a.task,"seed":a.seed,"controller":"rewritten_absolute_IK_quintic_actuator_trajectory",
            "program_source":"handwritten_reference_not_LLM_generated","mode":config["control"]["grasp_mode"],
            "oracle":True,"object_snap":False,"execution_qpos_write":False,"instant_rehome":False,"success":False}
    paths=[Path(__file__),ROOT/"core/single_arm_rebuilt.py",
           ROOT.parent/"CABTO/exp4_bt_tasks/stage2_scripted/exp4_env.py",
           ROOT.parent/"CABTO/exp4_bt_tasks/scenes/panda_body.py",
           ROOT.parent/"CABTO/exp4_bt_tasks/stage3_cabto/formal_bt.py"]
    result["source_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result["config_sha256"]=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    runtime=None;start=time.perf_counter()
    try:
        runtime=RebuiltSingleArm(config,a.seed,render=not a.no_render)
        (a.output/"scene.xml").write_text(runtime.xml)
        if runtime.env.render_enabled:runtime.env.save_png(str(a.output/"before.png"),"overview")
        result.update(runtime.run())
    except Exception as exc:
        result["exception"]=f"{type(exc).__name__}: {exc}"
        if runtime:result.update(runtime.assess());result["success"]=False
    finally:
        if runtime:
            result.update(events=runtime.events,simulation_seconds=float(runtime.env.d.time)-.16,
                          wall_seconds=time.perf_counter()-start,
                          max_command_speed_rad_s=runtime.max_command_speed,max_actual_speed_rad_s=runtime.max_actual_speed,
                          max_segment_endpoint_error_m=runtime.max_tracking,
                          home_event=runtime.home_event,video_fps=30,video_frames=len(runtime.frame_times),
                          all_contact_groups=list(runtime.contacts.values()))
            if hasattr(runtime,"bt_json"):
                (a.output/"tree.json").write_text(runtime.bt_json)
                (a.output/"tree.dot").write_text(runtime.bt_dot)
            result["mode_notes"]="Strict means no grasp weld. Oracle pose and explicit robot gravity compensation remain; not a real-robot or general safety claim."
            result["frame_timestamps"]={"first":runtime.frame_times[0],"last":runtime.frame_times[-1],"sampling":"uniform simulation time, 30fps"}
            (a.output/"trajectory.json").write_text(json.dumps(runtime.trace,separators=(",",":")))
            if runtime.env.render_enabled:
                from PIL import Image
                for name,image in runtime.key_frames.items():Image.fromarray(image).save(a.output/f"phase_{name}.png")
                runtime.env.save_png(str(a.output/"final.png"),"overview")
                runtime.env.save_png(str(a.output/"final_front.png"),"front")
                runtime.env.save_video(str(a.output/"rollout.mp4"),fps=30)
            runtime.close()
        result["sources_unchanged_during_run"]=all(hashlib.sha256(p.read_bytes()).hexdigest()==result["source_sha256"][str(p)] for p in paths)
        (a.output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("events","source_sha256","all_contact_groups")},indent=2),flush=True)
    if not result["success"]:raise SystemExit(1)


if __name__=="__main__":main()
