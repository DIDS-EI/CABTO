"""Cover: actual model-generated two program schemas reused across three pairs,
then scheduled by unchanged BT Expansion over all six grounded actions.
Oracle perception, weld grasps and explicit arm-rehome assistance are reported.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"CABTO"/"exp4_bt_tasks"/"stage3_cabto"))
from cover_bridge import CoverRuntime,models,effect_report,PICK_DOCS,PLACE_DOCS,CANONICAL_PICK_DOCS,CANONICAL_PLACE_DOCS
from policy_codegen import PolicySampler,PolicyProgram
from llm_backend import make_llm_backend
from formal_bt import ModelLibrary,action_id
from exp4_env import HOME_QPOS
from PIL import Image
import numpy as np

SMOOTH_PICK_DOCS={"pick_lid_complete":"api.pick_lid_complete(): zero arguments. It performs controlled HOME, approach, alignment, descent, grasp, lift, and offset measurement in the required order. Call exactly once."}
SMOOTH_PLACE_DOCS={"place_lid_complete":"api.place_lid_complete(): zero arguments. It moves above the matching base, aligns, lowers, releases, retreats, and settles in the required order. Call exactly once."}


class SmoothCoverRuntime(CoverRuntime):
    """Cover runtime with actuator-only smooth HOME motion.

    The inherited prepare_empty used ArmSkills.rehome(), which writes robot qpos
    instantaneously between pairs.  Here every in-episode reset is replaced by
    an 80-step smoothstep interpolation through the position actuators.  Episode
    initialization still uses the normal environment reset.
    """
    def controlled_home(self, reason="between_pairs", steps=80):
        if any(self.arm.is_holding(name) for name in self.env.obj_names):
            raise RuntimeError("controlled HOME requires an empty hand")
        # place_release disables future assisted attachments. Re-enable only the
        # already-declared grasp assistance for the next pick; this is not a pose write.
        if reason.startswith("before_"):
            self.arm._grasp_enabled=True
        start=self.arm.arm_target.copy()
        monitor_goal=reason.startswith("final_")
        event={"event":"controlled_actuator_home","reason":reason,
               "time":float(self.env.d.time),"steps":steps,
               "q_start":start.tolist(),"q_goal":HOME_QPOS.tolist(),
               "goal_monitor_active":monitor_goal,"goal_preserved":True if monitor_goal else None}
        self.events.append(event)
        for k in range(1,steps+1):
            t=k/steps;alpha=t*t*(3.0-2.0*t)
            q=start+(HOME_QPOS-start)*alpha
            self.env.step_joint(self.arm,q,grip=1.0,sub_steps=8)
            if monitor_goal:event["goal_preserved"]=event["goal_preserved"] and self.goal<=self.state()
            if k%2==0:self.env.record_frame("overview")
        for _ in range(12):
            self.env.step_joint(self.arm,HOME_QPOS,grip=1.0,sub_steps=8)
            if monitor_goal:event["goal_preserved"]=event["goal_preserved"] and self.goal<=self.state()
            self.env.record_frame("overview")
        actual=np.array([self.env.d.qpos[a] for a in self.arm.arm_qadr])
        error=float(np.linalg.norm(actual-HOME_QPOS))
        event["residual_rad"]=error
        if error>.10:raise RuntimeError(f"controlled HOME residual {error:.4f}rad")
        if monitor_goal and not event["goal_preserved"]:
            raise RuntimeError("cover goal was lost during controlled HOME")
        return error

    def api(self,model):
        base=super().api(model)
        if model["name"]=="pick_lid":
            base["prepare_empty"]=lambda:self.controlled_home("before_"+model["args"]["lid"])
            def pick_lid_complete():
                base["prepare_empty"]();pose=base["target_pose"]();base["approach"](pose)
                base["align_xy"](pose);base["descend_to"](pose[2],pose[:2]);base["grasp"](pose[:2])
                base["lift"]();base["record_grab_offset"]()
            return {"pick_lid_complete":pick_lid_complete}
        def place_lid_complete():
            pose=base["target_pose"]();base["move_above"](pose);base["align_target"](pose)
            base["lower"](pose);base["place_release"]();base["retreat"]();base["settle"]()
        return {"place_lid_complete":place_lid_complete}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--attempts",type=int,default=4);ap.add_argument("--episodes",type=int,default=3)
    ap.add_argument("--reuse",type=Path,help="reuse recorded model programs; no fake generation")
    ap.add_argument("--transfer-example",type=Path,help="actual successful Blocks result used as an in-context code example")
    ap.add_argument("--canonical-api",action="store_true",help="use consistent grasp/align_xy/descend_to names across tasks")
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    runtime=SmoothCoverRuntime();domain=models();programs={};sources={};sample_rows=[]
    result={"task":"cover_three_pairs_smooth_complete","perception":"oracle","control":"weld_assisted",
            "arm_rehome":"controlled_actuator_interpolation","instant_rehome":False,"object_snap":False,"model_source":"local_qwen_vl",
            "symbolic_models":"fixed_STRIPS_six_grounded_actions","samples":sample_rows,"episodes":[]}
    def execute(m,program,path):
        path.mkdir(parents=True,exist_ok=True)
        (path/"policy.py").write_text(program.source,encoding="utf-8")
        before=runtime.state();valid=set(m["pre"])<=before;trace=[];exc=None;executed=False
        Image.fromarray(runtime.env.render_cam()).save(path/"before.png")
        try:
            if valid:
                program.validate_api(runtime.api(m));executed=True
                trace=program.run(runtime.api(m))["trace"]
        except Exception as e:exc=f"{type(e).__name__}: {e}";trace=getattr(e,"policy_trace",[])
        after=runtime.state();ok,violations=effect_report(m,before,after)
        Image.fromarray(runtime.env.render_cam()).save(path/"after.png")
        calls=[r["method"] for r in trace]
        diagnosis=dict(violations)
        required_order=(["pick_lid_complete"] if m["name"]=="pick_lid" else ["place_lid_complete"])
        missing_calls=[name for name in required_order if name not in calls]
        order_ok=not missing_calls and all(calls.index(required_order[i])<calls.index(required_order[i+1])
                                           for i in range(len(required_order)-1))
        diagnosis["required_transition_order"]=required_order
        diagnosis["missing_transition_calls"]=missing_calls
        diagnosis["transition_order_ok"]=order_ok
        if m["name"]=="pick_lid":
            diagnosis["closed_command_issued"]=bool(set(calls)&{"close","grasp","pick_lid_complete"})
            diagnosis["lift_command_issued"]="lift" in calls or "pick_lid_complete" in calls
            diagnosis["gripper_width_m"]=runtime.arm.get_gripper_width()
            diagnosis["observed_missing_commands"]=(["gripper_close"] if not diagnosis["closed_command_issued"] else [])+(["lift"] if not diagnosis["lift_command_issued"] else [])
            if not diagnosis["closed_command_issued"]:
                diagnosis["message"]="No gripper CLOSE command was executed. Approaching/descending/measurement cannot grasp the lid. The hand is still empty."
        record={"valid_trial":valid,"code_executed":executed,"effect_ok":bool(valid and ok and order_ok and exc is None),
                "goal_ok":set(m["add"])<=after,"before":sorted(before),"after":sorted(after),
                "trace":trace,"exception":exc,"diagnostics":diagnosis,"geometry":runtime.geometry(),
                "model":m,"code_sha256":hashlib.sha256(program.source.encode()).hexdigest()}
        (path/"feedback.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
        return record
    try:
        if args.reuse:
            old=json.loads(args.reuse.read_text());result["programs_reused_from"]=str(args.reuse)
            for row in old["samples"]:
                accepted=next(r for r in reversed(row["attempts"]) if r["accepted"])
                source=accepted["source"];kind=row["kind"]
                sources[kind]=source;programs[kind]=PolicyProgram(source,row.get("api_docs",PICK_DOCS if kind=="pick_lid" else PLACE_DOCS))
                sample_rows.append(row)
        else:
            backend=make_llm_backend("local")
            # The generated code chooses the symbolic action composition; the
            # physically calibrated transition itself is exposed atomically to
            # prevent omitted release/retreat steps and repeated state changes.
            doc_pairs=(("pick_lid",SMOOTH_PICK_DOCS),("place_lid",SMOOTH_PLACE_DOCS))
            for i,(kind,docs) in enumerate(doc_pairs):
                model=deepcopy(domain[i]);model["description"]=(
                    "Write one call to api.pick_lid_complete(). It performs the calibrated pick transition and finishes holding the lid."
                    if kind=="pick_lid" else
                    "Write one call to api.place_lid_complete(). It performs calibrated lowering, release, retreat, and settle; finish empty and withdrawn.")
                count=[0]
                def evaluate(program):
                    count[0]+=1;runtime.reset()
                    if kind=="place_lid":programs["pick_lid"].run(runtime.api(domain[0]))
                    path=args.output/"sampling"/kind/f"attempt_{count[0]}"
                    rec=execute(model,program,path)
                    runtime.env.save_video(str(path/"rollout.mp4"),fps=30)
                    print("SAMPLE",kind,count[0],rec["effect_ok"],rec["exception"],rec["diagnostics"],flush=True)
                    return rec
                transfer_source=None;transfer_info=None
                if args.transfer_example:
                    example=json.loads(args.transfer_example.read_text())
                    previous=next(a for a in example["actions"] if a["name"]==("pick" if kind=="pick_lid" else "place_on"))
                    transfer_source=next(a["source"] for a in reversed(previous["attempts"]) if a["accepted"])
                    transfer_info={"kind":"in_context_transfer_example_not_current_failed_code",
                        "source_result":str(args.transfer_example),
                        "instruction":"This program previously succeeded in a DIFFERENT Blocks scene. "
                        "Adapt its approach to this Cover action using ONLY the new API reference. "
                        "Old API names/arguments are not available. Do not omit the actual closing/lifting for pick "
                        "or the actual lowering/releasing for place. payload_pose already provides calibrated grasp z."}
                row=PolicySampler(backend,focused_repair=True).synthesize_and_test(model,docs,evaluate,args.attempts,args.output/"sampling"/kind,
                    initial_source=transfer_source,initial_evidence=transfer_info)
                row["kind"]=kind;row["api_docs"]=docs;sample_rows.append(row)
                if not row["success"]:
                    result["blocked_on"]=kind;return
                accepted=next(r for r in reversed(row["attempts"]) if r["accepted"])
                sources[kind]=accepted["source"];programs[kind]=PolicyProgram(accepted["source"],docs)
        result["program_source_hashes"]={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
        library=ModelLibrary(domain)
        for seed in range(args.episodes):
            runtime.reset(seed);path=args.output/f"bt_seed{seed}";path.mkdir(exist_ok=True)
            bt=library.build(runtime.state(),runtime.goal)
            (path/"tree.json").write_text(bt.export(),encoding="utf-8");(path/"tree.dot").write_text(bt.export("dot"))
            episode={"seed":seed,"steps":[],"initial_qpos":runtime.env.d.qpos.tolist(),"kernel":bt.kernel}
            result["episodes"].append(episode)
            for step in range(16):
                status,m=bt.tick(runtime.state())
                if status!="running":break
                rec=execute(m,programs[m["name"]],path/f"step{step}")
                episode["steps"].append(rec)
                print("BT",seed,step,action_id(m),rec["effect_ok"],flush=True)
                if not rec["valid_trial"] or not rec["effect_ok"]:break
            # Complete the task physically: the last place program has released
            # and retreated, then the empty arm returns to HOME through actuators.
            pre_home_geometry=runtime.geometry()
            home_error=runtime.controlled_home("final_after_all_covers")
            stable=runtime.stable_goal();status,_=bt.tick(runtime.state())
            q=np.array([runtime.env.d.qpos[a] for a in runtime.arm.arm_qadr])
            final_home=bool(np.linalg.norm(q-HOME_QPOS)<.10 and runtime.arm._grip_cmd>=.5
                            and not any(runtime.arm.is_holding(n) for n in runtime.env.obj_names))
            episode.update(success=bool(stable and final_home and status=="success" and all(r["valid_trial"] and r["effect_ok"] for r in episode["steps"])),
                    stable=stable,stable_seconds=1.024,bt_status=status,final_geometry=runtime.geometry(),
                    pre_home_geometry=pre_home_geometry,final_state=sorted(runtime.state()),
                    final_home=final_home,home_error_rad=home_error,final_arm_qpos=q.tolist(),
                    assistance_events=runtime.events)
            runtime.env.save_png(str(path/"final.png"));runtime.env.save_png(str(path/"final_front.png"),"front")
            runtime.env.save_video(str(path/"rollout.mp4"),fps=30)
            print("COVER_FINAL",seed,episode["success"],episode["final_geometry"],flush=True)
        result["success_count"]=sum(e["success"] for e in result["episodes"])
    finally:
        (args.output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        runtime.close()

if __name__=="__main__":main()
