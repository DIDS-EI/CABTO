"""Actual local-model policy synthesis on the two-action Blocks smoke task.

Uses ground-truth perception, weld-assisted grasps and fixed STRIPS models.
NOT a strict-contact experiment, NOT all five tasks or a paper-number reproduction.
No candidate sees env/data/GT checker. Both success and failure artifacts are saved.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from functools import wraps

ROOT=Path(__file__).resolve().parent
STAGE3=ROOT.parent/"CABTO"/"exp4_bt_tasks"/"stage3_cabto"
sys.path.insert(0,str(STAGE3))
import numpy as np
from PIL import Image
from blocks_env import BlocksEnv
from primitives import PrimitiveRunner, grasp_z_for_top
from vlm_pointer3d import Pointer, obj_phrase
from planner import _instantiate
from llm_backend import make_llm_backend
from policy_codegen import PolicySampler, PolicyProgram
from feedback import build_failure_feedback
import world_state as ws

API_DOCS={
 "locate":"locate(obj_name)->tuple(x,y,z). z is object's TOP surface in world metres. Oracle perception for this diagnostic.",
 "grasp_height":"grasp_height(top_z)->float. Converts object TOP height to correct TCP grasp height.",
 "approach":"approach(xyz)->None. Moves open gripper above that object, not down to grasp height.",
 "align_xy":"align_xy(xyz)->None. Refines alignment above object with gripper OPEN; use before descending/grasping.",
 "descend_to":"descend_to(z, lock_xy)->None. Lowers OPEN gripper to grasp height keeping horizontal lock_xy (x,y).",
 "grasp":"grasp(xy)->None. Closes gripper at its current height, following (x,y). Does NOT approach, descend or lift.",
 "record_grab_offset":"record_grab_offset(obj_name)->None. After closing, calibrates held object offset required for placement. Diagnostic uses GT object pose.",
 "lift":"lift(xy=None)->None. Raises closed gripper holding object.",
 "move_above":"move_above(destination_top_xyz)->tuple(tcp_x,tcp_y). Carries held object over destination using calibrated grab offset.",
 "refine_above":"refine_above(destination_xy)->tuple(tcp_x,tcp_y). Optional placement alignment refinement; use its returned TCP coordinates for descend_place.",
 "descend_place":"descend_place(destination_top_z, tcp_xy)->float. Lowers held object to destination top; do not pass grasp_height or object-center height here.",
 "place_release":"place_release()->None. Opens gripper at current location without moving; call only at intended placement location.",
 "retreat":"retreat()->None. Retracts OPEN gripper after release.",
 "settle":"settle()->None. Holds gripper OPEN while scene settles; never call while intending to hold object."
}


def make_api(env, runner, pointer, *, target=None):
    """Trusted adapter; oracle and weld provenance must be declared by callers."""
    def plain(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,(tuple,list)): return tuple(plain(x) for x in value)
        if isinstance(value,np.generic): return value.item()
        return value
    def wrapped(fn):
        @wraps(fn)
        def call(*a,**kw): return plain(fn(*a,**kw))
        return call
    def offset(name): runner.record_grab_offset(name)
    api={"locate":lambda obj_name:plain(pointer.locate_3d(obj_phrase(obj_name),obj_name=obj_name)),
         "grasp_height":grasp_z_for_top,"record_grab_offset":offset}
    for name in API_DOCS:
        if name not in api: api[name]=wrapped(getattr(runner,name))
    if target is not None:
        api["target_pose"] = lambda:plain(pointer.locate_3d(obj_phrase(target),obj_name=target))
    return api


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=ROOT/"outputs"/"cabto_reproduction_audit"/"generated_blocks")
    ap.add_argument("--attempts",type=int,default=3)
    ap.add_argument("--backend",choices=("local","cloud"),default="local")
    ap.add_argument("--focused-contract",action="store_true",help="action-specific APIs and explicit tuple signatures")
    ap.add_argument("--focused-repair",action="store_true",help="compact last-error repair context instead of full diagnostic tree")
    ap.add_argument("--resume-result",type=Path,help="reuse accepted generated pick and seed failed place source from an actual prior result")
    ap.add_argument("--fresh-place",action="store_true",help="reuse accepted pick but sample place afresh, without previous failed code")
    ap.add_argument("--bound-target",action="store_true",help="bind read-only target_pose to action dst; no generated object-name lookup")
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    previous=json.loads(args.resume_result.read_text()) if args.resume_result else None
    started=time.monotonic()
    backend=make_llm_backend(args.backend)
    env=BlocksEnv(render=True,img_size=512)
    pointer=Pointer(None,env,backend="oracle")
    accepted_pick=None
    results={"backend":backend.name,"perception":"oracle","control":"weld_assisted",
             "symbolic_models":"fixed_STRIPS","auto_model_repair":False,
             "runtime":"finite_API_interpreter_no_exec","goals":["stacked_on(green_block,yellow_block)"],"actions":[]}
    try:
        for skill,argmap in (("pick",{"obj":"green_block"}),("place_on",{"obj":"green_block","dst":"yellow_block"})):
            action=_instantiate(skill,argmap)
            if args.focused_repair:
                action["description"] = (
                    "Grasp green_block from the table and finish holding it above the table."
                    if skill=="pick" else
                    "The gripper ALREADY HOLDS green_block. Place this held green_block ON TOP OF yellow_block. "
                    "yellow_block is the stationary DESTINATION support; green_block is the PAYLOAD, not the destination. "
                    "The final gripper must be empty and the green block must rest on the yellow block.")
            old_action=next((a for a in previous["actions"] if a["name"]==skill),None) if previous else None
            if skill=="pick" and old_action and old_action["success"]:
                accepted=next(a for a in reversed(old_action["attempts"]) if a["accepted"])
                accepted_pick=PolicyProgram(accepted["source"],API_DOCS)
                results["actions"].append({**old_action,"source_type":"cached_model_program",
                    "reused_from":str(args.resume_result),"revalidation":"replayed physically before every place candidate"})
                continue
            docs=dict(API_DOCS)
            if args.focused_contract:
                names=(["locate","grasp_height","approach","align_xy","descend_to","grasp","record_grab_offset","lift"]
                       if skill=="pick" else ["locate","move_above","refine_above","descend_place","place_release","retreat","settle"])
                docs={n:API_DOCS[n] for n in names}
                for n in ("approach","align_xy","move_above"):
                    if n in docs: docs[n]+=" Pass ONE tuple variable p: api."+n+"(p). Never split p into separate arguments."
                if "descend_to" in docs: docs["descend_to"]+=" Exactly TWO arguments: float height and tuple xy; e.g. api.descend_to(z, p[:2])."
                if "grasp" in docs: docs["grasp"]+=" Exactly ONE argument: tuple xy; e.g. api.grasp(p[:2])."
                if "lift" in docs: docs["lift"]+=" Use api.lift(xy=p[:2]); keep holding at the end to satisfy the action ADD effects."
            if args.focused_repair:
                # Typed API declarations describe individual functions, not an action sequence.
                docs = ({
                    "locate":"api.locate(obj_name: str) -> tuple[float,float,float] : object TOP coordinates. Save result p; use p[:2] for xy and p[2] for top_z.",
                    "move_above":"api.move_above(xyz: tuple[float,float,float]) -> tuple[float,float] : move held object above xyz. Save return value tcp_xy. Takes ONE tuple, not xyz[0].",
                    "refine_above":"api.refine_above(xy: tuple[float,float]) -> tuple[float,float] : optional alignment, returning TCP xy. Takes ONE 2D tuple, not a scalar.",
                    "descend_place":"api.descend_place(dst_top_z: float, tcp_xy: tuple[float,float]) -> float : lower held object. BOTH arguments required. tcp_xy is the RETURN VALUE from api.move_above or api.refine_above. It is not the target raw xy.",
                    "place_release":"api.place_release() -> None : opens gripper at current pose; does not move object. No arguments.",
                    "retreat":"api.retreat() -> None : move empty gripper away after release. No arguments.",
                    "settle":"api.settle() -> None : wait with gripper open. No arguments."
                } if skill=="place_on" else {k:"api."+v for k,v in docs.items()})
            if args.bound_target and skill=="place_on":
                action["description"]="Place the already-held object onto the support designated by the action's destination argument. Finish with the object supported and the gripper empty."
                docs.pop("locate",None)
                docs["target_pose"]="api.target_pose() -> tuple[float,float,float] : read-only TOP pose of the destination support. No arguments. Returns (x,y,z); indexing [2] is top_z and [:2] is xy."
                action["api_binding"]={"target_pose":"dst"}
            action_dir=args.output/skill; action_dir.mkdir(exist_ok=True)
            counter=0
            def evaluate(program):
                nonlocal counter
                counter+=1
                path=action_dir/f"attempt_{counter}"; path.mkdir(exist_ok=True)
                (path/"policy.py").write_text(program.source,encoding="utf-8")
                env.reset(); env.start_record(); env.record_frame("overview")
                runner=PrimitiveRunner(env,recorder=lambda:env.record_frame("overview"))
                api=make_api(env,runner,pointer,target=argmap.get("dst") if args.bound_target else None)
                if accepted_pick is not None:
                    accepted_pick.run(api)
                before=ws.compute_state(env)
                valid=all(before.get(p) is True for p in action["pre"])
                bp=path/"before.png"; Image.fromarray(env.render_cam("overview")).save(bp)
                trace=[]; exception=None; executed=False
                try:
                    if valid:
                        program.validate_api(api)
                        executed=True; trace=program.run(api)["trace"]
                except Exception as exc:
                    trace=getattr(exc,"policy_trace",[])
                    exception={"type":type(exc).__name__,"message":str(exc)}
                after=ws.compute_state(env)
                ok,violations=ws.check_effect(env,action["add"],action["del"],before)
                if not valid: violations.append("trial preconditions not established; cannot attribute candidate failure")
                apath=path/"after.png"; Image.fromarray(env.render_cam("overview")).save(apath)
                env.save_video(str(path/"rollout.mp4"),fps=30)
                feedback=build_failure_feedback(action,before,after,{"calls":trace},exception,violations,
                      {"before":str(bp),"after":str(apath)},program.source,{"source":"oracle","weld_assisted":True})
                feedback["policy_feedback"]=("This candidate was generated by the backend and interpreted as finite API calls. "
                    "Repair its code using the recorded exception, call trace and unmet effects. Keep h unchanged. "
                    "Do not infer an impossible action from this failed attempt.")
                feedback["consumer"]="PolicySampler next backend.chat if rejected and budget remains"
                lookups=[call["args"][0] for call in trace if call["method"]=="locate" and call["args"]]
                lookups += [argmap["dst"] for call in trace if call["method"]=="target_pose" and "dst" in argmap]
                diagnostics={"located_objects":lookups}
                if skill=="place_on":
                    diagnostics.update(expected_destination=argmap["dst"],payload=argmap["obj"],
                        destination_was_localized=argmap["dst"] in lookups,
                        payload_position=env.get_object_pose(argmap["obj"])[0].tolist(),
                        destination_position=env.get_object_pose(argmap["dst"])[0].tolist())
                    if lookups and argmap["dst"] not in lookups:
                        diagnostics["target_mismatch"]=("The executed code localized the payload instead of the destination. "
                            "The requested support is " + argmap["dst"] + "; actual locate calls: " + str(lookups))
                verdict={"valid_trial":valid,"code_executed":executed,"effect_ok":bool(ok and exception is None),
                         "goal_ok":all(after.get(p) is True for p in action["add"]),
                         "trace":trace,"feedback":feedback,"diagnostics":diagnostics,"execution_mode":"weld_assisted",
                         "whole_task_goal_ok":bool(after.get("stacked_on(green_block,yellow_block)",False))}
                (path/"feedback.json").write_text(json.dumps(verdict,indent=2,ensure_ascii=False),encoding="utf-8")
                print("CANDIDATE",skill,counter,"valid",valid,"effects",verdict["effect_ok"],"whole_goal",verdict["whole_task_goal_ok"],flush=True)
                return verdict
            failed=old_action["attempts"][-1] if old_action and not old_action["success"] and not args.fresh_place else None
            initial_evidence=(failed["evaluation"].get("feedback",{}).get("observed_facts",{}).get("exception")
                              or failed["evaluation"].get("exception")) if failed else None
            result=PolicySampler(backend,focused_repair=args.focused_repair).synthesize_and_test(
                action,docs,evaluate,args.attempts,action_dir,
                initial_source=failed.get("source") if failed else None,initial_evidence=initial_evidence)
            results["actions"].append({"name":skill,"api_docs":docs,**result})
            if not result["success"]: break
            if skill=="pick": accepted_pick=PolicyProgram(result["attempts"][-1]["source"],API_DOCS)
        final=ws.compute_state(env)
        results["success"]=bool(len(results["actions"])==2 and all(r["success"] for r in results["actions"])
                                and final.get("stacked_on(green_block,yellow_block)",False))
        results["elapsed_seconds"]=time.monotonic()-started
        results["final_state"]=final
        (args.output/"result.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
        print("FINAL_GENERATED_BLOCKS",results["success"],"mode=weld_assisted,perception=oracle",flush=True)
    finally:
        if env.renderer is not None: env.renderer.close()

if __name__=="__main__": main()
