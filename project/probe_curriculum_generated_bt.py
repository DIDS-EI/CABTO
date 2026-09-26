"""Generate limited controller programs, then execute a formal BT on frozen scenes.

Training uses recorded accepted model programs for prerequisite states, never
object teleportation. Final BT has no scripted step order. Uses oracle+weld.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"CABTO/exp4_bt_tasks/stage3_cabto"))
from core.curriculum_bridge import Runtime,domain,DOCS,effect_report
from formal_bt import ModelLibrary, action_id
from policy_codegen import PolicySampler,PolicyProgram
from llm_backend import make_llm_backend


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(runtime,model,program,path):
    path.mkdir(parents=True,exist_ok=True);(path/"policy.py").write_text(program.source)
    before=runtime.state();valid=set(model["pre"])<=before;exc=None;trace=[];executed=False
    runtime.env.save_png(str(path/"before.png"))
    try:
        if valid:
            api=runtime.api(model);program.validate_api(api);executed=True;trace=program.run(api)["trace"]
    except Exception as e:exc=f"{type(e).__name__}: {e}";trace=getattr(e,"policy_trace",[])
    after=runtime.state();ok,violations=effect_report(model,before,after)
    rec={"model":model,"action_id":action_id(model),"before":sorted(before),"after":sorted(after),
         "valid_trial":valid,"code_executed":executed,"effect_ok":bool(valid and executed and ok and exc is None),
         "goal_ok":set(model["add"])<=after,"exception":exc,"trace":trace,
         "diagnostics":violations,"code_sha256":hashlib.sha256(program.source.encode()).hexdigest(),
         "failure_attribution":"unresolved" if not ok or exc else None,"candidate_model_patch":None}
    calls=[r["method"] for r in trace]
    if model["program_kind"]=="pour" and "tip_pour" not in calls:
        rec["effect_ok"]=False
        rec["diagnostics"]["missing_transition_evidence"]="No tilt primitive observed; accidental spilling is not a pour-policy witness."
    if model["program_kind"]=="pick":
        rec["diagnostics"]["unissued_commands"]=[x for x in ("grasp","lift") if x not in calls]
    if model["program_kind"]=="pour":rec["diagnostics"]["tip_command_issued"]="tip_pour" in calls
    runtime.env.save_png(str(path/"after.png"));(path/"feedback.json").write_text(json.dumps(rec,indent=2))
    return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=ROOT/"outputs/cabto_feasible_curriculum/frozen_config.json")
    ap.add_argument("--attempts",type=int,default=4);ap.add_argument("--episodes",type=int,default=3)
    ap.add_argument("--allow-config-transfer",action="store_true",help="explicit cache transfer to a new config, requires full physical revalidation")
    ap.add_argument("--max-ticks",type=int,default=32)
    ap.add_argument("--goal-decomposition",action="store_true",help="explicit per-object goal scheduler with formal BT subtrees, not monolithic complete planning")
    ap.add_argument("--examples",type=Path,help="prior Cover model-generated source as explicit transfer examples")
    ap.add_argument("--reuse",type=Path);ap.add_argument("--tasks",nargs="+",default=["relay","packing","pour"],choices=["relay","packing","pour"])
    args=ap.parse_args()
    if args.attempts<1 or args.episodes<1:ap.error("attempts and episodes must be positive")
    args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text());hash0=digest(args.config)
    (args.output/"config.json").write_text(json.dumps(config,indent=2))
    source_paths=[Path(__file__),ROOT/"core/curriculum_bridge.py",ROOT/"core/curriculum_goal_bt.py",ROOT/"probe_feasible_curriculum.py",
                  ROOT.parent/"CABTO/exp4_bt_tasks/stage3_cabto/policy_codegen.py"]
    loaded_source_hashes={str(p):digest(p) for p in source_paths}
    for p in source_paths:(args.output/(p.stem+"_source.py")).write_bytes(p.read_bytes())
    result={"implementation_sha256_at_start":loaded_source_hashes,"goal_decomposition":args.goal_decomposition,
        "config_path":str(args.config),"config_sha256":hash0,"config_unchanged":True,
        "symbolic_models":"handwritten_STRIPS","planner":"formal_BT_Expansion_unmodified",
        "program_source":"actual_local_model_or_explicit_cache", "perception":"oracle","grasp":"weld_assisted",
        "object_snap":False,"arm_rehome":False,"samples":[],"episodes":[],"completed":False}
    programs={};sources={};backend=None
    try:
        if args.reuse:
            old=json.loads(args.reuse.read_text());result["programs_reused_from"]=str(args.reuse)
            if old.get("config_sha256")!=hash0:
                if not args.allow_config_transfer:raise ValueError("cache is from a different config; explicitly revalidate transfer separately")
                result["cross_config_transfer"]={"source_config_sha256":old.get("config_sha256"),"requires_physical_revalidation":True}
            for row in old["samples"]:
                if row["success"]:
                    kind=row["kind"];a=next(a for a in reversed(row["attempts"]) if a["accepted"])
                    if a.get("source_type")!="backend_generated" or a.get("sha256")!=hashlib.sha256(a["source"].encode()).hexdigest():
                        raise ValueError("cache missing backend/source hash evidence")
                    if row.get("api_docs")!=DOCS[kind]:raise ValueError("cached API contracts differ")
                    programs[kind]=PolicyProgram(a["source"],DOCS[kind]);sources[kind]=a["source"];result["samples"].append(row)
        # Canonical programs shared across objects/arms/tasks. Only training uses
        # these prerequisite lists; test execution is entirely BT.tick(observe).
        relay=domain("relay");pour=domain("pour")
        training=[("pick","relay",relay[0],[]),
                  ("place","relay",relay[1],[relay[0]]),
                  ("park","relay",relay[2],[relay[0],relay[1]]),
                  ("pour","pour",pour[1],[pour[0]])]
        for kind,task,model,setup in training:
            if kind in programs:continue
            if backend is None:backend=make_llm_backend("local")
            trial=[0]
            def evaluate(program):
                trial[0]+=1;p=args.output/"sampling"/kind/f"trial{trial[0]}"
                runtime=Runtime(config,task,p,0)
                try:
                    for j,sm in enumerate(setup):
                        rec=execute(runtime,sm,programs[sm["program_kind"]],p/f"setup{j}")
                        if not rec["effect_ok"]:
                            return {"valid_trial":False,"code_executed":False,"goal_ok":False,"effect_ok":False,
                                    "diagnostics":{"setup_failed":rec},"candidate_model_patch":None}
                    rec=execute(runtime,model,program,p/"candidate")
                    print("GENERATED",kind,trial[0],rec["effect_ok"],rec["exception"],rec["diagnostics"],flush=True)
                    runtime.save(p);return rec
                finally:runtime.close()
            example=None;info=None
            if args.examples and kind in ("pick","place"):
                old=json.loads(args.examples.read_text());r=next(r for r in old["samples"] if r["kind"]==("pick_lid" if kind=="pick" else "place_lid"))
                example=next(a["source"] for a in reversed(r["attempts"]) if a["accepted"])
                info={"kind":"transfer_example_not_current_failed_code","source_file":str(args.examples),
                      "instruction":"Adapt this actually generated old program to the NEW documented APIs. Do NOT call prepare_empty or rehome: initial trial already has an empty hand. The example is not the complete new action specification. Use all required current action effects."}
            row=PolicySampler(backend,focused_repair=True).synthesize_and_test(model,DOCS[kind],evaluate,args.attempts,args.output/"sampling"/kind,initial_source=example,initial_evidence=info)
            row["kind"]=kind;row["api_docs"]=DOCS[kind];result["samples"].append(row)
            if not row["success"]:result["blocked_on"]=kind;return
            accepted=next(a for a in reversed(row["attempts"]) if a["accepted"])
            sources[kind]=accepted["source"];programs[kind]=PolicyProgram(accepted["source"],DOCS[kind])
        result["program_hashes"]={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
        for task in args.tasks:
            models=domain(task,config)
            for seed in range(args.episodes):
                p=args.output/task/f"seed{seed}";p.mkdir(parents=True,exist_ok=True)
                runtime=Runtime(config,task,p,seed)
                episode={"task":task,"seed":seed,"steps":[],"success":False,"goal":sorted(runtime.goal)}
                result["episodes"].append(episode)
                try:
                    before=runtime.state();episode["initial_state"]=sorted(before)
                    episode["initial_qpos"]=runtime.env.d.qpos.tolist()
                    episode["scene_sha256"]=runtime.scout.result["scene_sha256"]
                    episode["initial_penetrations"]=runtime.scout.result["initial_penetrations"]
                    if runtime.goal<=before or not runtime.initial_task_ready():raise ValueError("invalid trial: task initial conditions not satisfied")
                    if args.goal_decomposition:
                        if task!="relay":raise ValueError("decomposition only implemented for relay")
                        from core.curriculum_goal_bt import GoalDecomposedBT
                        from core.curriculum_bridge import relay_objects
                        bt=GoalDecomposedBT(models,before,runtime.goal,[r["name"] for r in relay_objects(config)])
                    else:bt=ModelLibrary(models).build(before,runtime.goal)
                    episode["kernel"]=bt.kernel
                    (p/"tree.json").write_text(bt.export());(p/"tree.dot").write_text(bt.export("dot"))
                    for step in range(args.max_ticks):
                        status,m=bt.tick(runtime.state())
                        if status!="running":break
                        rec=execute(runtime,m,programs[m["program_kind"]],p/f"step{step}");episode["steps"].append(rec)
                        print("BT",task,seed,step,m["name"],m["args"],rec["effect_ok"],flush=True)
                        if not rec["effect_ok"]:break
                    stable=runtime.stable_goal();status,_=bt.tick(runtime.state())
                    episode.update(success=bool(stable and status=="success" and all(x["effect_ok"] for x in episode["steps"])),
                        stable=stable,stability_seconds=config["evaluation"]["stability_seconds"],final_state=sorted(runtime.state()),
                        final_positions={o:runtime.scout.pos(o).tolist() for o in runtime.env.obj_names},
                        assist_events=runtime.scout.events,pour_events=runtime.pour_events)
                    (p/"tree.json").write_text(bt.export());(p/"tree.dot").write_text(bt.export("dot"))
                    print("BT_FINAL",task,seed,episode["success"],flush=True)
                except Exception as e:episode["exception"]=f"{type(e).__name__}: {e}";print("EPISODE_ERROR",episode["exception"],flush=True)
                finally:runtime.save(p);runtime.close();(p/"result.json").write_text(json.dumps(episode,indent=2))
        result["success_count"]=sum(e["success"] for e in result["episodes"])
        result["completed"]=len(result["episodes"])==len(args.tasks)*args.episodes and all(e["success"] for e in result["episodes"])
    finally:
        result["config_unchanged"]=hash0==digest(args.config)
        result["implementation_sha256_at_end"]={str(p):digest(p) for p in source_paths}
        result["implementation_unchanged_during_run"]=result["implementation_sha256_at_end"]==loaded_source_hashes
        (args.output/"result.json").write_text(json.dumps(result,indent=2))

if __name__=="__main__":main()
