"""Bind accepted backend-generated Blocks programs to the unchanged formal BT kernel.

True state is observed after every complete policy call. No symbolic transition is
used for execution. Includes controlled missing-holding precondition diagnosis.
Oracle perception and weld assistance remain explicit; no claim of pure contact.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import re

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"CABTO"/"exp4_bt_tasks"/"stage3_cabto"))
import numpy as np
from PIL import Image
from blocks_env import BlocksEnv, HOME_QPOS
from primitives import PrimitiveRunner
from vlm_pointer3d import Pointer
from policy_codegen import PolicyProgram
from formal_bt import ModelLibrary, action_id
from llm_backend import make_llm_backend
import world_state as ws
from probe_blocks_generated_policy import make_api, API_DOCS


def observe(env): return {p for p,v in ws.compute_state(env).items() if v}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--programs",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--episodes",type=int,default=4)
    ap.add_argument("--model-repair",action="store_true")
    ap.add_argument("--contrastive-proposal",action="store_true",help="derive empirical precondition candidates from fitting state contrasts")
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    src=json.loads(args.programs.read_text())
    models=[]; programs={}; sources={}; hashes={}
    for row in src["actions"]:
        accepted=[a for a in row["attempts"] if a["accepted"]]
        if not row["success"] or not accepted: raise ValueError("Both generated actions must be accepted before BT binding")
        m=row["action_model"]; key=action_id(m); source=accepted[-1]["source"]
        models.append(m); programs[key]=PolicyProgram(source,row.get("api_docs",API_DOCS));sources[key]=source
        hashes[key]=hashlib.sha256(source.encode()).hexdigest()
    if len(models)!=2: raise ValueError("This probe expects the two grounded Blocks actions")
    library=ModelLibrary(models)
    goal={"stacked_on(green_block,yellow_block)"}
    env=BlocksEnv(render=True,img_size=512)
    base={k:v.copy() for k,v in env._init_obj_qpos.items()}
    pointer=Pointer(None,env,backend="oracle")
    histories=[]; tasks=[]; records=[]
    result={"mode":"oracle_perception_weld_assisted","policy_source":"cached_real_model_output",
            "source_result":str(args.programs),"source_hashes":hashes,"episodes":histories,
            "scene_variation":"initial xy jitter +/-5mm, seed0 unmodified", "symbolic_models":models,
            "instant_rehome":False,"final_home_required":True,
            "note":"finite tests; no contact-only or general soundness claim"}

    def controlled_home(reason="final_after_stack",steps=80):
        if env.is_holding():raise RuntimeError("controlled HOME requires released block")
        start=np.array(env.arm_target,float);event={"event":"controlled_actuator_home",
            "reason":reason,"time":float(env.d.time),"steps":steps,
            "q_start":start.tolist(),"q_goal":HOME_QPOS.tolist(),"goal_preserved":True}
        for k in range(1,steps+1):
            t=k/steps;alpha=t*t*(3.0-2.0*t)
            env.arm_target=start+(HOME_QPOS-start)*alpha
            env.hold_arm(grip=1.0,sub_steps=8)
            event["goal_preserved"]=event["goal_preserved"] and goal<=observe(env)
            if k%2==0:env.record_frame("overview")
        env.arm_target=HOME_QPOS.copy()
        for _ in range(12):
            env.hold_arm(grip=1.0,sub_steps=8)
            event["goal_preserved"]=event["goal_preserved"] and goal<=observe(env)
            env.record_frame("overview")
        q=np.array([env.d.qpos[a] for a in env.arm_qadr]);event["q_final"]=q.tolist()
        event["residual_rad"]=float(np.linalg.norm(q-HOME_QPOS))
        if event["residual_rad"]>.10:raise RuntimeError(f"controlled HOME residual {event['residual_rad']:.4f}rad")
        if not event["goal_preserved"]:raise RuntimeError("stack goal was lost during controlled HOME")
        return event

    def reset(seed):
        rng=np.random.default_rng(seed)
        for name,q in base.items():
            new=q.copy()
            if seed: new[:2]+=rng.uniform(-.005,.005,size=2)
            env._init_obj_qpos[name]=new
        env.reset();env.start_record();env.record_frame("overview")
        runner=PrimitiveRunner(env,recorder=lambda:env.record_frame("overview"))
        return runner,make_api(env,runner,pointer)

    def execute(m,api,label,role="validation"):
        key=action_id(m); before=observe(env); trace=[];exc=None
        bp=args.output/(label+"_before.png");Image.fromarray(env.render_cam("overview")).save(bp)
        local_api=dict(api)
        if m.get("api_binding",{}).get("target_pose")=="dst":
            local_api["target_pose"]=lambda:api["locate"](m["args"]["dst"])
        try: trace=programs[key].run(local_api)["trace"]
        except Exception as error:
            exc=f"{type(error).__name__}: {error}";trace=getattr(error,"policy_trace",[])
        after=observe(env)
        violations={"missing_add":sorted(set(m["add"])-after),"remaining_del":sorted(set(m["del"])&after),
                    "missing_preserved_pre":sorted((set(m["pre"])-set(m["del"]))-after)}
        apath=args.output/(label+"_after.png");Image.fromarray(env.render_cam("overview")).save(apath)
        record={"before":sorted(before),"after":sorted(after),"action_id":key,"code_sha256":hashes[key],
                "valid_trial":set(m["pre"])<=before,"exception":exc,"code_executed":True,
                "role":role,"trace":trace,"violations":violations,
                "effect_ok":exc is None and not any(violations.values()),
                "before_image":bp.name,"after_image":apath.name,"policy_source":sources[key]}
        return record

    try:
        for seed in range(args.episodes):
            runner,api=reset(seed)
            start=observe(env);tasks.append({"start":sorted(start),"goal":sorted(goal)})
            tree=library.build(start,goal)
            if seed==0:
                (args.output/"blocks_bt.json").write_text(tree.export(),encoding="utf-8")
                (args.output/"blocks_bt.dot").write_text(tree.export("dot"),encoding="utf-8")
                result["kernel"]=tree.kernel
            episode={"seed":seed,"steps":[],"success":False,
                     "settled_initial_qpos":env.d.qpos.tolist(),
                     "settled_object_poses":{n:env.get_object_pose(n)[0].tolist() for n in base}};histories.append(episode)
            for step in range(8):
                current=observe(env);status,m=tree.tick(current)
                if status=="success": break
                if status!="running":
                    episode["failure"]="BT returned failure on observed state";break
                rec=execute(m,api,f"seed{seed}_step{step}")
                episode["steps"].append(rec);records.append(rec)
                if not rec["valid_trial"] or not rec["effect_ok"]: break
            # Complete the physical task after the generated place program:
            # object released -> retreat already executed -> actuator-only HOME.
            home_event=controlled_home()
            # The released stack must stay stable while and after the arm returns.
            stable=True
            for _ in range(32):
                env.hold_arm(grip=1.0,sub_steps=16);env.record_frame("overview")
                stable = stable and goal<=observe(env)
            status,_=tree.tick(observe(env))
            q=np.array([env.d.qpos[a] for a in env.arm_qadr])
            final_home=bool(np.linalg.norm(q-HOME_QPOS)<.10 and env._grip_cmd>=.5 and not env.is_holding())
            episode.update(success=bool(goal<=observe(env) and stable and final_home and status=="success"
                           and all(r["valid_trial"] and r["effect_ok"] for r in episode["steps"])),
                           final_state=sorted(observe(env)),stable_seconds=1.024,stable=stable,bt_status=status,
                           final_home=final_home,home_event=home_event,final_arm_qpos=q.tolist())
            Image.fromarray(env.render_cam("overview")).save(args.output/f"seed{seed}_final_home.png")
            env.save_video(str(args.output/f"seed{seed}_bt_rollout.mp4"),fps=30)
            print("FORMAL_BT",seed,episode["success"],len(episode["steps"]),flush=True)

        if args.model_repair:
            # Controlled fault: remove the payload-holding precondition from model,
            # never alter the actual code. This is not an autonomous open-world discovery.
            place=next(m for m in models if m["name"]=="place_on");key=action_id(place)
            bad=deepcopy(place);bad["pre"]=[p for p in bad["pre"] if p!="in_gripper(green_block)"]
            faulty=ModelLibrary([bad if m["name"]=="place_on" else m for m in models])
            _,api=reset(0);start=observe(env);bad_tree=faulty.build(start,goal)
            status,selected=bad_tree.tick(start)
            if status!="running" or action_id(selected)!=key:
                raise RuntimeError("Controlled faulty model did not cause BT to select place; do not manufacture a counterexample")
            fault=execute(selected,api,"missing_pre_counterexample",role="counterexample")
            positives=[r for r in records if r["action_id"]==key]
            if len(positives)<2: raise RuntimeError("Need distinct fitting and held-out validation episodes")
            fitting=[fault,positives[0]]
            evidence=[fault]+positives[1:]  # heldout transitions never shown to proposer
            backend=make_llm_backend("local")
            prompt={"action_model":bad,"observed_transitions":[{k:r[k] for k in
                    ("before","after","exception","violations","code_sha256")} for r in fitting],
                    "permitted_precondition_atoms":sorted(set().union(*(set(r["before"]) for r in fitting))),
                    "request":"The same unchanged policy failed in one state and succeeded in another. "
                    "Propose a conservative missing precondition ONLY if supported. Output JSON "
                    "{\"add_pre\":[\"atom\"]}. Do not change add/del; do not claim proof from samples."}
            if args.contrastive_proposal:
                # Computed exclusively from fitting data, not known correct preconditions.
                contrasts=sorted(set(fitting[1]["before"])-set(fitting[0]["before"])-set(bad["pre"])-set(bad["add"]))
                prompt={"action":bad["name"],"current_pre":bad["pre"],"immutable_effects":{"add":bad["add"],"del":bad["del"]},
                        "permitted_precondition_atoms":contrasts,
                        "state_contrast":[{"atom":p,"before_success":True,"before_failure":False} for p in contrasts],
                        "request":"Select one candidate from permitted_precondition_atoms to require before executing this action. "
                        "The same unchanged code worked in the success state but not the failure state. "
                        "Only select an atom from this list; do not select a goal/effect. "
                        "Output JSON {\"add_pre\":[\"selected exact atom\"]}. This is a hypothesis, not a proof."}
            attempts=[]
            messages=[{"role":"system","content":"You repair a STRIPS action contract using evidence. Return JSON only."},
                      {"role":"user","content":json.dumps(prompt,ensure_ascii=False)}]
            for attempt in range(3):
                raw=backend.chat(messages,max_tokens=256,temperature=.2*attempt)
                rec={"raw":raw,"messages":deepcopy(messages)}
                try:
                    cleaned=re.sub(r"```(?:json)?|```","",raw).strip()
                    obj=json.loads(cleaned); additions=obj["add_pre"]
                    if not isinstance(additions,list) or any(not isinstance(x,str) for x in additions): raise ValueError("invalid add_pre")
                    if not set(additions)<=set(prompt["permitted_precondition_atoms"]): raise ValueError("unknown atoms")
                    candidate=deepcopy(bad);candidate["pre"]=sorted(set(candidate["pre"])|set(additions))
                    rec["candidate"]=candidate
                    rec["gate"]=faulty.propose_patch(key,candidate,evidence,tasks)
                except Exception as exc: rec["gate"]={"accepted":False,"reason":str(exc)}
                attempts.append(rec)
                if "candidate" in rec: break  # freeze first valid candidate; never adapt to heldout feedback
                messages += [{"role":"assistant","content":raw},{"role":"user","content":json.dumps(rec["gate"])+"\nTry a supported precondition, JSON only."}]
            result["model_repair"]={"fault_injection":"removed in_gripper(green_block) from place_on pre",
                "bad_tree_first_action":action_id(selected) if selected else None,"counterexample":fault,
                "candidate_source":backend.name,"attempts":attempts,"version":faulty.version,
                "validation_scope":"finite_same_code_tests_plus_symbolic_task_regression",
                "heldout_definition":"remaining episode records excluded from model prompt"}
            if faulty.version:
                runner,api=reset(0);tree=faulty.build(observe(env),goal); trace=[]
                for step in range(8):
                    status,m=tree.tick(observe(env))
                    if status!="running":break
                    rec=execute(m,api,f"repaired_bt_step{step}");trace.append(rec)
                    if not rec["effect_ok"]:break
                stable=True
                for _ in range(32):
                    env.hold_arm(grip=1.0,sub_steps=16);env.record_frame("overview")
                    stable=stable and goal<=observe(env)
                status,_=tree.tick(observe(env))
                result["model_repair"]["physical_replay"]={"steps":trace,"goal_ok":goal<=observe(env),
                    "stable_seconds":1.024,"stable":stable,"bt_status":status,
                    "success":bool(stable and status=="success" and all(r["effect_ok"] and r["valid_trial"] for r in trace))}
                env.save_video(str(args.output/"repaired_bt_rollout.mp4"),fps=30)
            result["repair_closed_loop_success"] = bool(faulty.version and result["model_repair"]["physical_replay"]["success"])
        result["success_count"]=sum(e["success"] for e in histories)
    finally:
        (args.output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        env.renderer.close()

if __name__=="__main__": main()
