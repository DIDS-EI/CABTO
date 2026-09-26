"""Reference-initialized empirical cross-level model refinement from real execution.
LM proposes add/delete patches; no silent automatic patch and no universal claim.
Policies remain byte-identical and must undergo fresh final physical validation.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json
from core.cover_cabto_grounding import LocalModel,save,parse_json,digest
from formal_bt import action_id,validate_model
from run_remaining_cabto import plancheck


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--models",type=Path,required=True);ap.add_argument("--policies",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.models.read_text());cache=json.loads(a.policies.read_text());client=LocalModel("mlx-community/Qwen3.5-9B-4bit")
    models=deepcopy(data["models"]);attempts=[]
    bykey={action_id(m):m for m in models}
    for key,entry in cache["accepted"].items():
        witnesses=[x["evaluation"] for x in cache["attempts"] if x.get("accepted") and x["action_id"]==key]
        witness=witnesses[-1];v=witness["violations"];adds=v["unexpected_added"];deletes=v["unexpected_removed"]
        if not adds and not deletes:continue
        before=set(witness["before"]);after=set(witness["after"]);m=bykey[key]
        prompt=("A robot control program executed successfully but its STRIPS model omitted observed state changes. Propose a MINIMAL model patch. Return ONLY JSON with extra_add and extra_del arrays. "
                "Use exact predicate strings from the evidence. Do not change preconditions or code, do not fabricate objects. A newly true unmodeled fact belongs to extra_add; a lost unmodeled fact belongs to extra_del. "
                "This is a finite observed counterexample, not a guarantee for all physical states.\nAction: "+json.dumps(m,ensure_ascii=False)
                +"\nObserved newly true facts not predicted: "+json.dumps(adds)+"\nObserved removed facts not predicted: "+json.dumps(deletes)
                +"\nCurrent full before: "+json.dumps(sorted(before))+"\nCurrent full after: "+json.dumps(sorted(after)))
        for i in range(3):
            call=client.call(prompt,max_tokens=600,temperature=.1*i);row={"action_id":key,"attempt":i+1,"call":call,"witness":witness}
            try:
                patch=parse_json(call["raw"]);pa=set(patch["extra_add"]);pd=set(patch["extra_del"])
                if not pa<=set(adds) or not pd<=set(deletes):raise ValueError("Unsupported effect patch outside observed evidence")
                candidate={**m,"add":sorted(set(m["add"])|pa),"del":sorted(set(m["del"])|pd)};validate_model(candidate)
                predicted=(before|set(candidate["add"]))-set(candidate["del"])
                if predicted!=after:raise ValueError("Patch does not explain full observed transition")
                row.update(patch=patch,model=candidate,accepted=True);m.clear();m.update(candidate);break
            except Exception as exc:
                row.update(accepted=False,error=str(exc));prompt+="\nRejected response: "+call["raw"]+"\nError: "+str(exc)
            finally:
                attempts.append(row);save(a.output/f"patch{len(attempts):02d}.json",row)
        else:raise RuntimeError("Model refinement budget exhausted")
        print("REFINED",data["task"],key,flush=True)
    check=plancheck(models,data["P"],a.output/"planning")
    if not check["complete_for_declared_P"]:raise RuntimeError("Model patch destroyed completeness")
    refined={**data,"models":models,"check":check,"provenance":{"model":"mlx-community/Qwen3.5-9B-4bit","mode":"reference_initialized_LM_effect_refinement_from_real_execution",
         "original_models":str(a.models),"original_sha256":digest(a.models),"counterexample_policy_file":str(a.policies),"scope":"finite empirical effect refinement, preconditions remain engineering reference"}}
    save(a.output/"models.json",refined)
    transferred=deepcopy(cache)
    for key,e in transferred["accepted"].items():e["original_sampled_model"]=e["model"];e["model"]=deepcopy(bykey[key]);e["fresh_revalidation_required"]=True
    transferred.update(model_source=str(a.output/"models.json"),model_input_sha256=digest(a.output/"models.json"),model_refinement_source=str(a.policies),
                       success=False,status="model_effects_refined_policy_unchanged_revalidation_required")
    save(a.output/"policies_for_revalidation.json",transferred)

if __name__=="__main__":main()
