"""Reasoning-enabled Qwen proposal over a fixed predicate vocabulary and signatures."""
import argparse,json,re
from core.cover_generation_fix import ThinkingModel
from core.cover_cabto_grounding import PAIRS,TASKS,CONFIG,digest,save,check_models
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--model",default="mlx-community/Qwen3.5-9B-4bit")
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);client=ThinkingModel(args.model)
    prompt="""Reason briefly, then give your final answer in a single JSON code fence.
Construct STRIPS models for three robot action signatures: pick, place, home.
Use ONLY these EXACT proposition strings: empty(), at_source(OBJECT), holding(OBJECT), at_target(OBJECT), home_completed().
empty() = gripper OPEN and holds no object. at_source(OBJECT) = object remains at original table location. holding(OBJECT) = gripper holds object in air. at_target(OBJECT) = object is released at its destination support. home_completed() = empty arm remains at HOME.
pick grasps and lifts from table. place transports the ALREADY held payload to its matching support, lowers, releases, retreats. home moves the empty open arm to HOME and changes no object states.
All tasks start with empty() and each object at_source. Each task places a chosen subset (one/two/three objects) and ends with empty() and home_completed(). HOME must not require all objects to have been placed.
Important STRIPS semantics: a proposition persists unless the delete list explicitly removes it. An object cannot be both at_source and holding, or both holding and at_target. An empty hand cannot simultaneously hold anything. Movement away from HOME invalidates home_completed(). Pick/place are reusable for any of the three object/support pairs; do NOT add ordering constraints.
Output JSON {"schemas":[{"name":...,"pre":[...],"add":[...],"del":[...]}, ...]} with EXACTLY three schemas named pick/place/home. There is no args field. Use OBJECT literally; code will mechanically substitute actual object names. For home, no OBJECT propositions. pre/add disjoint; add/del disjoint. Do not invent extra propositions. Use empty() exactly, NOT empty(hand).
"""
    feedback=[]
    for i in range(3):
        call=client.call(prompt+"\nPrevious validation: "+json.dumps(feedback),max_tokens=6000)
        row={"attempt":i+1,"call":call}
        try:
            fences=re.findall(r"```json\s*(.*?)```",call["raw"],re.S)
            if not fences:raise ValueError("No complete final JSON fence; no salvage from reasoning")
            schemas=json.loads(fences[-1])["schemas"]
            mapping={s["name"]:s for s in schemas}
            if set(mapping)!={"pick","place","home"}:raise ValueError("Expected pick/place/home schemas")
            models=[]
            for o,s in PAIRS:
                for kind in ("pick","place"):
                    models.append({"name":kind,"args":{"object":o,"support":s},**{f:[a.replace("OBJECT",o) for a in mapping[kind][f]] for f in ("pre","add","del")}})
            models.append({**mapping["home"],"args":{}})
            check=check_models(models,out/f"attempt{i+1}_trees");row.update(schemas=schemas,models=models,check=check,accepted=check["complete_for_P"] and check["admissible_reachable_states"])
        except Exception as exc:row.update(accepted=False,error=str(exc))
        save(out/f"attempt{i+1}.json",row);print("REASONED",i+1,row["accepted"],row.get("error"),flush=True)
        if row["accepted"]:
            save(out/"models.json",{"models":models,"proposal_model":args.model,"config_sha256":digest(CONFIG),"P":TASKS,"check":check,
                 "proposal_scope":"model-generated3 schemas within fixed CP/signatures, mechanical grounding to7actions, no human edits",
                 "raw_attempt":str(out/f"attempt{i+1}.json"),"schemas":schemas,"feedback_cycles":i})
            return
        feedback=[{"last_models":row.get("models"),"check":row.get("check"),"error":row.get("error")}]
    raise SystemExit(2)


if __name__=="__main__":main()
