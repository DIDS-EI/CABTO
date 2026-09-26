"""Repair the specific offending schema selected by an actual mutex counterexample."""
import json,argparse
from pathlib import Path
from core.cover_cabto_grounding import LocalModel,parse_json,save,PAIRS,TASKS,CONFIG,digest,check_models


def ground(mapping):
    models=[]
    for o,s in PAIRS:
        for kind in ("pick","place"):
            models.append({"name":kind,"args":{"object":o,"support":s},**{f:[x.replace("OBJECT",o) for x in mapping[kind][f]] for f in ("pre","add","del")}})
    return models+[{**mapping["home"],"args":{}}]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    src=json.loads(a.source.read_text());mapping={m["name"]:m for m in src["schemas"]};c=LocalModel("mlx-community/Qwen3.5-9B-4bit")
    for i in range(4):
        check=check_models(ground(mapping));example=check["mutex_violations"][0]
        kind=example["action"].split("(")[0]
        prompt=("Correct ONE STRIPS action based on this counterexample. Return ONLY JSON with name,pre,add,del for this action. Use OBJECT literally.\n"
                "Current action: "+json.dumps(mapping[kind])+"\nActual counterexample from symbolic validator: "+json.dumps(example)
                +"\nRule violated: home_completed() means arm is at HOME with an EMPTY OPEN gripper. The arm cannot be home_completed() while holding an object. This action moves the robot away from HOME. "
                "Facts persist unless explicitly removed by del. The action must work even if it starts at HOME. Do not hide the invalid state by requiring home_completed() or by changing the goal. "
                "The offending action is "+kind+", NOT place or home. Preserve its pickup semantics; no new predicates; pre/add and add/del must be disjoint.")
        call=c.call(prompt,max_tokens=650,temperature=.3);row={"attempt":i+1,"counterexample":example,"call":call}
        try:
            m=parse_json(call["raw"])
            if m["name"]!=kind:raise ValueError("Wrong action")
            mapping[kind]={k:m[k] for k in ("name","pre","add","del")}
            models=ground(mapping);newcheck=check_models(models,a.output/f"attempt{i+1}_trees")
            row.update(revised_schema=m,schemas=list(mapping.values()),models=models,check=newcheck,
                       accepted=newcheck["complete_for_P"] and newcheck["admissible_reachable_states"])
        except Exception as exc:row.update(accepted=False,error=str(exc))
        save(a.output/f"attempt{i+1}.json",row);print("SINGLE_MODEL_REPAIR",i+1,row["accepted"],row.get("error"),flush=True)
        if row["accepted"]:
            save(a.output/"models.json",{"models":models,"proposal_model":c.model_id,"config_sha256":digest(CONFIG),"P":TASKS,"check":newcheck,
                 "schemas":list(mapping.values()),"raw_source":str(a.source),"raw_source_sha256":digest(a.source),
                 "final_call_source":str(a.output/f"attempt{i+1}.json"),"feedback_cycles":i+1,
                 "proposal_scope":"LM-generated schemas, finalization, then targeted validator counterexample; manual CP/signature/semantic constraints, no host edits to model effects"})
            return
    raise SystemExit(2)


if __name__=="__main__":main()
