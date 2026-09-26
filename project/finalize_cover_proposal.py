"""Budgeted reasoning -> separate model finalization, with no contract patching."""
import argparse,json
from pathlib import Path
from core.cover_cabto_grounding import LocalModel,parse_json,save,PAIRS,TASKS,CONFIG,digest,check_models


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--reasoning",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();d=json.loads(a.reasoning.read_text());c=LocalModel(d["call"]["model_id"]);history=[]
    messages=[{"role":"system","content":"Complete the final answer from your analysis. Return ONLY valid JSON. No reasoning or explanation. Use the exact requested symbols."},
              {"role":"user","content":d["call"]["messages"]},
              {"role":"assistant","content":d["call"]["raw"]},
              {"role":"user","content":"Reasoning budget is exhausted. Give the final JSON now: {\"schemas\":[{\"name\":\"pick\",\"pre\":[...],\"add\":[...],\"del\":[...]}, place, home]}. Exactly 3 schemas. No Markdown. Use empty() and OBJECT exactly. Remember moving away from home invalidates home_completed()."}]
    for i in range(3):
        call=c.call(messages,max_tokens=1200,temperature=.2);row={"attempt":i+1,"call":call}
        try:
            schemas=parse_json(call["raw"])["schemas"];mapping={s["name"]:s for s in schemas}
            if set(mapping)!={"pick","place","home"}:raise ValueError("Need three named schemas")
            models=[]
            for o,s in PAIRS:
                for kind in ("pick","place"):
                    models.append({"name":kind,"args":{"object":o,"support":s},**{f:[x.replace("OBJECT",o) for x in mapping[kind][f]] for f in ("pre","add","del")}})
            models.append({**mapping["home"],"args":{}})
            check=check_models(models,a.output/f"attempt{i+1}_trees");row.update(models=models,schemas=schemas,check=check,accepted=check["complete_for_P"] and check["admissible_reachable_states"])
        except Exception as exc:row.update(accepted=False,error=str(exc))
        save(a.output/f"attempt{i+1}.json",row);history.append(row);print("FINALIZE",i+1,row["accepted"],row.get("error"),flush=True)
        if row["accepted"]:
            save(a.output/"models.json",{"models":models,"proposal_model":c.model_id,"config_sha256":digest(CONFIG),"P":TASKS,"check":check,"schemas":schemas,
                 "proposal_scope":"model-generated 3 schemas with bounded reasoning and separately recorded finalization; engineer-fixed CP/signatures; mechanical grounding; no host contract edits",
                 "reasoning_input":str(a.reasoning),"reasoning_sha256":digest(a.reasoning),"source_attempt":str(a.output/f"attempt{i+1}.json"),"feedback_cycles":i})
            return
        feedback=row.get("check",row.get("error"))
        messages.extend([{"role":"assistant","content":call["raw"]},{"role":"user","content":"Fix your final model based on actual validator errors and return full JSON schemas only:\n"+json.dumps(feedback)}])
    raise SystemExit(2)


if __name__=="__main__":main()
