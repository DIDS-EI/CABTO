"""Repair only model-generated proposals using concise exact validator feedback.
No correct library or session-generated candidate is consulted.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from core.cover_cabto_grounding import (LocalModel,parse_json,check_models,save,CONFIG,digest,TASKS,CP,PAIRS)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--model",default="mlx-community/Qwen3.5-9B-4bit");ap.add_argument("--attempts",type=int,default=5)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source.read_text());candidate=source["models"]
    client=LocalModel(args.model);history=[]
    system="You repair STRIPS action contracts using symbolic validation errors. Return ONLY a JSON object with key models and a list of action dictionaries. Each action requires name, args, pre, add, del. Do not invent new predicates. Never output Python, an explanation or the feedback itself."
    semantics=("The only facts are "+json.dumps(sorted(CP))+". Each object is exactly one of at_source, holding, at_target. "
               "empty() means the hand holds NOTHING; closing around and lifting an object makes this false. "
               "home_completed() means the arm remains at HOME with an empty open hand; leaving HOME makes it false. "
               "place means release the held object at its destination: after place the hand is empty. "
               "A true atom persists unless del explicitly removes it. pre/add must be disjoint; add/del must be disjoint. "
               "Pick and place each concern exactly their bound object, not all three. home() may follow any subset of completed placements. "
               "Tasks start with all objects at_source and empty hand; goals are shrimp placed, shrimp+apple placed, all three placed, each with empty and home_completed. No change to task goals.")
    for i in range(args.attempts):
        try:check=check_models(candidate)
        except Exception as exc:check={"structural_error":str(exc)}
        # Keep at most one example of each action type so feedback fits the model.
        compact={"complete_for_P":check.get("complete_for_P"),"structural_error":check.get("structural_error"),"counterexamples":[]}
        seen=set()
        for v in check.get("mutex_violations",[]):
            kind=v["action"].split("(")[0]
            if kind not in seen:seen.add(kind);compact["counterexamples"].append(v)
        compact["planning"]=check.get("tasks",[])
        messages=[{"role":"system","content":system},{"role":"user","content":semantics},
                  {"role":"assistant","content":json.dumps({"models":candidate})},
                  {"role":"user","content":"This proposal failed. Actual validator report:\n"+json.dumps(compact)+"\nReturn the FULL corrected models library. Repair the action effects, not the report. Never retain both empty() and holding at the same time."}]
        call=client.call(messages,max_tokens=2400,temperature=.2+i*.1);row={"attempt":i+1,"input_feedback":compact,"call":call}
        try:
            candidate=parse_json(call["raw"])["models"];check=check_models(candidate,args.output/f"attempt{i+1}_trees")
            row.update(models=candidate,check=check,accepted=check["complete_for_P"] and check["admissible_reachable_states"])
        except Exception as exc:row.update(accepted=False,error=str(exc))
        history.append(row);save(args.output/f"attempt{i+1}.json",row)
        print("MODEL_REPAIR",i+1,row["accepted"],row.get("error"),flush=True)
        if row["accepted"]:
            save(args.output/"models.json",{"models":candidate,"proposal_model":args.model,"config_sha256":digest(CONFIG),"P":TASKS,"check":check,
                 "model_generated_source":str(args.source),"source_sha256":digest(args.source),"feedback_cycles":i+1,
                 "proposal_scope":"engineer-defined CP and pick/place/home signatures; model-generated contracts with semantic and planner feedback; no host contract edits"})
            return
    save(args.output/"blocked.json",{"status":"budget_exhausted","attempts":len(history)})
    raise SystemExit(2)


if __name__=="__main__":main()
