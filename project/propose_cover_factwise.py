"""LM classifies each condition's prerequisite/effect role; deterministic grounding.
A restricted fact-wise proposal interface, not free action discovery. No known-good
contract or policy is read. All Boolean/effect choices must come from actual replies.
"""
import argparse,json
from pathlib import Path
from core.cover_cabto_grounding import LocalModel,parse_json,PAIRS,TASKS,CONFIG,digest,save,check_models

ATOMS={"empty()":"the robot hand is empty and open",
       "at_source(OBJECT)":"the object is still resting at its original table source",
       "holding(OBJECT)":"the object is held by the closed gripper in the air",
       "at_target(OBJECT)":"the object is released and resting on its destination support",
       "home_completed()":"the empty open robot arm is at HOME"}
MEANING={"pick":"grasp an object from its table source and lift it, leaving it held in the air; the arm moves away from HOME",
         "place":"move an ALREADY HELD object to its destination, lower, release and withdraw the hand; do not go HOME",
         "home":"move an EMPTY OPEN arm to HOME; do not touch any object"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--model",default="mlx-community/Qwen3.5-9B-4bit")
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);client=LocalModel(args.model);feedback={}
    for cycle in range(3):
        schemas={};all_calls=[]
        for kind in ("pick","place","home"):
            schema={"pre":[],"add":[],"del":[]};atoms=ATOMS if kind!="home" else {k:v for k,v in ATOMS.items() if k in ("empty()","home_completed()")}
            for atom,meaning in atoms.items():
                prompt=(f"Action: {MEANING[kind]}.\nAnalyze ONLY this proposition: {meaning}.\n"
                        "Question 1: must that proposition be TRUE before starting this action? required_true=true or false. "
                        "Question 2: what does completing this action do to that proposition? effect=add (makes it true), delete (makes it false), or keep (does not change it). "
                        "If it must be true beforehand and stays true, use keep, not add. False preconditions are not represented. "
                        "Return ONLY JSON {\"required_true\": boolean, \"effect\": \"add|delete|keep\"}. "
                        "Do NOT discuss other propositions.\n"+"Prior validator feedback: "+json.dumps(feedback))
                call=client.call(prompt,max_tokens=160,temperature=.15+.1*cycle)
                row={"cycle":cycle+1,"signature":kind,"atom":atom,"call":call}
                try:
                    vote=parse_json(call["raw"])
                    if type(vote.get("required_true")) is not bool or vote.get("effect") not in ("add","delete","keep"):
                        raise ValueError("Classification response outside declared schema")
                    row["classification"]=vote
                    if vote["required_true"]:schema["pre"].append(atom)
                    if vote["effect"]=="add":schema["add"].append(atom)
                    if vote["effect"]=="delete":schema["del"].append(atom)
                except Exception as exc:row["error"]=str(exc)
                all_calls.append(row);save(out/f"cycle{cycle+1}_{kind}_{len(all_calls)}.json",row)
            schemas[kind]=schema
        models=[]
        for o,s in PAIRS:
            for kind in ("pick","place"):
                models.append({"name":kind,"args":{"object":o,"support":s},**{f:[a.replace("OBJECT",o) for a in schemas[kind][f]] for f in ("pre","add","del")}})
        models.append({"name":"home","args":{},**schemas["home"]})
        try:
            check=check_models(models,out/f"cycle{cycle+1}_trees")
            ok=check["complete_for_P"] and check["admissible_reachable_states"] and not any("error" in r for r in all_calls)
        except Exception as exc:check={"error":str(exc)};ok=False
        save(out/f"cycle{cycle+1}_summary.json",{"models":models,"calls":all_calls,"check":check,"accepted":ok})
        print("FACTWISE",cycle+1,ok,schemas,flush=True)
        if ok:
            save(out/"models.json",{"models":models,"proposal_model":args.model,"config_sha256":digest(CONFIG),"P":TASKS,"check":check,
                 "proposal_scope":"model classifies each atom pre/effect role for3 engineer-specified signatures; mechanical grounding; no host contract edits",
                 "schemas":schemas,"calls":all_calls,"feedback_cycles":cycle})
            return
        feedback={"violations":check.get("mutex_violations",[])[:2],"error":check.get("error"),"previous_schemas":schemas}
    save(out/"blocked.json",feedback);raise SystemExit(2)


if __name__=="__main__":main()
