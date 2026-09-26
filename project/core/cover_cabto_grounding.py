"""CABTO phases for the frozen kitchen scene; genuine local-model proposals.
No dependency on the earlier atomic/no-op replay adapter.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(PACKAGE / "CABTO/exp4_bt_tasks/stage3_cabto"))
from formal_bt import ModelLibrary, validate_model, action_id, symbolic_dry_run
from policy_codegen import PolicyProgram, extract_source

PAIRS = [("shrimp", "bowl"), ("apple", "board"), ("potato", "pan")]
CONFIG = ROOT / "scenes/cover_kitchen_sort_v2.json"
CP = {"empty()", "home_completed()"} | {f"{p}({obj})" for obj, _ in PAIRS for p in ("at_source", "holding", "at_target")}
INITIAL = {"empty()"} | {f"at_source({obj})" for obj, _ in PAIRS}
TASKS = [{"id": f"C{n}", "initial": sorted(INITIAL), "goal": sorted({"empty()", "home_completed()"} | {f"at_target({o})" for o, _ in PAIRS[:n]})} for n in (1, 2, 3)]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LocalModel:
    """Own instance per exact model ID; raw prompts/replies persisted by callers."""
    def __init__(self, model_id="mlx-community/Qwen3.5-0.8B-4bit"):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model_id = model_id
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

    def call(self, text, images=(), max_tokens=2400, temperature=0.0):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = apply_chat_template(self.processor, self.config, text, num_images=len(images), enable_thinking=False)
        started = time.perf_counter()
        args = {"max_tokens": max_tokens, "temperature": temperature, "verbose": False}
        if images: args["image"] = [str(p) for p in images]
        output = generate(self.model, self.processor, prompt, **args)
        return {"model_id": self.model_id, "text": text, "image_files": [str(p) for p in images],
                "formatted_prompt": prompt, "temperature": temperature, "max_tokens": max_tokens,
                "raw": output.text if hasattr(output, "text") else str(output),
                "wall_seconds": time.perf_counter() - started}


def parse_json(raw):
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try: return decoder.raw_decode(text[i:])[0]
            except ValueError: pass
    raise ValueError("No JSON object in model response")


def state_valid(s):
    held = [f"holding({o})" in s for o, _ in PAIRS]
    if sum(held) > 1 or (("empty()" in s) == any(held)): return False
    if "home_completed()" in s and "empty()" not in s: return False
    for o, _ in PAIRS:
        if sum(f"{p}({o})" in s for p in ("at_source", "holding", "at_target")) != 1: return False
    return True


def check_models(models, out=None):
    ids = []
    for m in models:
        ids.append(validate_model(m))
        unknown = set().union(*(set(m[k]) for k in ("pre", "add", "del"))) - CP
        if unknown: raise ValueError(f"Unknown CP atoms {sorted(unknown)}")
        if m["name"] not in ("pick", "place", "home"): raise ValueError("Admissible skill signatures: pick, place, home")
        if m["name"] == "home":
            if m["args"]: raise ValueError("home has no args")
        else:
            if set(m["args"]) != {"object", "support"} or tuple(m["args"][k] for k in ("object", "support")) not in PAIRS:
                raise ValueError("args must bind one valid object/support pair")
    if len(set(ids)) != len(ids): raise ValueError("Duplicate action IDs")
    # Independent finite BFS + mutex validation. This checks the proposed symbolic
    # model, not the real physical transition.
    seen = {frozenset(INITIAL)}; queue = deque([frozenset(INITIAL)]); violations = []
    while queue:
        s = set(queue.popleft())
        for m in models:
            if set(m["pre"]) <= s:
                nxt = (s | set(m["add"])) - set(m["del"])
                if not state_valid(nxt):
                    violations.append({"action": action_id(m), "before": sorted(s), "after": sorted(nxt)})
                f = frozenset(nxt)
                if f not in seen: seen.add(f); queue.append(f)
        if len(seen) > 2048: raise ValueError("CP state bound exceeded")
    tasks = []
    for t in TASKS:
        rec = {"task": t["id"], "bfs_reachable": any(set(t["goal"]) <= s for s in seen)}
        try:
            bt = ModelLibrary(models).build(INITIAL, set(t["goal"]))
            rec["planner"] = symbolic_dry_run(bt, INITIAL, set(t["goal"]), 128)
            rec["bt_nodes"] = len(json.loads(bt.export())["nodes"])
            if out:
                p = Path(out) / t["id"]; p.mkdir(parents=True, exist_ok=True)
                (p / "tree.json").write_text(bt.export()); (p / "tree.dot").write_text(bt.export("dot"))
        except Exception as exc:
            rec["planner"] = {"reached_goal": False, "reason": str(exc)}
        tasks.append(rec)
    return {"complete_for_P": all(t["bfs_reachable"] and t["planner"]["reached_goal"] for t in tasks),
            "admissible_reachable_states": not violations, "reachable_state_count": len(seen),
            "mutex_violations": violations[:12], "tasks": tasks,
            "scope": "finite declared P, symbolic; NOT physical consistency"}


def propose(out, model_id, attempts=3):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    client = LocalModel(model_id)
    base = {"request": "Propose a STRIPS action-model library sufficient to solve EVERY task in P. Return JSON only, key models with a list. Do not generate policies or motion coordinates.",
            "scene": json.loads(CONFIG.read_text())["specification"], "objects_supports": PAIRS,
            "P": TASKS, "CP": sorted(CP),
            "atom_semantics": {"empty()": "gripper open and empty", "at_source(o)": "o still at its original table location", "holding(o)": "o held by closed gripper above the table", "at_target(o)": "o released and physically supported by its matching support", "home_completed()": "arm returned to HOME after work with open empty gripper"},
            "HP_constraints": ["Each JSON action has name, args, pre, add, del. pre/add/del are lists of exact CP strings.", "Only admissible names/signatures: pick(object,support), place(object,support), home(). Pick/place args must bind one listed pair. No predefined effects are supplied; choose effects and preconditions.", "Use grounded action dictionaries, not variables or schemas. args is {object: item, support: matching_support}; home args is {}.", "pre and add must be disjoint; add and del must be disjoint.", "At action boundaries each item is exactly one of at_source, holding, at_target. At most one held object; empty iff none held.", "All three tasks share the initial state. Do not require all three placements for the home action if this makes C1 or C2 unsolvable."]}
    history = []
    for i in range(attempts):
        prompt = json.dumps({**base, "previous_attempts_and_planner_feedback": history}, ensure_ascii=False)
        call = client.call(prompt, max_tokens=2600, temperature=0.2 * i)
        row = {"attempt": i + 1, "call": call}
        try:
            models = parse_json(call["raw"])["models"]
            check = check_models(models, out / f"attempt{i+1}_trees")
            row.update(models=models, check=check)
            row["accepted"] = check["complete_for_P"] and check["admissible_reachable_states"]
        except Exception as exc: row.update(accepted=False, error=f"{type(exc).__name__}: {exc}")
        save(out / f"attempt{i+1}.json", row)
        print("MODEL_PROPOSAL", i + 1, row["accepted"], row.get("error"), flush=True)
        if row["accepted"]:
            save(out / "models.json", {"models": models, "proposal_model": model_id, "source_attempt": i+1,
                                       "config_sha256": digest(CONFIG), "P": TASKS, "check": check})
            return models
        history.append({"raw_model_answer": call["raw"], "feedback": row.get("check", row.get("error"))})
    save(out / "blocked.json", {"status": "no_complete_admissible_model_library", "attempts": len(history)})
    raise RuntimeError("Model proposal budget exhausted; see high-level feedback")


def propose_structured(out, model_id, attempts=3, client=None):
    """Restricted HP: operator signatures are supplied; all contracts are generated.
    This interface restriction is explicit and is not unconstrained action discovery.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True);client=client or LocalModel(model_id)
    signatures=[{"name":kind,"args":{"object":o,"support":s}} for o,s in PAIRS for kind in ("pick","place")]+[{"name":"home","args":{}}]
    library=[];calls=[];feedback=[]
    for cycle in range(attempts):
        library=[]
        for i,signature in enumerate(signatures):
            o=signature["args"].get("object");s=signature["args"].get("support")
            semantics=({"pick":f"Pick up {o} from its source on the table, leave it held above the table by the gripper.",
                        "place":f"Place the already held {o} on/in {s}, release it and withdraw the hand.",
                        "home":"Return the empty gripper to HOME after any requested subset of placements."})[signature["name"]]
            atoms=sorted({"empty()","home_completed()"}|({f"{p}({o})" for p in ("at_source","holding","at_target")} if o else set()))
            prompt=("Write the STRIPS contract of ONE action. Return ONLY JSON with three keys pre, add, del, each an array of literal condition strings. Do not repeat the task list; no code.\n"
                    +"Action signature: "+json.dumps(signature)+"\nMeaning: "+semantics
                    +"\nAllowed condition strings: "+json.dumps(atoms)
                    +"\nConditions: empty() means open empty gripper; holding(obj) means held above table; at_source(obj) means at original table source; at_target(obj) means released at its matching support; home_completed() means at HOME with empty open gripper. Leaving HOME invalidates home_completed()."
                    +"\npre describes what must hold BEFORE. add describes newly true facts AFTER. del describes facts made false. An atom cannot be both pre and add, or both add and del. Each object must occupy exactly one state at action boundaries."
                    +"\nTask context: independently complete shrimp->bowl; shrimp->bowl+apple->board; or all three including potato->pan. Each finishes empty at HOME. All start empty with the three objects at source."
                    +"\nPrior planner feedback: "+json.dumps(feedback))
            call=client.call(prompt,max_tokens=700,temperature=.15*cycle)
            row={"cycle":cycle+1,"signature":signature,"call":call}
            try:
                contract=parse_json(call["raw"])
                m={**signature,**{k:contract[k] for k in ("pre","add","del")}}
                validate_model(m);library.append(m);row["model"]=m
            except Exception as exc:row["error"]=str(exc)
            calls.append(row);save(out/f"cycle{cycle+1}_action{i}.json",row)
        try:
            check=check_models(library,out/f"cycle{cycle+1}_trees")
            ok=len(library)==len(signatures) and check["complete_for_P"] and check["admissible_reachable_states"]
        except Exception as exc:check={"error":str(exc)};ok=False
        print("STRUCTURED_PROPOSAL",cycle+1,ok,check,flush=True)
        save(out/f"cycle{cycle+1}_check.json",check)
        if ok:
            save(out/"models.json",{"models":library,"proposal_model":model_id,"config_sha256":digest(CONFIG),"P":TASKS,"check":check,
                 "proposal_scope":"model-generated pre/add/del within seven engineer-specified grounded HP signatures; no human contract edits",
                 "calls":calls,"feedback_cycles":cycle})
            return library
        feedback=[{"check":check,"previous_models":library,"format_errors":[r["error"] for r in calls[-7:] if "error" in r]}]
    save(out/"blocked.json",{"reason":"structured model proposal budget exhausted","feedback":feedback})
    raise RuntimeError("Structured proposal failed")


def propose_indexed(out, model_id, attempts=3):
    """Generate 3 relational contracts with indexed atoms then ground by substitution.
    Only serialization and grounding are mechanical; no contract edits by host.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True);client=LocalModel(model_id)
    atoms=["empty()","at_source(OBJECT)","holding(OBJECT)","at_target(OBJECT)","home_completed()"]
    meanings={"pick":"PICKING UP an object from its table source and leaving it HELD, not placed.",
              "place":"PLACING an ALREADY HELD object at destination and opening the gripper, not picking it up.",
              "home":"RETURNING an OPEN EMPTY robot arm to HOME. Do not touch or change objects."}
    contracts={};calls=[]
    for kind in ("pick","place","home"):
        feedback=[]
        for attempt in range(attempts):
            prompt=("Write a STRIPS contract for "+meanings[kind]+"\n"
                "Choose integer indices only from: 0=hand empty, 1=object at table source, 2=object held, 3=object released at destination, 4=arm home. "
                "Return JSON with pre, add, del arrays. pre means required BEFORE; add means newly true AFTER; del means made false AFTER. "
                "pre and add disjoint; add and del disjoint. Exactly one object location among 1,2,3 must remain; empty and held cannot coexist. Leaving home invalidates fact4. "
                "For HOME, only facts0 and4 are admissible. Task set: place shrimp, or shrimp+apple, or all3; end empty at home. All start at source with empty gripper.\n"
                "Previous proposal and validator feedback: "+json.dumps(feedback))
            c=client.call(prompt,max_tokens=400,temperature=.2*attempt);row={"signature":kind,"call":c,"attempt":attempt+1}
            try:
                raw=parse_json(c["raw"])
                for f in ("pre","add","del"):
                    if not isinstance(raw[f],list) or any(type(v)!=int or v not in range(5) for v in raw[f]): raise ValueError("Use ONLY integer arrays with indices0..4")
                if kind=="home" and any(v not in (0,4) for f in ("pre","add","del") for v in raw[f]):raise ValueError("HOME may only reference0 and4")
                m={"name":kind,"args":{},**{f:[atoms[v] for v in raw[f]] for f in ("pre","add","del")}}
                validate_model(m)
                # Enumerate valid local states to detect mutex-violating operators.
                states=[{0,1},{0,1,4},{2},{0,3},{0,3,4}]
                wrong=[]
                for s in states:
                    if set(raw["pre"])<=s:
                        nxt=(s|set(raw["add"]))-set(raw["del"])
                        if sum(i in nxt for i in (1,2,3))!=1 or (0 in nxt)==(2 in nxt) or (4 in nxt and 0 not in nxt): wrong.append({"before":sorted(s),"after":sorted(nxt)})
                if wrong:raise ValueError("Mutex violations: "+json.dumps(wrong))
                contracts[kind]=m;row["accepted"]=True;row["model"]=m
            except Exception as exc:row.update(accepted=False,error=str(exc))
            calls.append(row);save(out/f"{kind}_attempt{attempt+1}.json",row)
            print("INDEXED",kind,attempt+1,row["accepted"],row.get("error"),flush=True)
            if row["accepted"]:break
            feedback=[{"raw":c["raw"],"error":row["error"]}]
        else:save(out/"blocked.json",{"kind":kind,"calls":calls});raise RuntimeError("Indexed contract budget exhausted")
    library=[]
    for o,s in PAIRS:
        for kind in ("pick","place"):
            m=contracts[kind];library.append({"name":kind,"args":{"object":o,"support":s},**{f:[a.replace("OBJECT",o) for a in m[f]] for f in ("pre","add","del")}})
    library.append(contracts["home"])
    check=check_models(library,out/"trees")
    save(out/"models.json",{"models":library,"proposal_model":model_id,"config_sha256":digest(CONFIG),"P":TASKS,"check":check,"calls":calls,
        "proposal_scope":"3 engineer-named signatures, LM-generated indexed pre/add/del, mechanical grounding to7 actions; no host contract edits"})
    if not check["complete_for_P"] or not check["admissible_reachable_states"]:raise RuntimeError("Indexed model library not complete/admissible")
    return library
