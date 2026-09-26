"""Replay previously model-generated atomic Handover wrappers on handleless v3 physics.

The generated wrappers were produced before the box-handle geometry redesign, but
they only call the same six atomic API methods. This script does not claim visual-
only equivalence: it revalidates every STRIPS transition and the final physical goal
on the latest handle-free scene.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
STAGE3=ROOT.parent/"CABTO/exp4_bt_tasks/stage3_cabto"
sys.path.insert(0,str(STAGE3))
from core.handleless_tea_handover import HandlelessTeaRuntime, box_domain, goal
from core.smooth_handover_bridge import DOCS
from formal_bt import ModelLibrary, action_id
from policy_codegen import PolicyProgram, extract_source


def programs_from_cache(path):
    old=json.loads(path.read_text());programs={};evidence=[]
    for row in old["samples"]:
        attempt=next(a for a in reversed(row["attempts"]) if a["accepted"])
        source=attempt["source"]
        assert attempt.get("source_type")=="backend_generated"
        assert extract_source(attempt["raw"])==source
        assert hashlib.sha256(source.encode()).hexdigest()==attempt["sha256"]
        assert row["api_docs"]==DOCS[row["kind"]]
        programs[row["kind"]]=PolicyProgram(source,DOCS[row["kind"]])
        evidence.append({"kind":row["kind"],"source":source,"sha256":attempt["sha256"],"source_type":attempt["source_type"]})
    return programs,evidence


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=ROOT/"scenes/handover_tea_box_v3.json")
    ap.add_argument("--programs",type=Path,default=ROOT.parent/"tasks/handover/generated_bt/source_generated_programs.json")
    ap.add_argument("--seed",type=int,default=0);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.config.read_text());programs,evidence=programs_from_cache(a.programs)
    r=HandlelessTeaRuntime(cfg,a.output,a.seed,render=True);models=box_domain();target=goal()
    result={"task":"handleless_tea_handover_v3","program_source":"backend_generated_atomic_wrappers_from_prior_api-compatible_run",
            "physics_transfer":"not_assumed; fully_replayed_and_revalidated_on_handleless_v3","generated_programs":evidence,
            "steps":[],"success":False,"object_snap":False,"handles_exist":False}
    try:
        bt=ModelLibrary(models).build(r.state(),target)
        (a.output/"tree.json").write_text(bt.export());(a.output/"tree.dot").write_text(bt.export("dot"))
        for k in range(10):
            status,m=bt.tick(r.state())
            if status!="running":break
            before=r.state();p=programs[m["program_kind"]];api=r.api(m);p.validate_api(api);trace=p.run(api)["trace"];after=r.state()
            ok=bool(set(m["add"])<=after and not(set(m["del"])&after) and r.collision_free())
            result["steps"].append({"action_id":action_id(m),"before":sorted(before),"after":sorted(after),"trace":trace,"effect_ok":ok,"program_sha256":hashlib.sha256(p.source.encode()).hexdigest()})
            if not ok:break
        stable=r.stable_goal();status,_=bt.tick(r.state())
        result.update(success=bool(stable and status=="success" and all(x["effect_ok"] for x in result["steps"])),stable=stable,bt_status=status)
    except Exception as e:result["exception"]=f"{type(e).__name__}: {e}"
    finally:
        result.update(final_state=sorted(r.state()),placement=r.placement_geometry(),handover_verified=r.handover_verified(),collision_free=r.collision_free(),events=r.events)
        r.save(a.output);r.close();(a.output/"result.json").write_text(json.dumps(result,indent=2))
    print("HANDLELESS_GENERATED_FINAL",result["success"],result.get("exception"));raise SystemExit(0 if result["success"] else 1)

if __name__=="__main__":main()
