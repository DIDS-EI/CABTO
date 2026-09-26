"""Independent result/provenance checks, no simulator writes or model inference."""
from pathlib import Path
import json,hashlib
from html.parser import HTMLParser
from core.cover_cabto_grounding import digest
from policy_codegen import extract_source
from run_remaining_cabto import plancheck,config_path,waypoint_provenance,docs
from policy_codegen import PolicyProgram
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/remaining_cabto"
SAMPLES={"blocks":"sampling","pour":"sampling_v5","handover":"sampling_v3","storage":"sampling_v2"}

def read(p):return json.loads(Path(p).read_text())

def origin(source,where,depth=0):
    assert depth<12
    if "#" in where:
        path,kind=where.rsplit("#",1);data=read(path)
        for row in data["samples"]:
            if row.get("kind")!=kind:continue
            for a in reversed(row["attempts"]):
                if a.get("accepted") and a.get("source")==source:
                    assert extract_source(a["raw"])==source
                    return {"source":where,"type":"archived_model_raw_output","backend":a.get("backend"),"transfer_depth":depth}
        raise AssertionError("Archive does not contain matching accepted raw program")
    path=Path(where)
    file=path/"attempt.json" if (path/"attempt.json").exists() else path/"feedback.json"
    data=read(file);assert data["accepted"] and data["source"]==source
    if "call" in data:
        assert extract_source(data["call"]["raw"])==source
        return {"source":str(file),"type":"actual_model_raw_output","backend":data["call"]["model_id"],"transfer_depth":depth}
    if "transfer" in data:return origin(source,data["transfer"]["source_attempt"],depth+1)
    if "schema_transfer" in data:return origin(source,data["schema_transfer"]["from_attempt"],depth+1)
    raise AssertionError("No model raw source for accepted program")

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ("src","href","poster") and not v.startswith(("http","data:","#")))

def main():
    records={};count=0;transitions=0
    for task,sampling in SAMPLES.items():
        p=OUT/task;d=read(p/("proposal" if task=="blocks" else "refinement")/"models.json")
        cache=read(p/sampling/"policies.json");assert cache["success"]
        assert d["config_sha256"]==digest(config_path(task))
        assert plancheck(d["models"],d["P"])["complete_for_declared_P"]
        provenance={}
        for key,e in cache["accepted"].items():
            assert hashlib.sha256(e["source"].encode()).hexdigest()==e["sha256"]
            provenance[key]=origin(e["source"],e["source_attempt"])
            waypoint_provenance(PolicyProgram(e["source"],docs(task,e["model"])))
        if task!="blocks":
            for patchfile in (p/"refinement").glob("patch*.json"):
                patch=read(patchfile)
                if not patch["accepted"]:continue
                assert patch["call"]["model_id"]=="mlx-community/Qwen3.5-9B-4bit"
                from core.cover_cabto_grounding import parse_json
                assert parse_json(patch["call"]["raw"])==patch["patch"]
                w=patch["witness"];m=patch["model"]
                assert (set(w["before"])|set(m["add"]))-set(m["del"])==set(w["after"])
        for seed in (0,1):
            run=read(p/f"final_seed{seed}/result.json")
            assert run["success_count"]==run["total"]
            assert run["sources_unchanged"]
            assert digest(run["model_input"])==run["model_sha256"]
            assert digest(run["policies_input"])==run["policies_sha256"]
            for source,h in run["source_hashes"].items():assert digest(source)==h,source
            for e in run["episodes"]:
                assert e["success"] and e["full_frame_all_match"]
                video=p/f"final_seed{seed}"/e["task"]["id"]/"rollout.mp4";assert video.stat().st_size>1000
                for step in e["steps"]:
                    assert step["effect_ok"] and step["valid_trial"] and step["strict_CP_transition_match"]
                    assert not step["exception"] and not step["missing_required_calls"]
                    transitions+=1
                count+=1
        records[task]={"actions":len(cache["accepted"]),"policy_origins":provenance,"config_sha256":d["config_sha256"]}
    parser=Links();parser.feed((OUT/"index.html").read_text());missing=[r for r in parser.refs if r!="verification.json" and not (OUT/r).is_file()];assert not missing,missing
    for file,n in [("remaining_tests.log",36),("visual_regression.log",25),("generation_regression.log",8),("cover_regression.log",44)]:
        text=(OUT/file).read_text();assert f"Ran {n} tests" in text and "OK" in text and "FAILED" not in text
    assert "ALL 164 TESTS PASSED" in (OUT/"legacy_tests.log").read_text()
    report={"passed":True,"final_episode_count":count,"strict_frame_transition_count":transitions,"tests":277,"report_links":len(parser.refs),"broken_links":missing,
            "task_sources":records,"scope":"finite_execution_and_source_integrity; dual_oracle+weld; no universal consistency proof"}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in report.items() if k!="task_sources"},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
