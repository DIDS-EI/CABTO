"""Independent provenance, physical-scene and output integrity checks for the fix."""
import ast,hashlib,json
from pathlib import Path
from html.parser import HTMLParser
import mujoco
import numpy as np
from core.cover_cabto_grounding import CONFIG,check_models,digest
from core.single_arm_rebuilt import scene
from policy_codegen import extract_source

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/cover_cabto_fix"

def read(p):return json.loads(Path(p).read_text())


def policy_origin(entry,depth=0):
    assert depth<8
    p=Path(entry["source_attempt"])/"feedback.json";d=read(p)
    assert d["accepted"] and d["source"]==entry["source"]
    assert hashlib.sha256(entry["source"].encode()).hexdigest()==entry["sha256"]
    if "call" in d:
        assert d["call"]["model_id"]=="mlx-community/Qwen3.5-9B-4bit"
        assert extract_source(d["call"]["raw"])==entry["source"]
        return {"original_call":str(p),"model":d["call"]["model_id"],"schema_transfer_depth":depth}
    transfer=d["schema_transfer"]
    assert transfer["code_modified"] is False and transfer["new_model_call"] is False
    return policy_origin({**entry,"source_attempt":transfer["from_attempt"]},depth+1)


class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        self.paths.extend(v for k,v in attrs if k in ("href","src","poster") and not v.startswith(("http","data:","#")))


def main():
    p=read(OUT/"proposal_final_v2/models.json");policies=read(OUT/"sampling_9b_complete/policies.json")
    assert p["config_sha256"]==digest(CONFIG)
    check=check_models(p["models"]);assert check["complete_for_P"] and check["admissible_reachable_states"]
    assert policies["success"] and policies["generation_channel"]=="local_live_model"
    origins={k:policy_origin(v) for k,v in policies["accepted"].items()};assert len(origins)==7
    physics_checks=[];total=0
    for folder in ("final_visual","final_holdout1","final_holdout2"):
        d=read(OUT/folder/"result.json");assert d["success_count"]==d["total"] and d["sources_unchanged"]
        for f,h in d["source_hashes"].items():assert digest(f)==h,f
        assert d["proposal_input_sha256"]==digest(OUT/"proposal_final_v2/models.json")
        assert d["policies_input_sha256"]==digest(OUT/"sampling_9b_complete/policies.json")
        assert d["proposal_origin"]==d["policy_origin"]=="mlx-community/Qwen3.5-9B-4bit"
        for e in d["episodes"]:
            assert e["success"] and not e["weld_active"] and not e["metrics"]["strict_weld_violation"]
            assert not e["metrics"]["forbidden_contacts"]
            assert all(s["strict_CP_transition_match"] and s["effect_ok"] for s in e["steps"])
            xml=OUT/folder/e["task"]["id"]/"scene.xml"
            actual=mujoco.MjModel.from_xml_path(str(xml))
            reference_xml,_=scene(read(CONFIG),e["seed"]);reference=mujoco.MjModel.from_xml_string(reference_xml)
            for attribute in ("geom_size","geom_pos","geom_quat","geom_friction","geom_contype","geom_conaffinity","geom_solref","geom_solimp","body_pos","body_quat","body_mass","body_inertia","jnt_pos","jnt_range","qpos0"):
                np.testing.assert_allclose(getattr(actual,attribute),getattr(reference,attribute),atol=1e-12,rtol=0,err_msg=attribute)
            assert not np.allclose(actual.cam_pos,reference.cam_pos)
            physics_checks.append({"task":e["task"]["id"],"seed":e["seed"],"physical_arrays_equal":True,"camera_explicitly_changed":True});total+=1
    parser=Links();parser.feed((OUT/"index.html").read_text())
    missing=[p for p in parser.paths if not (OUT/p).is_file()];assert not missing,missing
    logs={"visual_tests.log":25,"generation_tests.log":8,"method_tests.log":44}
    for file,count in logs.items():
        text=(OUT/file).read_text();assert f"Ran {count} tests" in text and "OK" in text and "FAILED" not in text
    assert "ALL 164 TESTS PASSED" in (OUT/"legacy_tests.log").read_text()
    for source in (ROOT/"project").glob("*.py"):
        if "cover" in source.name:ast.parse(source.read_text())
    report={"passed":True,"successful_episodes":total,"models_complete_for_P":True,"original_local_model_calls":origins,
            "physics_checks":physics_checks,"report_links":len(parser.paths),"broken_links":missing,
            "tests":241,"final_source_hashes_match":True,"model_and_code_inputs_match":True}
    (OUT/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ("original_local_model_calls","physics_checks")},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
