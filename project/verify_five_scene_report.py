"""Validate the five-scene presentation, image provenance and preserved results.
Independent from physical trials; never updates their success or perception type.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib,json
from PIL import Image
from core.cover_cabto_api import Perception

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/remaining_cabto"
EXPECTED={"cover":6,"blocks":4,"pour":3,"handover":3,"storage":4}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Document(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set();self.scenes=[];self.cards=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if "id" in a:self.ids.add(a["id"])
        if "data-scene" in a:self.scenes.append(a["data-scene"])
        if "point-card" in a.get("class","").split():self.cards.append(a["data-kind"])
        if tag=="img" and "src" in a:self.images.append(a["src"])
        for k in ("href","src","poster","data-original","data-overlay","data-full"):
            if k in a:self.links.append(a[k])

def main():
    doc=Document();doc.feed((OUT/"index.html").read_text());gallery=read(OUT/"camera_pointing/manifest.json");summary=read(OUT/"summary.json")
    assert doc.scenes==list(EXPECTED)
    assert list(summary["tasks"])==list(EXPECTED)
    assert len(doc.cards)==20 and doc.cards.count("online")==10 and doc.cards.count("posthoc")==10
    missing=[]
    for url in doc.links:
        if url.startswith(("http:","https:","data:")):continue
        parsed=urlsplit(url)
        if not parsed.path:
            assert parsed.fragment in doc.ids,parsed.fragment
        elif parsed.path!="presentation_verification.json" and not (OUT/unquote(parsed.path)).is_file():missing.append(url)
    assert not missing,missing
    origins=[]
    for task,n in EXPECTED.items():
        records=gallery["scenes"][task]["items"];assert len(records)==n
        for item in records:
            evidence=read(OUT/item["record"]);inputpath=OUT/item["input"];overlay=OUT/item["overlay"]
            assert sha(inputpath)==item["input_sha256"] and sha(overlay)==item["overlay_sha256"]
            with Image.open(inputpath) as image:assert list(image.size)==item["image_size"]
            with Image.open(overlay) as image:assert list(image.size)==item["image_size"]
            if task in ("cover","blocks"):
                assert item["used_in_control"] is True and evidence["origin"]=="existing_closed_loop_call"
                assert sha(evidence["source_json"])==evidence["source_sha256"]
                original=read(evidence["source_json"])[evidence["record_index"]]
                assert original==evidence["record"]
                call=original["call"];assert sha(call["image_files"][0])==sha(inputpath)
            else:
                assert item["used_in_control"] is False and evidence["used_in_control"] is False
                assert evidence["original_execution_perception"]=="oracle_geometry"
                assert evidence["origin"]=="new_posthoc_keyframe_Qwen_diagnostic"
                assert sha(evidence["source_image"])==evidence["source_image_sha256"]==sha(inputpath)
                assert sha(evidence["source_episode_result"])==evidence["source_episode_sha256"]
                original=read(evidence["source_episode_result"])
                assert original["perception"]=="oracle" and original["success"]
                call=evidence["call"]
            assert item["raw"]==call["raw"] and item["prompt"]==call["text"]
            parsed,_=Perception.parse_pixel(call,max(item["image_size"]))
            assert parsed.tolist()==item["point"]
            origins.append({"task":task,"title":item["title"],"used_in_control":item["used_in_control"],"raw_reply_matches_point":True,"image_sha256_valid":True})
    for f in read(OUT/"cover/display_copy_manifest.json")["files"]:
        assert sha(f["source"])==sha(OUT/f["display_copy"])==f["sha256"]
    episodes=[]
    for task in EXPECTED:
        branches=("final_visual","final_holdout1","final_holdout2") if task=="cover" else ("final_seed0","final_seed1")
        current=[]
        for b in branches:current.extend(read(OUT/task/b/"result.json")["episodes"])
        assert summary["tasks"][task]["total"]==len(current)
        assert summary["tasks"][task]["success_count"]==sum(e["success"] for e in current)
        episodes.extend(current)
    transitions=[s for e in episodes for s in e["steps"]]
    assert len(episodes)==summary["total"]==17 and all(e["success"] for e in episodes)
    assert len(transitions)==summary["action_transitions"]==113 and all(s["strict_CP_transition_match"] for s in transitions)
    assert summary["camera_gallery"]["diagnostics_included_in_execution_scores"] is False
    result={"passed":True,"validation_scope":"presentation_integrity_not_new_physical_trials","scene_order":doc.scenes,"image_groups":EXPECTED,
            "closed_loop_images":10,"supplementary_diagnostic_images":10,"original_executions":17,"original_transitions":113,
            "preserved_original_sources":True,"cover_copy_hashes_match":True,"local_links_checked":len(doc.links),"broken_links":missing,"image_evidence":origins}
    (OUT/"presentation_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="image_evidence"},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
