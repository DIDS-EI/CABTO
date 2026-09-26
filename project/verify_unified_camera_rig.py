"""Independent geometry, source integrity, image and page checks for camera rig."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import hashlib,json
import numpy as np
import mujoco
from PIL import Image
from core.unified_camera_rig import PROFILE,physics_comparison,project,intersect_plane

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"outputs/unified_cameras/selected_v3";WEB=ROOT/"outputs/remaining_cabto"

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Parser(HTMLParser):
    def __init__(self):super().__init__();self.ids=set();self.rigs=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if "id" in a:self.ids.add(a["id"])
        if "data-rig" in a:self.rigs.append(a["data-rig"])
        for k in ("src","href","poster"):
            if k in a:self.links.append(a[k])


def main():
    rows=[];frames=0;images=0;roundtrip=0;local_error=0
    for task in ("storage","pour","handover","cover","blocks"):
        out=BASE/task;d=read(out/"result.json");assert d["success"] and d["sources_unchanged"]
        assert d["new_cameras_used_for_control"] is False and d["perception_used_for_motion"]=="oracle_only_diagnostic_run"
        assert d["camera_rig"]["profile"]==PROFILE
        for p,h in d["source_hashes"].items():assert sha(p)==h,p
        assert sha(d["model_input"])==d["model_input_sha256"]
        assert sha(d["policies_input"])==d["policies_input_sha256"]
        original=mujoco.MjModel.from_xml_path(str(out/"scene_reference.xml"));model=mujoco.MjModel.from_xml_path(str(out/"scene_with_cameras.xml"))
        physics_comparison(original,model)
        cameras=d["camera_rig"]["cameras"];n=2 if task in ("cover","blocks") else 3;assert len(cameras)==n
        assert model.ncam-original.ncam==n
        for name in d["camera_rig"]["old_cameras_preserved"]:
            i=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_CAMERA,name);j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,name)
            for a in ("cam_pos","cam_quat","cam_fovy","cam_bodyid"):np.testing.assert_array_equal(getattr(original,a)[i],getattr(model,a)[j])
        camera_positions={c:[] for c in cameras}
        for c in d["captures"]:
            assert c["images_simultaneous_no_step"] and c["render_did_not_change_state"]
            for v in c["views"]:
                assert sha(v["image"])==v["image_sha256"]
                with Image.open(v["image"]) as image:assert image.size==(720,720)
                cal=v["calibration"];assert cal["time"]==c["simulation_time"]
                if v["camera"] in camera_positions:camera_positions[v["camera"]].append(np.array(cal["T_world_camera"]["position"]))
                if v["camera"].startswith("rig_wrist"):
                    reference=next(m["T_hand_camera"] for m in d["camera_rig"]["wrist_mounts"] if m["name"]==v["camera"])
                    for field in ("position","rotation"):
                        e=float(np.max(np.abs(np.array(cal["measured_T_body_camera"][field])-reference[field])));local_error=max(local_error,e);assert e<1e-9
                for target in c["evaluation_only_targets"]:
                    uv=project(cal,target["xyz"])
                    if uv is None:continue
                    z=target["xyz"][2]
                    try:e=float(np.linalg.norm(intersect_plane(cal,uv,z)-target["xyz"]))
                    except ValueError:continue
                    roundtrip=max(roundtrip,e);assert e<1e-8
                images+=1
            frames+=1
        for camera,positions in camera_positions.items():
            change=max(np.linalg.norm(p-positions[0]) for p in positions)
            if camera=="rig_global":assert change<1e-12
            else:assert change>.05
        assert all(s["effect_ok"] for s in d["steps"])
        rows.append({"task":task,"cameras_added":n,"control_compatibility_pass":True,"physics_arrays_unchanged":True,"camera_rendering_state_unchanged":True,
                     "wrist_moves_with_parent":True,"identical_profile":True,"captures":len(d["captures"])})
    points=read(ROOT/"outputs/unified_cameras/pointing_v3/results.json")["point_results"]
    assert len(points)==28 and all(p["used_by_robot"] is False for p in points)
    for p in points:
        assert sha(p["input_image"])==p["input_sha256"]
        assert sha(p["original_run"])==p["original_run_sha256"]
        assert p["call"]["image_files"]==[p["input_image"]]
    parser=Parser();parser.feed((WEB/"index.html").read_text());assert set(parser.rigs)=={"cover","blocks","pour","handover","storage"}
    broken=[]
    for url in parser.links:
        u=urlsplit(url)
        if u.scheme:continue
        if not u.path:
            if u.fragment not in parser.ids:broken.append(url)
        elif u.path!="unified_cameras/verification.json" and not (WEB/unquote(u.path)).is_file():broken.append(url)
    assert not broken,broken
    test=(ROOT/"outputs/unified_cameras/camera_tests.log").read_text();assert "Ran 27 tests" in test and "OK" in test and "FAILED" not in test
    for file,n in [("remaining_regression.log",36),("visual_regression.log",25),("generation_regression.log",8),("cover_regression.log",44)]:
        text=(ROOT/"outputs/unified_cameras"/file).read_text();assert f"Ran {n} tests" in text and "OK" in text and "FAILED" not in text
    assert "ALL 164 TESTS PASSED" in (ROOT/"outputs/unified_cameras/legacy_regression.log").read_text()
    result={"passed":True,"camera_profile":PROFILE,"scene_checks":rows,"frames":frames,"images":images,
            "max_project_roundtrip_m":roundtrip,"max_local_mount_error":local_error,"tests":27,"previous_regression_tests":277,"total_tests":304,"diagnostic_calls":len(points),
            "point_returned":sum(p["status"]=="point_returned_not_validated" for p in points),"abstained":sum(p["status"]=="model_abstained" for p in points),
            "model_points_not_executable":True,"linked_local_files":len(parser.links),"broken_links":broken,
            "scope":"camera installation and finite control-compatibility, NOT new vision-guided task success; reference-z errors evaluation-only"}
    (WEB/"unified_cameras/verification.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    (ROOT/"outputs/unified_cameras/verification.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ("scene_checks","camera_profile")},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
