"""Fast integrity check for the curated five-task package (no long simulation)."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TASKS=("cover","blocks","pour","handover","storage")
REQUIRED={
 "cover":["direct/rollout.mp4","direct/result.json","generated_bt/generated_programs.json","generated_bt/latest_scene_replay/result.json","generated_bt/latest_scene_replay/rollout.mp4","generated_bt/latest_scene_replay/tree.svg","report.html"],
 "blocks":["direct/rollout.mp4","direct/result.json","generated_bt/generated_programs.json","generated_bt/latest_scene_replay/result.json","generated_bt/latest_scene_replay/rollout.mp4","generated_bt/latest_scene_replay/tree.svg","report.html"],
 "pour":["direct/rollout.mp4","direct/result.json","generated_bt/generated_programs.json","generated_bt/rollout.mp4","generated_bt/tree.svg","report.html"],
 "handover":["direct/rollout.mp4","direct/result.json","formal_bt/tree.svg","generated_bt/source_generated_programs.json","generated_bt/replay/result.json","generated_bt/replay/rollout.mp4","generated_bt/replay/tree.svg","report.html"],
 "storage":["direct/rollout.mp4","direct/result.json","generated_bt/generated_programs.json","generated_bt/rollout.mp4","generated_bt/tree.svg","report.html"],
}

def success(task,path):
 d=json.loads(path.read_text())
 if task=="cover" and path.parent.name=="direct":return d.get("success") is True
 if task=="blocks" and path.parent.name=="direct":return d.get("success") is True
 if task=="pour" and path.parent.name=="direct":return d.get("success") is True
 if task=="handover":return d.get("success") is True
 if task=="storage" and path.parent.name=="direct":return d.get("success") is True
 return True

def main():
 missing=[];bad=[];files=[]
 for task,rels in REQUIRED.items():
  for rel in rels:
   p=ROOT/"tasks"/task/rel
   if not p.is_file():missing.append(str(p))
   else:
    files.append({"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
  rp=ROOT/"tasks"/task/"direct/result.json"
  if rp.is_file() and not success(task,rp):bad.append(str(rp))
  latest_replay=ROOT/"tasks"/task/"generated_bt/latest_scene_replay/result.json"
  if latest_replay.is_file():
   replay=json.loads(latest_replay.read_text())
   if replay.get("success") is not True:bad.append(str(latest_replay))
   config=ROOT/"project/scenes"/("cover_kitchen_sort_v2.json" if task=="cover" else "blocks_rebuilt_v1.json")
   if task in ("cover","blocks") and replay.get("config_sha256")!=hashlib.sha256(config.read_bytes()).hexdigest():bad.append(f"config hash mismatch: {latest_replay}")
 for p in [ROOT/"tests/results/unit_test_summary.json",ROOT/"tests/results/direct_smoke.json",ROOT/"tests/results/generated_bt_smoke.json",
           ROOT/"paper_reproduction/perception_pointing/evaluation_summary.json",
           ROOT/"paper_reproduction/perception_pointing/qwen3_5_0_8b_report.html",
           ROOT/"paper_reproduction/perception_pointing/franka_grasp_dp/exp2_codegen/results/eval_pick_qwen3_5_0_8b.json",
           ROOT/"paper_reproduction/perception_pointing/franka_grasp_dp/exp2_codegen/renders/pointing_overlay_qwen3_5_0_8b_rubber_ball.png",
           ROOT/"paper_reproduction/perception_pointing/franka_grasp_dp/exp2_codegen/renders/rollout_qwen3_5_0_8b_rubber_ball.mp4"]:
  if not p.is_file():missing.append(str(p))
 active_scenes=sorted(p.name for p in (ROOT/"project/scenes").glob("*.json"))
 expected_scenes=sorted(["cover_kitchen_sort_v2.json","blocks_rebuilt_v1.json","held_basin_pour_smooth_v2.json","handover_tea_box_v3.json","packing_dual_lift_shelf_storage_v3.json"])
 if active_scenes != expected_scenes:bad.append(f"active scenes mismatch: {active_scenes}")
 q08=ROOT/"paper_reproduction/perception_pointing/franka_grasp_dp/exp2_codegen/results/eval_pick_qwen3_5_0_8b.json"
 if q08.is_file():
  summary=json.loads(q08.read_text())["summary"]
  if summary.get("model_path")!="mlx-community/Qwen3.5-0.8B-4bit" or summary.get("n_total")!=20:
   bad.append(str(q08))
 report={"ok":not missing and not bad,"tasks":list(TASKS),"active_scenes":active_scenes,"missing":missing,"bad_results":bad,"checked_files":len(files),"files":files}
 (ROOT/"package_manifest.json").write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!="files"},indent=2))
 raise SystemExit(0 if report["ok"] else 1)
if __name__=="__main__":main()
