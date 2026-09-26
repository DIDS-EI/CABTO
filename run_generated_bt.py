"""Replay latest available generated programs through formal BT execution."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
P=ROOT/"project"

def command(task,out):
    py=[sys.executable,"-B"]
    if task in ("cover","blocks"):
        return py+[str(P/"probe_latest_single_arm_generated_bt.py"),task,
                   "--config",str(P/"scenes"/("cover_kitchen_sort_v2.json" if task=="cover" else "blocks_rebuilt_v1.json")),
                   "--programs",str(ROOT/f"tasks/{task}/generated_bt/generated_programs.json"),
                   "--output",str(out)]
    if task=="pour":
        return py+[str(P/"probe_smooth_held_basin_codegen.py"),"--output",str(out),"--reuse",str(ROOT/"tasks/pour/generated_bt/generated_programs.json"),"--episodes","1"]
    if task=="handover":
        return py+[str(P/"probe_handleless_generated_bt.py"),"--output",str(out)]
    if task=="storage":
        return py+[str(P/"probe_dual_lift_codegen.py"),"--output",str(out),"--reuse",str(ROOT/"tasks/storage/generated_bt/generated_programs.json"),"--episodes","1"]
    raise ValueError(task)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("task",choices=["cover","blocks","pour","handover","storage"]);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    raise SystemExit(subprocess.run(command(a.task,a.output.resolve()),cwd=P).returncode)

if __name__=="__main__":main()
