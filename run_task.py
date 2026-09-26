"""Run one latest direct five-task reference from the self-contained package."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
P=ROOT/"project"
COMMANDS={
    "cover":[P/"run_single_arm.py","cover"],
    "blocks":[P/"run_single_arm.py","blocks"],
    "pour":[P/"probe_smooth_held_basin_reference.py"],
    "handover":[P/"run_handleless_handover.py"],
    "storage":[P/"probe_dual_lift_shelf.py"],
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("task",choices=COMMANDS);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seed",type=int,default=0);ap.add_argument("--no-render",action="store_true");a=ap.parse_args()
    cmd=[sys.executable,"-B",str(COMMANDS[a.task][0]),*[str(x) for x in COMMANDS[a.task][1:]],"--output",str(a.output.resolve()),"--seed",str(a.seed)]
    if a.no_render and a.task in ("cover","blocks","handover"):cmd.append("--no-render")
    raise SystemExit(subprocess.run(cmd,cwd=P).returncode)

if __name__=="__main__":main()
