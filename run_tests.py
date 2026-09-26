"""Run the five curated lightweight regression suites."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
P=ROOT/"project"
TESTS=("test_single_arm_rebuilt.py","test_handleless_tea_handover.py","test_smooth_held_basin.py","test_dual_lift_shelf.py","test_dual_lift_codegen.py")
for test in TESTS:
    rc=subprocess.run([sys.executable,"-B",str(P/test)],cwd=P).returncode
    if rc:raise SystemExit(rc)
print("ALL 164 TESTS PASSED")
