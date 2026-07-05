"""Bridge module for stage3b_dual_cabto.

We do NOT copy the stage2 dual-arm engine/skills/scenes. Instead we add the
stage2 source directory onto sys.path and import them directly, so any bug
fixes in stage2 propagate automatically. This keeps stage3b a thin CABTO
closed-loop layer (planner / codegen / world_state / effect_checker / loop)
on top of the already-validated stage2 dual-arm primitives.
"""
import os
import sys

# stage2 scripted dual-arm engine lives here (in the real CABTO repo).
STAGE2_DIR = "/Users/yishuaicai/mywork/CABTO/exp4_bt_tasks/stage2_scripted"

if STAGE2_DIR not in sys.path:
    sys.path.insert(0, STAGE2_DIR)

# Re-export the reusable stage2 building blocks.
from exp4_env import Exp4Env, ArmInterface  # noqa: E402
from arm_skills import ArmSkills  # noqa: E402
import scene_handover  # noqa: E402
import scene_pour  # noqa: E402
import scene_storage  # noqa: E402
import scene_dual_common  # noqa: E402

__all__ = [
    "Exp4Env",
    "ArmInterface",
    "ArmSkills",
    "scene_handover",
    "scene_pour",
    "scene_storage",
    "scene_dual_common",
    "STAGE2_DIR",
]
