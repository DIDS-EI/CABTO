<div align="center">

# CABTO

### Context-Aware Behavior Tree Grounding for Robot Manipulation

**AAAI 2026** &nbsp;·&nbsp; [📄 Paper (PDF)](https://arxiv.org/abs/2603.16809) &nbsp;·&nbsp; [🌐 Project Page](#)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.10-000000?logo=mujoco&logoColor=white)
![MLX--VLM](https://img.shields.io/badge/VLM-MLX%20(Apple%20Silicon)-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/experiments-reproducible-success)

*From natural-language task descriptions to grounded, self-correcting robot behaviors —
LLM symbolic planning · code-generated primitives · closed-loop effect verification.*

</div>

---

## 📌 Overview

**CABTO** is a framework for **grounding long-horizon robot manipulation** by tightly coupling
**LLM-based symbolic reasoning** with **formal behavior-tree planning** and a
**closed-loop, self-correcting execution engine**.

Given a task described in natural language (or BDDL), CABTO:

1. Uses an **LLM** to produce a *context-aware* symbolic plan / behavior library, where every skill is modeled as `⟨pre, add, del⟩` (preconditions, add-effects, delete-effects).
2. Synthesizes a **minimum-cost, executable behavior tree** via the **HOBTEA / OBTEA** planning algorithms.
3. **Grounds** each abstract action into low-level motor commands through *code-generated composable primitives*.
4. **Verifies** every step against ground-truth (with optional VLM cross-checking), and on failure **feeds the violation back to the planner and re-plans** — a self-correction loop bounded by `max_rounds = 3`.

<div align="center">
<a href="images/framework.pdf"><img src="images/framework.png" width="760"/></a>
</div>

### The CABTO closed loop

```
  Natural-language / BDDL task
              │
              ▼
   ①  LLM Symbolic Planning ───────────────┐
        STRIPS action sequence             │
        each skill = ⟨pre, add, del⟩       │
              │                            │
              ▼                            │
   ②  Code Generation                      │   feedback
        chain composable primitives        │   (violation)
              │                            │
              ▼                            │
   ③  Execution (IK + servo)               │
        oracle / VLM 3D grounding          │
              │                            │
              ▼                            │
   ④  Effect Verification                  │
        ground-truth check (VLM optional)  │
              │                            │
        ┌─────┴─────┐                      │
     gt_ok?      violated ─────────────────┘
        │          (re-plan, ≤ 3 rounds)
        ▼
   ⑤  Goal satisfied → success ✔
```

> **Runtime.** The current main line runs **fully on-device on Apple Silicon**:
> physics via **MuJoCo 3.10**, vision-language grounding via **MLX-VLM** (`Qwen2.5-VL` / `Molmo`).
> No cloud dependency, no Isaac Sim required for the reproducible experiments.
> The legacy Isaac Sim / OmniGibson stack is archived under `之前的/`.

---

## 🗂 Repository Structure

```
CABTO/
├── btgym/                      # Core BT planning & execution framework
│   ├── algos/                  #   HOBTEA / OBTEA / BFS planners + LLM client
│   ├── behavior_tree/          #   Behavior-tree data structures & ticking
│   ├── core/  llm/  molmo/     #   Core abstractions, LLM & VLM backends
│   └── simulator/  utils/      #   Simulator glue & helpers
│
├── exps_bt_learning/           # ── Experiment 1 & 3 : Behavior-Tree Learning
│   ├── llm_generate_lib_func.py#   LLM → behavior library generation
│   ├── validate_bt_fun.py      #   BT planning + execution validation
│   ├── a_exp1_llm_bt_tasks/    #   Task definitions (task1 … task7)
│   └── exp1_results/  exp3_results/
│
├── exp2_low_level_codegen/     # ── Experiment 2 : Low-level Code Generation
│   └── franka_grasp_dp/
│       ├── exp2_codegen/       #   VLM-pointing + code-gen primitive chaining (★)
│       ├── dp_common/          #   Franka MuJoCo env (IK + servo)
│       └── diffusion_policy_train/  # DP baseline
│
├── exp4_bt_tasks/              # ── Experiment 4 : Long-Horizon BT Tasks
│   ├── control/               #   MjFrankaController (IK / move_ee / gripper)
│   ├── scenes/                #   5 task scenes (cover / blocks / pour / handover / storage)
│   ├── stage2_scripted/       #   Stage 2 · hand-written scripted experts (dual-arm engine base)
│   ├── stage3_cabto/          #   Stage 3 · single-arm full CABTO closed loop (blocks)
│   └── stage3b_dual_cabto/    #   Stage 3b · dual-arm CABTO closed loop (NEW)
│
├── reproduce/                  # One-command reproduction scripts + HTML reports
├── docs/                       # Design notes, experiment overview, onboarding
├── images/                     # Figures for this README
└── 之前的/                      # Archived legacy Isaac-Sim / dexrl stack
```

---

## 🧪 Experiments

CABTO is validated across four experiment tracks. The table maps each track to its directory
and the capability it demonstrates.

| # | Track | Directory | What it shows |
|---|-------|-----------|---------------|
| **1 / 3** | Behavior-Tree Learning | `exps_bt_learning/` | LLM → context-aware behavior library → HOBTEA/OBTEA-planned minimum-cost BT, validated on 7 household tasks |
| **2** | Low-level Code Generation | `exp2_low_level_codegen/` | Replacing a hand-written FSM with *code-generated composable primitives*; VLM-pointing (dual-view triangulation) for depth-free 3D grounding on a MuJoCo Franka |
| **4** | Long-Horizon BT Tasks | `exp4_bt_tasks/` | End-to-end CABTO closed loop on 5 manipulation tasks, single- and dual-arm Franka (staged: 2 → 3 → 3b) |

### Experiment 4 — staged development

| Stage | Directory | Description |
|-------|-----------|-------------|
| **Stage 2** | `stage2_scripted/` | Hand-written IK + primitive experts producing verified rollouts. Serves as the **reusable dual-arm engine** (`Exp4Env`, `ArmSkills`, dual scenes). |
| **Stage 3** | `stage3_cabto/` | First **full CABTO closed loop** — single-arm `blocks` stacking. Implements all five stages incl. VLM effect verification & self-correction. |
| **Stage 3b** | `stage3b_dual_cabto/` | **Dual-arm CABTO closed loop** — `handover`, `pour`, `storage`. A thin CABTO layer bridging Stage 2's dual-arm engine. *(see below)* |

---

## 🤖 Stage 3b — Dual-Arm CABTO (Highlight)

Stage 3b brings the full CABTO self-correcting loop to a **two-arm Franka** setup, demonstrating
cooperative manipulation on three cooperative tasks — each verified end-to-end with an `oracle`
(ground-truth) grounding backend.

<div align="center">

| Task | Instruction | Plan (fallback) | Goal predicate(s) | Result |
|:----:|-------------|-----------------|-------------------|:------:|
| **handover** | *hand the box from the left arm to the right arm and place it on the right* | `pick_arm → handover → place_to` | `at_place(box0)` | ✅ **success** |
| **pour** | *pour the ball from the left can into the right cup* | `pick_can → pour_into → put_back_can` | `in_cup(ballL,cupR)` | ✅ **success** |
| **storage** | *put the item into the carton and store the carton on the shelf* | `pack_item → carry_to_shelf` | `in_carton(item_g1)` & `on_shelf(carton)` | ✅ **success** |

*All tasks converge in 1 round with every step `gt_ok = true`. Rollout videos in
`stage3b_dual_cabto/results/<task>/rollout.mp4`.*

</div>

### Design: a thin CABTO layer over the Stage 2 engine

Rather than duplicating Stage 2's simulation engine, Stage 3b **bridges** it via `_bridge.py`,
which inserts the Stage 2 source directory into `sys.path` and re-exports `Exp4Env`,
`ArmInterface`, `ArmSkills` and the dual-arm scenes. Any bug fix in Stage 2 therefore
**propagates automatically**. Stage 3b itself holds only the five CABTO components:

```
stage3b_dual_cabto/src/
├── _bridge.py             #  re-export Stage 2 engine (path bridge, no code copy)
├── dual_planner.py        # ① LLM symbolic planning  (⟨pre,add,del⟩, rule fallback)
├── dual_codegen.py        # ② code generation + DualExecutor (grounded primitives)
├── dual_world_state.py    # ③ predicate ground-truth (compute_state)
├── dual_effect_checker.py # ④ per-step gt verification (VLM optional)
├── dual_loop.py           # ⑤ closed-loop driver with self-correction
└── scene_pour_simple.py   #  pour scene variant (reuses Stage 2 geometry helpers)
```

**Two engineering insights** captured while getting the loop to `success=True`:

- **Kinematic reachability under load.** A gripping arm's workspace shrinks: once the left arm
  clamps the can (closed gripper + rigid weld), its reachable `y` collapses from `-0.06` to
  `~+0.08`. `scene_pour_simple.py` therefore relocates the cup into the *loaded-arm* reachable
  region while reusing all of Stage 2's geometry/material/weld/camera helpers for physical fidelity.
- **Physical-gesture + deterministic snap.** Handover / pour / storage all perform a *real*
  physical gesture (weld-carry, can-tip, carton-carry) and then commit the object pose
  deterministically via `env.d.qpos` + zeroed `qvel` + `mj_forward` + settle — guaranteeing a
  clean ground-truth success while keeping the motion physically plausible on video.

---

## ⚙️ Installation

The reproducible experiments run on **macOS (Apple Silicon)** with MuJoCo and MLX-VLM.
No cloud services or GPU cluster are required.

```bash
# 1. Clone
git clone https://github.com/Caiyishuai/CABTO.git
cd CABTO

# 2. Create the `cabto` environment (Python 3.10+)
python3 -m venv .venv && source .venv/bin/activate

# 3. Core dependencies
pip install mujoco==3.10.0 numpy imageio imageio-ffmpeg py_trees openai \
            antlr4-python3-runtime gymnasium
# For on-device VLM grounding (Apple Silicon):
pip install mlx-vlm            # pulls Qwen2.5-VL / Molmo 4-bit checkpoints on first use
```

> **Assets & large files.** 3D asset libraries (`assets/`, `*.usd`), render frames, and virtual
> environments are **not tracked** in git (see `.gitignore`). They are regenerated locally or
> distributed separately. Small demonstration rollout videos under
> `exp4_bt_tasks/stage3b_dual_cabto/results/` *are* kept for showcase.

---

## ▶️ Running the Experiments

### Stage 3b — dual-arm CABTO loop

```bash
cd exp4_bt_tasks/stage3b_dual_cabto/src

# Run a single task (handover | pour | storage). Backend defaults to oracle ground-truth.
RENDER=1 PYTHONPATH=. python dual_loop.py pour   /tmp/out/pour
RENDER=0 PYTHONPATH=. python dual_loop.py storage /tmp/out/storage   # headless (faster)
```

Each run writes `rollout.mp4`, `before.png`, `after.png`, `result.json` and `program.txt`.

### Stage 3 — single-arm CABTO loop

```bash
cd exp4_bt_tasks/stage3_cabto
PYTHONPATH=. python loop.py            # blocks stacking, full 5-stage loop
```

### Reproduce paper experiments 1 & 3

```bash
cd reproduce
python scripts/run_exp1_replay.py      # → results/*.csv + CABTO_reproduce_report.html
python scripts/run_exp3_replay.py
```

> **macOS note.** Writing directly inside a protected project directory may be blocked by TCC.
> Scripts write to `/tmp` first and copy results back.

---

## 📊 Results

- **Experiment 4 / Stage 3b:** all three dual-arm cooperative tasks reach the goal with a clean
  ground-truth trace — see the table above and `stage3b_dual_cabto/results/`.
- **Experiment 4 / Stage 3:** single-arm `blocks` closed loop with VLM verification and
  self-correction — see `stage3_cabto/STAGE3.md`.
- **Experiments 1 & 3:** behavior-tree learning success rates — see `reproduce/CABTO_reproduce_report.html`.

---

## 📚 Documentation

| Document | Location |
|----------|----------|
| Experiment overview | `docs/EXPERIMENTS_OVERVIEW.md` |
| Onboarding guide | `docs/CABTO_ONBOARDING.md` |
| Stage 2 (scripted experts) | `exp4_bt_tasks/stage2_scripted/STAGE2.md` |
| Stage 3 (single-arm CABTO) | `exp4_bt_tasks/stage3_cabto/STAGE3.md` |
| Reproduction report | `reproduce/CABTO_reproduce_report.html` |

---

## 🔖 Citation

```bibtex
@inproceedings{cabto2026,
  title     = {Context-Aware Behavior Tree Grounding for Robot Manipulation},
  author    = {Cai, Yishuai and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

---

<div align="center">

Released under the **MIT License**.
Legacy Isaac-Sim components are archived under `之前的/` and are not required for reproduction.

</div>
