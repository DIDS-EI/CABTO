# CABTO: Context-Aware Behavior Tree Grounding for Robot Manipulation

**AAAI 2026** | [[PDF]](https://arxiv.org/abs/2603.16809) | [Project Page](#)

![Python Version](images/python310.svg)
![GitHub license](images/license.svg)

CABTO is a framework that combines LLM-based behavior library generation with formal behavior tree planning for robot manipulation. Given a task description (BDDL), CABTO uses an LLM to generate a context-aware action/condition library, then applies the HOBTEA/OBTEA planning algorithm to synthesize a minimum-cost, executable behavior tree.

---

## Overview

<a href="images/framework.pdf"><img src="images/framework.png" width="800"/></a>

```
Task Description (BDDL)
        │
        ▼
┌───────────────────────┐
│  LLM (GPT-4o)         │  ← context-aware prompt with task, objects, initial state
│  Behavior Library     │  → Action classes (OGAction) + Condition classes (OGCondition)
│  Generation           │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  BT Planning          │  ← HOBTEA / OBTEA algorithm
│  (btgym)              │  → Minimum-cost Behavior Tree
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Execution            │  ← NVIDIA Isaac Sim (OmniGibson) or Real Franka Arms
│  & Validation         │
└───────────────────────┘
```

---

## Project Structure

```
CABTO/
├── btgym/                        # Core BT planning & execution framework
│   ├── algos/
│   │   ├── bt_planning/          # Planning algorithms (HOBTEA, OBTEA, BFS, ...)
│   │   │   ├── HOBTEA.py         # Hierarchical Optimal BT Expansion Algorithm (main)
│   │   │   ├── OBTEA.py          # Optimal BT Expansion Algorithm
│   │   │   ├── BTPAlgo.py        # Abstract base class for all planning algorithms
│   │   │   ├── behaviour_tree.py # Intermediate BT representation used during planning
│   │   │   ├── Action.py         # Planning-level action representation
│   │   │   └── main_interface.py # BTExpInterface: unified planner entry point
│   │   └── llm_client/           # LLM API clients (GPT-4, GPT-3.5, ERNIE)
│   ├── behavior_tree/
│   │   ├── base_nodes/           # BT node base classes (Action, Condition, Sequence, Selector)
│   │   ├── behavior_libs/        # Dynamic behavior library loader (ExecBehaviorLibrary)
│   │   ├── behavior_trees/       # BehaviorTree executor (py_trees-based)
│   │   └── btml/                 # BTML language compiler (ANTLR4)
│   ├── core/                     # OmniGibson simulator interface & Tiago controller
│   └── assets/                   # Activity definitions (900+ BDDL tasks), scene lists
│
├── dexrl/                        # Simulation environments & Isaac Sim extension
│   ├── envs/                     # Gymnasium-compatible RL environments
│   ├── sim/                      # Simulation scenarios (Franka, dual-arm, IK)
│   └── sim_extension/            # Isaac Sim Omni extension (UI + scenario management)
│
├── exps_bt_learning/             # Experiment scripts & task definitions
│   ├── a_exp1_llm_bt/            # Experiment 1: LLM-BT pipeline (success rate)
│   │   └── main_llm_bt.py        # Entry point
│   ├── a_exp1_llm_bt_tasks/      # Task definitions (task1–task7)
│   │   ├── _base/                # OGAction / OGCondition base classes
│   │   ├── task1/                # PlaceApple
│   │   ├── task2/                # ActivateLights
│   │   ├── task3/                # PutInDrawer
│   │   ├── task4/                # HomeRearrangement
│   │   ├── task5/                # MealPreparation
│   │   ├── task6/                # OpenContainer & PlaceItem
│   │   └── task7/                # (extended task)
│   ├── llm_generate_lib_func.py  # LLM → behavior library generation
│   ├── validate_bt_fun.py        # BT planning & execution validation
│   ├── tools.py                  # BDDL parser, prompt builder, code extractor
│   └── prompt_generate_libs.txt  # LLM prompt template
│
├── sim_extension/                # Isaac Sim extension module (loaded via --ext-folder)
│   ├── ext_sim.py                # IExt entry point
│   ├── ext_scenarios/            # Scenario wrappers for Isaac Sim
│   └── ui_builder.py             # Omni UI panels
│
├── global_config.ini             # System configuration (paths, hardware IPs, API key)
├── global_config.py              # Config loader
└── launch_sim.sh                 # Isaac Sim launcher
```

---

## Requirements

- **OS**: Ubuntu 20.04 / 22.04
- **GPU**: NVIDIA GPU with CUDA support
- **Python**: 3.10+ (via Isaac Sim's bundled Python)
- **Isaac Sim**: 4.2
- **OmniGibson**: compatible version

Key Python dependencies:
```
py_trees
openai
antlr4-python3-runtime
torch
numpy
gymnasium
omni.*        # bundled with Isaac Sim
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Caiyishuai/CABTO.git
cd CABTO

# 2. Install system dependencies
bash install/install.sh

# 3. Configure paths and API key
cp global_config.ini global_config.ini.bak
# Edit global_config.ini (see Configuration section)
```

---

## Configuration

Edit `global_config.ini` before running:

```ini
[simulation]
isaacsim_path = /path/to/IsaacSim4.2       # Isaac Sim installation directory
data_path     = /path/to/OmniBT-Data/      # OmniGibson scene data
model_path    = /path/to/models/           # Pretrained model checkpoints

[real]
left_arm_ip    = 192.168.100.100           # Left Franka arm IP (real hardware only)
right_arm_ip   = 192.168.100.101           # Right Franka arm IP (real hardware only)
left_hand_port = /dev/ttyUSB0
right_hand_port= /dev/ttyUSB1
top_camera_id  = 102422076737              # Intel RealSense camera serial numbers
three_camera_id= 050122072371
left_camera_id = 230322272965
right_camera_id= 230322273715

[api]
api_key  = sk-...                          # OpenAI-compatible API key
base_url = https://api.openai.com/v1       # API endpoint (or compatible proxy)
```

> **Security**: Do not commit `global_config.ini` with real API keys or hardware addresses to a public repository. Add it to `.gitignore` if needed.

---

## Running Experiments

### Experiment 1: LLM-BT Success Rate Evaluation

```bash
cd exps_bt_learning/a_exp1_llm_bt

# Configure in main_llm_bt.py:
#   task_id         = 1~5   (which task to evaluate)
#   total_try_times = 10    (number of independent trials)
#   model           = "gpt-4o"

python main_llm_bt.py
```

Results are saved to `exps_bt_learning/results/exp1_task{N}_success_rate_*.csv`.

Each trial executes the full pipeline:
1. **Generate** — LLM produces `exec_lib/Action/*.py` and `exec_lib/Condition/*.py`
2. **Plan** — OBTEA synthesizes a behavior tree from the generated library
3. **Execute** — The BT is simulated; outcome is recorded
4. **Record** — success/fail, action count, expanded node count, planning time

### Available Tasks

| ID | Name              | Goal |
|----|-------------------|------|
| 1  | PlaceApple        | `On(apple, coffeetable)` |
| 2  | ActivateLights    | `ToggledOn(light1) & ToggledOff(light2)` |
| 3  | PutInDrawer       | `In(pen, cabinet)` |
| 4  | HomeRearrangement | `On(pen,coffeetable) & IsClose(cabinet) & On(apple,coffeetable)` |
| 5  | MealPreparation   | `IsClose(oven) & ToggledOn(oven) & On(apple,coffeetable) & In(chickenleg,oven)` |

### Launching Isaac Sim

```bash
# Standard Isaac Sim (no extension)
./launch_sim.sh

# With dexrl extension (GUI with scenario controls)
./launch_sim.sh -e
```

---

## Core Concepts

### Behavior Library

The behavior library is a set of Python classes generated by the LLM for a specific task context. It is written to `exec_lib/Action/` and `exec_lib/Condition/` on disk, then loaded dynamically by `ExecBehaviorLibrary`.

- **Action** classes inherit `OGAction` and define `pre`, `add`, `del_set` (preconditions, add-effects, delete-effects)
- **Condition** classes inherit `OGCondition` and define their predicate evaluation

Example (LLM-generated):
```python
class PickUp(OGAction):
    num_args = 1
    valid_args = [{"apple", "pen"}]

    @classmethod
    def get_info(cls, obj):
        return {
            "pre":     {f"IsHandEmpty()", f"Near({obj})"},
            "add":     {f"Holding({obj})"},
            "del_set": {f"IsHandEmpty()"},
        }
```

### Planning Algorithms

| Algorithm | Description |
|-----------|-------------|
| `hobtea`  | Hierarchical Optimal BT Expansion — primary algorithm in the paper |
| `obtea`   | Optimal BT Expansion (non-hierarchical) |
| `bfs`     | Breadth-First Search baseline |
| `weak`    | Reactive/greedy planning (no lookahead) |
| `hbtp`    | Hierarchical BT Planning variant |

### BTML Format

Behavior trees are serialized in BTML (Behavior Tree Markup Language), compiled with an ANTLR4 grammar:

```
-> (Sequence)
    ? (Selector)
        Holding(apple)
        -> (Sequence)
            Near(apple)
            PickUp(apple)
    On(apple, coffeetable)
    PutOn(apple, coffeetable)
```

---

## Key Files Reference

| File | Role |
|------|------|
| `exps_bt_learning/a_exp1_llm_bt/main_llm_bt.py` | Main experiment entry point |
| `exps_bt_learning/llm_generate_lib_func.py` | LLM behavior library generation |
| `exps_bt_learning/validate_bt_fun.py` | BT planning + execution validation |
| `btgym/algos/bt_planning/HOBTEA.py` | Core planning algorithm |
| `btgym/algos/bt_planning/main_interface.py` | BTExpInterface (unified planner API) |
| `btgym/behavior_tree/behavior_libs/ExecBehaviorLibrary.py` | Dynamic library loader |
| `global_config.ini` | System configuration |
| `launch_sim.sh` | Isaac Sim launcher |

---

## Citation

```bibtex
@inproceedings{cabto2026,
  title     = {CABTO: Context-Aware Behavior Tree Grounding for Robot Manipulation},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```
