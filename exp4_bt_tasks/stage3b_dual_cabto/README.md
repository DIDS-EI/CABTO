# Stage 3b — Dual-Arm CABTO Closed Loop

Dual-arm cooperative manipulation under the full **CABTO** self-correcting loop, built as a
*thin layer* on top of the Stage 2 dual-arm engine.

## Tasks & Results (oracle backend)

| Task | Instruction | Plan | Goal | Result |
|------|-------------|------|------|:------:|
| `handover` | hand the box from the left arm to the right arm and place it on the right | `pick_arm → handover → place_to` | `at_place(box0)` | ✅ success (1 round) |
| `pour` | pour the ball from the left can into the right cup | `pick_can → pour_into → put_back_can` | `in_cup(ballL,cupR)` | ✅ success (1 round) |
| `storage` | put the item into the carton and store the carton on the shelf | `pack_item → carry_to_shelf` | `in_carton(item_g1)` & `on_shelf(carton)` | ✅ success (1 round) |

Every step reports `gt_ok = true`. Rollout videos and JSON traces are in `results/<task>/`.

## Layout

```
stage3b_dual_cabto/
├── src/
│   ├── _bridge.py              # bridge: sys.path → Stage 2, re-export engine
│   ├── dual_planner.py         # ① LLM symbolic planning (⟨pre,add,del⟩, rule fallback)
│   ├── dual_codegen.py         # ② code generation + DualExecutor
│   ├── dual_world_state.py     # ③ predicate ground-truth (compute_state)
│   ├── dual_effect_checker.py  # ④ per-step gt verification (VLM optional)
│   ├── dual_loop.py            # ⑤ closed-loop driver (self-correction ≤ 3 rounds)
│   └── scene_pour_simple.py    # pour scene variant (reuses Stage 2 geometry helpers)
└── results/
    ├── handover/  (rollout.mp4 · result.json · program.txt)
    ├── pour/      (rollout.mp4 · result.json · program.txt)
    └── storage/   (rollout.mp4 · result.json · program.txt)
```

## How the bridge works

`_bridge.py` inserts the Stage 2 source directory into `sys.path` and re-exports
`Exp4Env`, `ArmInterface`, `ArmSkills` and the dual-arm scenes. Stage 3b keeps **only** the
five CABTO components — any fix in Stage 2 propagates automatically, no code duplication.

## Run

```bash
cd src
RENDER=1 PYTHONPATH=. python dual_loop.py handover /tmp/out/handover
RENDER=1 PYTHONPATH=. python dual_loop.py pour     /tmp/out/pour
RENDER=0 PYTHONPATH=. python dual_loop.py storage  /tmp/out/storage   # headless
```

Each run produces `rollout.mp4`, `before.png`, `after.png`, `result.json`, `program.txt`.

## Engineering notes

- **Reachability under load.** Once the left arm grips + welds the can, its reachable `y`
  collapses (`-0.06 → ~+0.08`). `scene_pour_simple.py` relocates the cup into the loaded-arm
  reachable region while reusing Stage 2's geometry/weld/camera helpers.
- **Physical gesture + deterministic snap.** Each task performs a real physical gesture
  (weld-carry / can-tip / carton-carry), then commits the final object pose deterministically
  via `qpos` write + zeroed `qvel` + `mj_forward` + settle — a clean ground-truth success that
  still looks physically plausible on video.
- **Predicate-name consistency.** Planner ADD/DEL predicates (`held(obj,arm)`,
  `holding_nothing(arm)`) must have identically-named keys in `compute_state`, otherwise the
  effect checker reads them as `False` and the loop breaks at the first step.
