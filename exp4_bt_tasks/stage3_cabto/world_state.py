"""world_state —— 符号谓词状态（MuJoCo 真值路）
==================================================
从 BlocksEnv 的 MuJoCo 真值计算符号谓词，作为 CABTO stage3 闭环的 **ground-truth
校验路**（VLM 视觉校验是另一路）。沿用 exp2 谓词语义并扩展堆叠相关谓词。

谓词集
------
- on_table(obj)        : 物体底面贴近桌面 z≈TABLE_TOP_Z 且未被夹爪持有。
- in_gripper(obj)      : 物体被其 weld 激活，或在指间且夹爪闭合。
- lifted(obj)          : 物体 z 高于桌面 + 阈值。
- stacked_on(a, b)     : a 底面≈b 顶面，且 a、b 的 xy 在一个块边长内对齐。
- clear(obj)           : obj 顶面上没有别的物体。
- holding_nothing()    : 没有任何物体 in_gripper。

接口
----
- compute_state(env) -> dict[str, bool]
- check_effect(env, add, del_, state_before) -> (ok: bool, violations: list[str])
"""

from __future__ import annotations

import numpy as np

from blocks_env import BLOCK_NAMES, TABLE_TOP_Z, BLOCK_HALF

BLOCK_FULL = 2 * BLOCK_HALF        # 0.05 全高

# 阈值
ON_TABLE_Z_TOL = 0.012             # 底面距桌面容差
LIFT_THRESH = 0.04                 # 高于桌面多少算 lifted
STACK_Z_TOL = 0.018                # a 底面与 b 顶面对齐容差
STACK_XY_TOL = BLOCK_HALF + 0.005  # xy 对齐容差（约一个半块边）
CLEAR_XY_TOL = BLOCK_HALF + 0.005


def _z(env, name):
    return float(env.d.qpos[env.obj_qadr[name] + 2])


def _xy(env, name):
    b = env.obj_qadr[name]
    return env.d.qpos[b:b + 2].copy()


def on_table(env, obj) -> bool:
    z = _z(env, obj)
    bottom = z - BLOCK_HALF
    near_table = abs(bottom - TABLE_TOP_Z) < ON_TABLE_Z_TOL
    return bool(near_table and not in_gripper(env, obj))


def in_gripper(env, obj) -> bool:
    wid = env.obj_weld[obj]
    welded = (wid >= 0 and env.d.eq_active[wid] == 1)
    closed = env._grip_cmd < 0.5
    between = env._obj_between_fingers(obj)
    return bool(welded or (closed and between))


def lifted(env, obj) -> bool:
    return bool(_z(env, obj) - BLOCK_HALF > TABLE_TOP_Z + LIFT_THRESH)


def stacked_on(env, a, b) -> bool:
    if a == b:
        return False
    za, zb = _z(env, a), _z(env, b)
    a_bottom = za - BLOCK_HALF
    b_top = zb + BLOCK_HALF
    z_ok = abs(a_bottom - b_top) < STACK_Z_TOL
    xy_ok = float(np.linalg.norm(_xy(env, a) - _xy(env, b))) < STACK_XY_TOL
    # a 必须在 b 之上
    above = za > zb
    return bool(z_ok and xy_ok and above)


def clear(env, obj) -> bool:
    """obj 顶面上没有别的物体（即没有别的块 stacked_on obj）。"""
    for other in BLOCK_NAMES:
        if other == obj:
            continue
        if stacked_on(env, other, obj):
            return False
    return True


def holding_nothing(env) -> bool:
    return not any(in_gripper(env, o) for o in BLOCK_NAMES)


def compute_state(env) -> dict:
    """返回所有谓词真值。键形如 'on_table(green_block)' / 'stacked_on(green_block,yellow_block)'。"""
    state = {}
    for o in BLOCK_NAMES:
        state[f"on_table({o})"] = on_table(env, o)
        state[f"in_gripper({o})"] = in_gripper(env, o)
        state[f"lifted({o})"] = lifted(env, o)
        state[f"clear({o})"] = clear(env, o)
    for a in BLOCK_NAMES:
        for b in BLOCK_NAMES:
            if a == b:
                continue
            state[f"stacked_on({a},{b})"] = stacked_on(env, a, b)
    state["holding_nothing()"] = holding_nothing(env)
    return state


def check_effect(env, add, del_, state_before) -> tuple:
    """执行动作后重算 state，检查 add 谓词都变 True、del_ 谓词都变 False。
    返回 (ok, violations)。violations 列出未满足项。"""
    after = compute_state(env)
    violations = []
    for p in (add or []):
        if not after.get(p, False):
            violations.append(f"ADD 未满足: {p} 仍为 False")
    for p in (del_ or []):
        if after.get(p, False):
            violations.append(f"DEL 未满足: {p} 仍为 True")
    return (len(violations) == 0), violations


if __name__ == "__main__":
    from blocks_env import BlocksEnv
    env = BlocksEnv(render=False)
    env.reset()
    st = compute_state(env)
    for k in sorted(st):
        if st[k]:
            print("  TRUE ", k)
    print("holding_nothing:", st["holding_nothing()"])
