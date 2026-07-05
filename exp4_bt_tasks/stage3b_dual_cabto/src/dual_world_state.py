"""dual_world_state —— 双臂任务的符号谓词状态（MuJoCo 真值路）
================================================================
为 stage3b_dual_cabto 的三个双臂任务（handover / pour / storage）提供 ground-truth
谓词计算，作为 CABTO 闭环的失败判定权威（gt_ok）。VLM 视觉校验是另一路对比记录。

设计要点
--------
三个任务各有一套语义谓词，用一个 task 标签切换要计算/校验哪一组，从而在同一
compute_state / check_effect 接口下服务不同场景。物体真值全部来自 stage2 的
Exp4Env（get_object_pose / get_body_pos），双臂即直接读真值，无需额外定位。

谓词（按任务）
--------------
handover:
  - held_L(box) / held_R(box) : 盒被左/右臂持有（weld 激活或指间闭合）
  - handed_over(box)          : 盒已从左臂交到右臂（右手持有过 → 记忆位）
  - at_place(box)             : 盒落在 -y 目标位、贴桌
pour:
  - held(can, arm)            : 罐被指定臂持有
  - poured(can)               : 罐已完成倾倒动作（tipped 记忆位）
  - in_cup(ball, cup)         : 球落入杯内（xy 在杯口内、低于杯沿、未掉地）
storage:
  - in_carton(item)           : item 落在 carton 内腔
  - on_shelf(carton)          : carton 被搬到 shelf 台面
  - held_R(carton)            : carton 被右臂持有

接口
----
- set_task(task) / set_meta(**kw)  : 切换任务与阈值元数据
- compute_state(env) -> dict[str, bool]
- check_effect(env, add, del_, state_before) -> (ok, violations)
"""
from __future__ import annotations

import numpy as np

import _bridge as B
import scene_pour_simple as sp_simple

sh = B.scene_handover
sp = sp_simple                       # pour 用简化场景（CUP 常量与坐标一致）
ss = B.scene_storage

TABLE_TOP_Z = B.scene_dual_common.TABLE_H  # 0.40

# ---- 当前任务上下文 ----
_TASK = "handover"
# 记忆位：某些谓词是"过程已发生"（如交接、倾倒）而非纯几何瞬时状态，
# 由 codegen 在执行到相应原语后置位，compute_state 读取。
_FLAGS: dict = {}


def set_task(task: str):
    global _TASK, _FLAGS
    _TASK = task
    _FLAGS = {}


def set_flag(name: str, val: bool = True):
    _FLAGS[name] = bool(val)


def get_flag(name: str) -> bool:
    return bool(_FLAGS.get(name, False))


# ===================== 通用真值读取 =====================
def _obj_xyz(env, name):
    return np.asarray(env.get_object_pose(name)[0], dtype=float)


def _body_xyz(env, name):
    return np.asarray(env.get_body_pos(name), dtype=float)


def _held_by(env, arm, obj) -> bool:
    """obj 是否被 arm 持有：该臂对该物体的 weld 激活，或夹爪闭合且物体在指间。"""
    try:
        return bool(arm.is_holding(obj))
    except Exception:
        return False


# ===================== handover 谓词 =====================
BOX_HALF_Z = sh.BOX_HALF[2]
PLACE_TOL = 0.06
PLACE0 = sh.PLACE_XYS[0]  # 简化：只交接 box0 到第一个 -y 目标位


def held_L(env, box) -> bool:
    return _held_by(env, env.arms[0], box)


def held_R(env, box) -> bool:
    return _held_by(env, env.arms[1], box)


def handed_over(env, box) -> bool:
    """已完成交接：右臂现在或曾经持有过该盒（记忆位）。"""
    return get_flag(f"handed:{box}") or held_R(env, box)


def at_place(env, box, place_xy=None) -> bool:
    p = _obj_xyz(env, box)
    px, py = place_xy if place_xy is not None else PLACE0
    off = float(np.linalg.norm(p[:2] - np.array([px, py])))
    on_table = abs(p[2] - (TABLE_TOP_Z + BOX_HALF_Z)) < 0.05
    on_right = p[1] < 0.02
    return bool(off < PLACE_TOL and on_table and on_right)


# ===================== pour 谓词 =====================
def held(env, can, arm_suffix) -> bool:
    arm = env.arms[0] if arm_suffix == "_L" else env.arms[1]
    return _held_by(env, arm, can)


def poured(env, can) -> bool:
    """罐已完成倾倒（tipped 记忆位）。"""
    return get_flag(f"poured:{can}")


def in_cup(env, ball, cup) -> bool:
    bp = _obj_xyz(env, ball)
    cup_xy = _body_xyz(env, cup)[:2]
    xy_off = float(np.linalg.norm(bp[:2] - cup_xy))
    return bool(xy_off < sp.CUP_R_IN + 0.012
                and bp[2] < sp.CUP_TOP + 0.02
                and bp[2] > TABLE_TOP_Z - 0.01)


# ===================== storage 谓词 =====================
def in_carton(env, item) -> bool:
    ip = _obj_xyz(env, item)
    cx, cy = _body_xyz(env, "carton")[:2] if _has_body(env, "carton") else ss.CARTON_XY
    inx, iny = ss.CARTON_IN
    dx = abs(ip[0] - cx)
    dy = abs(ip[1] - cy)
    z_ok = ip[2] < ss.CARTON_TOP_Z + 0.02
    return bool(dx < inx and dy < iny and z_ok)


def on_shelf(env, carton) -> bool:
    cp = _obj_xyz(env, carton)
    sx, sy = ss.SHELF_XY
    deck = ss.SHELF_DECK_Z
    # 收紧 y 容差，避免初始 carton 位 (~-0.11) 落入宽货架范围造成误报；
    # 要求 carton 中心真正靠近 shelf 中心 (~-0.20)。
    return bool(abs(cp[0] - sx) < 0.08
                and abs(cp[1] - sy) < 0.05
                and abs(cp[2] - (deck + 0.002)) < 0.06)


def _has_body(env, name) -> bool:
    try:
        env.get_body_pos(name)
        return True
    except Exception:
        return False


# ===================== compute_state / check_effect =====================
def compute_state(env) -> dict:
    st = {}
    if _TASK == "handover":
        box = "box0"
        st[f"held_L({box})"] = held_L(env, box)
        st[f"held_R({box})"] = held_R(env, box)
        st[f"handed_over({box})"] = handed_over(env, box)
        st[f"at_place({box})"] = at_place(env, box)
        # 与 planner ADD/DEL 命名一致的谓词别名（held(obj,arm)/holding_nothing(arm)）
        st[f"held({box},_L)"] = held_L(env, box)
        st[f"held({box},_R)"] = held_R(env, box)
        st["holding_nothing(_L)"] = not held_L(env, box)
        st["holding_nothing(_R)"] = not held_R(env, box)
    elif _TASK == "pour":
        st["held(canL,_L)"] = held(env, "canL", "_L")
        st["poured(canL)"] = poured(env, "canL")
        st["in_cup(ballL,cupR)"] = in_cup(env, "ballL", "cupR")
        st["holding_nothing(_L)"] = not held(env, "canL", "_L")
    elif _TASK == "storage":
        st["in_carton(item_g1)"] = in_carton(env, "item_g1")
        st["held_R(carton)"] = _held_by(env, env.arms[1], "carton")
        st["on_shelf(carton)"] = on_shelf(env, "carton")
        st["holding_nothing(_L)"] = not _held_by(env, env.arms[0], "item_g1")
        st["holding_nothing(_R)"] = not _held_by(env, env.arms[1], "carton")
    return st


def check_effect(env, add, del_, state_before) -> tuple:
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
    import numpy as np
    arms = [("_L", -np.pi / 2), ("_R", np.pi / 2)]
    for task, xml, objs in [
        ("handover", sh.build_scene_xml(), [f"box{i}" for i in range(sh.N_BOX)]),
        ("pour", sp.build_scene_xml(), ["canL", "ballL"]),
        ("storage", ss.build_scene_xml(), ["item_g1", "item_g2", "carton"]),
    ]:
        set_task(task)
        env = B.Exp4Env(xml, arms=arms, obj_names=objs, render=False, img_size=256)
        env.reset()
        st = compute_state(env)
        print(f"[{task}] init state:")
        for k, v in st.items():
            print(f"    {k} = {v}")
