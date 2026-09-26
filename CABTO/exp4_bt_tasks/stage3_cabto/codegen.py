"""C. ② 代码生成：把符号动作翻成「生成的程序」串接原语
========================================================
对应 CABTO「代码生成串接低层原语」。每个 skill 对应一个被生成的程序函数，
内部用 Pointer 定位 + PrimitiveRunner 原语完成。

- gen_pick(runner, pointer, obj):
    locate_3d(obj) → approach → align → descend → grasp → lift
- gen_place_on(runner, pointer, obj, dst):
    locate_3d(dst) → move_above → descend_place → release → retreat → settle
- execute_step(env, runner, pointer, step) -> log：按 step["name"] 分派。
- render_program_text(step) -> str：返回该步「生成的代码」文本（可视化/文档用）。

定位 3D 点语义：locate_3d 返回目标物体「顶面附近」世界点。
- pick：用其 xy 作抓取 xy，顶面 z 算 grasp_z（抓上部）。
- place_on：用其 z 作 dst 顶面高度，xy 作目标中心。
"""

from __future__ import annotations

import numpy as np

from primitives import grasp_z_for_top
from vlm_pointer3d import obj_phrase


def gen_pick(runner, pointer, obj):
    log = {"skill": "pick", "obj": obj, "steps": []}
    # 1) 定位（VLM 打点→3D 或 oracle 真值）
    p = pointer.locate_3d(obj_phrase(obj), obj_name=obj)
    log["located_3d"] = [float(x) for x in p]
    top_z = float(p[2])
    grasp_z = grasp_z_for_top(top_z)
    xy = p[:2]

    # 2) approach → align → descend → grasp → lift
    runner.approach(p);                         log["steps"].append("approach")
    runner.align_xy(p);                         log["steps"].append("align_xy")
    runner.descend_to(grasp_z, lock_xy=xy);     log["steps"].append(f"descend_to z={grasp_z:.3f}")
    runner.grasp(xy);                           log["steps"].append("grasp")
    runner.record_grab_offset(obj)
    log["grab_dz"] = runner.grab_dz
    log["grab_dxy"] = [float(v) for v in runner.grab_dxy]
    runner.lift(xy=xy);                         log["steps"].append("lift")

    holding = runner.env.is_holding(obj)
    log["holding_after"] = bool(holding)
    return log


def gen_place_on(runner, pointer, obj, dst, oracle_refine=False):
    log = {"skill": "place_on", "obj": obj, "dst": dst, "steps": []}
    # 1) 定位 dst 顶面
    p = pointer.locate_3d(obj_phrase(dst), obj_name=dst)
    log["located_3d"] = [float(x) for x in p]
    dst_top_z = float(p[2])
    dst_xy = p[:2]

    # 2) move_above（补偿抓偏）→ 可选真值精修 → descend_place → release → retreat → settle
    tcp_xy = runner.move_above(p);              log["steps"].append("move_above")
    refine_obj = obj if oracle_refine else None
    tcp_xy = runner.refine_above(dst_xy, obj_name=refine_obj); log["steps"].append("refine_above")
    runner.descend_place(dst_top_z, tcp_xy);    log["steps"].append("descend_place")
    runner.place_release();                     log["steps"].append("place_release")
    runner.retreat();                           log["steps"].append("retreat")
    runner.settle();                            log["steps"].append("settle")
    return log


def execute_step(env, runner, pointer, step, oracle_refine=False):
    """按 step["name"] 分派到 gen_ 函数，传 step["args"]。返回执行 log。"""
    name = step["name"]
    args = step.get("args", {})
    if name == "pick":
        return gen_pick(runner, pointer, args["obj"])
    if name == "place_on":
        return gen_place_on(runner, pointer, args["obj"], args["dst"],
                            oracle_refine=oracle_refine)
    return {"skill": name, "error": "unknown skill", "steps": []}


def render_program_text(step):
    """返回该 step「生成的代码」文本（体现 codegen 语义，给可视化/文档展示）。"""
    name = step["name"]
    a = step.get("args", {})
    if name == "pick":
        o = a["obj"]
        return (f"def pick_{o}(runner, pointer):\n"
                f"    p = pointer.locate_3d('{obj_phrase(o)}')   # VLM 打点→3D\n"
                f"    grasp_z = p[2] - 0.30*BLOCK_FULL\n"
                f"    runner.approach(p)\n"
                f"    runner.align_xy(p)\n"
                f"    runner.descend_to(grasp_z, lock_xy=p[:2])\n"
                f"    runner.grasp(p[:2])\n"
                f"    runner.lift()\n")
    if name == "place_on":
        o, d = a["obj"], a["dst"]
        return (f"def place_{o}_on_{d}(runner, pointer):\n"
                f"    p = pointer.locate_3d('{obj_phrase(d)}')   # 定位目标顶面\n"
                f"    tcp_xy = runner.move_above(p)\n"
                f"    runner.descend_place(p[2], tcp_xy)\n"
                f"    runner.place_release()\n"
                f"    runner.retreat()\n"
                f"    runner.settle()\n")
    return f"# (unknown skill: {name})\n"
