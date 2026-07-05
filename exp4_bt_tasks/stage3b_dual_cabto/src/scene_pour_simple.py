"""stage3b 简化版单向倒水场景（复用 stage2 几何构件，仅调整杯/罐坐标）
====================================================================
动机
----
stage2 原 scene_pour 把接球杯 cupR 放在 y=-0.06（越过双臂中线，进入右臂区）。
实测：左臂**空载** EE 能到 y=-0.06，但一旦**夹持 canL**（gripper 闭合 + weld 刚性），
左臂的可达 y 被显著压缩到约 +0.08，无法把罐搬到 y=-0.06 上方倾倒——这是该布局
下左臂的真实运动学极限（详见诊断：move_above 后 can 仍停在 y≈0.12）。

按用户明确授权「可以简化场景和任务，提升成功率」，本场景把**接球杯放进左臂夹持
可达区**：canL 起始 (0.30, 0.11)，cupR 放 (0.32, 0.02)（近中线、左臂持罐可稳到）。
任务语义不变：左臂抓起装球的罐 → 移到另一位置的固定杯上方 → 翻罐把球倒入杯中。

仅本文件重定义坐标；罐/杯/球的几何构件、材质、weld、桌面相机全部复用 stage2
scene_pour 的 helper，确保物理一致、bug fix 自动传播。
"""
from __future__ import annotations

import _bridge as B

sp = B.scene_pour                       # stage2 原 pour 场景（helper 来源）
dc = B.scene_dual_common

# ---- 复用 stage2 的所有尺寸常量（供 codegen / world_state 直接引用）----
TABLE_H = sp.TABLE_H
CAN_R_IN = sp.CAN_R_IN
CAN_WALL = sp.CAN_WALL
CAN_HH = sp.CAN_HH
CUP_R_IN = sp.CUP_R_IN
CUP_WALL = sp.CUP_WALL
CUP_HH = sp.CUP_HH
CONTENT_R = sp.CONTENT_R
CUP_TOP = sp.CUP_TOP

# ---- 简化坐标：全部落在左臂「夹持可达区」（y 约 +0.02 ~ +0.12）----
CANL_XY = (0.28, 0.13)          # 左臂罐（内含 ballL）
CUPR_XY = (0.185, 0.045)        # 接球杯：放在「稳定倾倒位下球倒出的自然落点」
# 稳定倾倒位（实测：左臂持罐可稳到、tip 不爆炸的位形），球由此往 -x 倒出恰落入上面的杯
POUR_POSE_XY = (0.275, 0.06)
# 为保持与 codegen 中 CANR_XY 引用兼容（put_back 用），给一个占位右罐坐标（本任务不用）
CANR_XY = (0.30, -0.11)


def build_scene_xml():
    """只放一个装球的罐 canL + 一个固定接球杯 cupR（单向倒，最小闭环）。"""
    objs = "\n".join([
        sp._can_body("canL", CANL_XY, "ballL", "mat_blue"),
        sp._cup_body_fixed("cupR", CUPR_XY),
    ])
    welds = "\n".join([
        '    <weld name="grasp_weld_L_canL" body1="hand_L" body2="canL" '
        'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>',
    ])
    return dc.make_dual_xml("exp4_pour_simple", objs, welds)


if __name__ == "__main__":
    import mujoco
    xml = build_scene_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"[ok] pour_simple 编译成功 nq={m.nq} nv={m.nv} neq={m.neq}")
    print(f"  canL@{CANL_XY} -> cupR@{CUPR_XY}  CUP_TOP={CUP_TOP:.3f} CUP_R_IN={CUP_R_IN}")
