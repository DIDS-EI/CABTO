"""pour 双臂场景（双向互倒 · 固定杯版）—— 真实规格：
① 左臂抓 canL（内含 ballL）→ 移到「右臂区固定杯 cupR」上方 → 翻罐倒球 → ballL 落入 cupR；
② 右臂抓 canR（内含 ballR）→ 移到「左臂区固定杯 cupL」上方 → 翻罐倒球 → ballR 落入 cupL。

设计决策（经诊断后用户选定）：
- 杯 cupL / cupR **固定**（无 freejoint），消除「自由杯被罐/臂撞飞」的根本问题
  （这是老单向版 success=True 的关键：cup 固定，用 get_body_pos 查询）。
- 罐 canL/canR 可抓（freejoint），各放本臂稳定抓取区。
- 倾倒沿用已验证手法：把罐移到固定杯上方预偏点，绕世界水平轴翻转 ~125°（total_angle=2.2），
  罐口画弧落到杯心上方，球滚出落入杯。
weld：grasp_weld_L_canL（左臂抓 canL）、grasp_weld_R_canR（右臂抓 canR）。
成功：ballL 落入 cupR、ballR 落入 cupL（xy 在杯口内、低于杯沿、未掉地）。
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_dual_common as dc

TABLE_H = dc.TABLE_H

# 罐（带球，被倒）几何
CAN_R_IN = 0.038
CAN_WALL = 0.006
CAN_HH = 0.050
# 杯（固定，接球）几何 —— 口大一点便于接住
CUP_R_IN = 0.055
CUP_WALL = 0.006
CUP_HH = 0.050
CONTENT_R = 0.020
CUP_TOP = TABLE_H + 2 * CUP_HH    # 0.50

# 摆位：罐放各臂稳定抓取区(x≈0.29, |y|≈0.10~0.13)；
# 固定杯放「对侧倒罐臂能在其上方翻罐倾倒」的位置——参考老单向版 cup(0.31,-0.06) 左臂可倒。
# 左臂倒入的 cupR 放右半区(y<0)；右臂倒入的 cupL 放左半区(y>0)，镜像。
CANL_XY = (0.29, 0.11)         # 左臂罐（带 ballL）
CANR_XY = (0.29, -0.11)        # 右臂罐（带 ballR）—— 镜像
CUPR_XY = (0.31, -0.06)        # 右半区固定杯（接 ballL）；左臂持 canL 在其上方倾倒
CUPL_XY = (0.31, 0.06)         # 左半区固定杯（接 ballR）；右臂持 canR 在其上方倾倒


def _ring_wall(r_in, wall, h_bottom, h_top, material, n=16):
    segs = []
    rr = r_in + wall * 0.5
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = rr * math.cos(a), rr * math.sin(a)
        segs.append(f'<geom type="capsule" fromto="{x} {y} {h_bottom} {x} {y} {h_top}" '
                    f'size="{wall*0.95}" material="{material}"/>')
    return "\n      ".join(segs)


def _can_body(name, xy, ball_name, ball_mat, can_mat="mat_red"):
    cx, cy = xy
    can_cz = TABLE_H + CAN_HH + 0.002
    can_bottom = -CAN_HH
    wall = _ring_wall(CAN_R_IN, CAN_WALL, can_bottom + 0.008, CAN_HH, can_mat)
    ball_z = TABLE_H + 0.008 + CONTENT_R + 0.002
    return f"""    <body name="{name}" pos="{cx} {cy} {can_cz}">
      <freejoint name="{name}_free"/>
      <geom type="cylinder" size="{CAN_R_IN+CAN_WALL} 0.008" pos="0 0 {can_bottom+0.004}" material="{can_mat}" mass="0.04"/>
      {wall}
    </body>
    <body name="{ball_name}" pos="{cx} {cy} {ball_z}">
      <freejoint name="{ball_name}_free"/>
      <geom type="sphere" size="{CONTENT_R}" material="{ball_mat}" mass="0.012" friction="0.5 0.01 0.001"/>
    </body>"""


def _cup_body_fixed(name, xy, mat="mat_cup"):
    """固定杯（无 freejoint），坐桌面；body 原点在桌面，壁从 0.014 到 2*CUP_HH。"""
    ux, uy = xy
    return f"""    <body name="{name}" pos="{ux} {uy} {TABLE_H}">
      <geom type="cylinder" size="{CUP_R_IN+CUP_WALL} 0.008" pos="0 0 0.008" material="{mat}"/>
      {_ring_wall(CUP_R_IN, CUP_WALL, 0.014, 2*CUP_HH, mat)}
    </body>"""


def build_scene_xml():
    objs = "\n".join([
        _can_body("canL", CANL_XY, "ballL", "mat_blue"),
        _cup_body_fixed("cupL", CUPL_XY),
        _can_body("canR", CANR_XY, "ballR", "mat_yellow"),
        _cup_body_fixed("cupR", CUPR_XY),
    ])
    # 只有两个罐可抓；杯固定不焊接。
    welds = "\n".join([
        '    <weld name="grasp_weld_L_canL" body1="hand_L" body2="canL" active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>',
        '    <weld name="grasp_weld_R_canR" body1="hand_R" body2="canR" active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>',
    ])
    return dc.make_dual_xml("exp4_pour", objs, welds)


if __name__ == "__main__":
    import mujoco
    xml = build_scene_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"[ok] pour 编译成功 nq={m.nq} nv={m.nv} neq={m.neq}")
    print(f"  canL@{CANL_XY} cupR@{CUPR_XY} | canR@{CANR_XY} cupL@{CUPL_XY}")
    print(f"  CAN_HH={CAN_HH} CUP_HH={CUP_HH} CONTENT_R={CONTENT_R} CUP_TOP={CUP_TOP:.3f}")
