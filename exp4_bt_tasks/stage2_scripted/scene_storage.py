"""storage 双臂场景（入箱 + 搬箱上架版）：
阶段一（入箱）：左臂把 item_g1、右臂把 item_g2 分别放进纸箱 carton。
阶段二（搬箱上架）：双臂协同夹住纸箱两侧 → 抬起 → 搬到 -y 侧货架 shelf 台面上放下。

- carton：开口向上的纸箱（四壁 + 底盘），**带 freejoint 可被搬运**，初始放桌中两臂重叠区(0.30,0.0)。
  两侧壁外面各设一个 weld 挂点：grasp_weld_L_carton（左臂夹 +y 侧壁）、grasp_weld_R_carton（右臂夹 -y 侧壁）。
- item_g1：左臂可达区(0.27,+0.12)绿色小方块；grasp_weld_L_item_g1。
- item_g2：右臂可达区(0.27,-0.12)绿色小方块；grasp_weld_R_item_g2。
- shelf：-y 侧的固定货架台面（一块抬高的板 + 两腿），台面高于桌面，箱子最终放其上。
- 成功：两 item 都在 carton 内；且 carton 最终被搬到 shelf 台面上（xy 落在 shelf 范围、z≈台面高）。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_dual_common as dc

TABLE_H = dc.TABLE_H

# ---- 纸箱：偏 -y 侧（右臂舒适抓提手区，实测 y≈-0.11 高位 IK 最稳），开口向上，可搬运（freejoint）----
CARTON_XY = (0.30, -0.11)
CARTON_IN = (0.090, 0.060)       # 内半宽 (x, y)
CARTON_WALL = 0.008
CARTON_H = 0.030                 # 箱壁半高（箱口高度 = TABLE_H + 2*CARTON_H）
CARTON_TOP_Z = TABLE_H + 2 * CARTON_H
CARTON_MASS = 0.30               # 轻到右臂能经 weld 抬起；阶段间 snap 归位消除入箱漂移

# ---- 两个可抓绿物 ----
ITEM_HALF = (0.020, 0.020, 0.025)
ITEM1_XY = (0.27, 0.12)
ITEM2_XY = (0.27, -0.16)     # 右臂舒适抓取区，离 carton(-0.11) 足够远避免手指撞箱壁

# ---- 货架：-y 侧固定台面，箱子搬上去（拉近+降低到右臂舒适搬运区）----
SHELF_XY = (0.30, -0.20)         # 货架台面中心（右臂稳定区；远了单点悬挂会大幅摆动）
SHELF_TOP_HALF = (0.13, 0.11, 0.008)   # 台面板半尺寸（放大便于箱落入判定范围）
SHELF_DECK_Z = TABLE_H + 0.03    # 台面略高于桌面（搬运近水平，右臂只需小幅抬升即可达成）


# ---- 提手：箱口上方沿 x 方向的细横梁，供单臂从上抓取整箱搬运 ----
# 关键：finger_mid 比 ee(hand) 高约 0.045；抓取时 ee 落在右臂稳定区(世界z≈0.48)，
# 对应 finger_mid≈0.525。故把提手中心设在世界 z≈0.525（相对箱底≈0.123），
# 让手指闭合时恰好夹住横梁。再高(rel>0.145)右臂近乎完全伸展、IK 发散。
HANDLE_Z = 0.160                        # 提手中心相对箱底的高度（世界 z≈0.560，落在右臂自然可达高度）
HANDLE_HALF = (0.012, 0.018, 0.013)     # 横梁半尺寸（沿 x 长、加厚便于夹爪夹住）


def _carton_walls():
    """四壁 + 底盘 + 顶部提手（局部坐标，body 原点在箱底中心）。开口向上。"""
    ix, iy = CARTON_IN
    w = CARTON_WALL
    h = 2 * CARTON_H
    mid_z = h * 0.5
    g = []
    g.append(f'<geom type="box" size="{ix+w} {iy+w} {w*0.5}" pos="0 0 {w*0.5}" '
             f'material="mat_carton_in" mass="{CARTON_MASS*0.55}" friction="1.5 0.05 0.002"/>')
    g.append(f'<geom type="box" size="{w*0.5} {iy+w} {mid_z}" pos="{ix+w*0.5} 0 {mid_z}" '
             f'material="mat_carton" mass="{CARTON_MASS*0.12}"/>')
    g.append(f'<geom type="box" size="{w*0.5} {iy+w} {mid_z}" pos="{-(ix+w*0.5)} 0 {mid_z}" '
             f'material="mat_carton" mass="{CARTON_MASS*0.12}"/>')
    g.append(f'<geom type="box" size="{ix+w} {w*0.5} {mid_z}" pos="0 {iy+w*0.5} {mid_z}" '
             f'material="mat_carton" mass="{CARTON_MASS*0.12}"/>')
    # 提手：两根立柱（在 ±x 端，y=0）+ 横梁（顶部沿 x），形成可夹的把手
    hcx, hcy, hcz = HANDLE_HALF
    post_h = (HANDLE_Z - h) * 0.5
    post_cz = h + post_h
    g.append(f'<geom type="box" size="0.008 0.008 {post_h}" pos="{ix*0.6} 0 {post_cz}" '
             f'material="mat_carton" mass="{CARTON_MASS*0.04}"/>')
    g.append(f'<geom type="box" size="0.008 0.008 {post_h}" pos="{-ix*0.6} 0 {post_cz}" '
             f'material="mat_carton" mass="{CARTON_MASS*0.04}"/>')
    g.append(f'<geom type="box" size="{ix*0.6} {hcy} {hcz}" pos="0 0 {HANDLE_Z}" '
             f'material="mat_carton" mass="{CARTON_MASS*0.05}"/>')
    g.append(f'<geom type="box" size="{ix+w} {w*0.5} {mid_z}" pos="0 {-(iy+w*0.5)} {mid_z}" '
             f'material="mat_carton" mass="{CARTON_MASS*0.15}"/>')
    return "\n      ".join(g)


def _shelf():
    """固定货架：抬高台面板 + 四角细腿。body 原点在 SHELF_XY 桌面。"""
    sx, sy = SHELF_XY
    tx, ty, tz = SHELF_TOP_HALF
    deck_top = SHELF_DECK_Z
    deck_cz = deck_top - tz                       # 台面板中心 z（相对世界）
    leg_h = (deck_top - tz - TABLE_H) * 0.5       # 腿半高
    leg_cz = TABLE_H + leg_h
    legs = []
    for lx in (tx - 0.02, -(tx - 0.02)):
        for ly in (ty - 0.02, -(ty - 0.02)):
            legs.append(f'<geom type="box" size="0.012 0.012 {leg_h}" '
                        f'pos="{lx} {ly} {leg_cz}" material="table_dark"/>')
    legs_xml = "\n      ".join(legs)
    return f"""    <body name="shelf" pos="{sx} {sy} 0">
      <geom type="box" size="{tx} {ty} {tz}" pos="0 0 {deck_cz}" material="table_dark"/>
      {legs_xml}
    </body>"""


def build_scene_xml():
    cx, cy = CARTON_XY
    i1x, i1y = ITEM1_XY
    i2x, i2y = ITEM2_XY
    hx, hy, hz = ITEM_HALF
    iz = TABLE_H + hz + 0.002
    carton_z = TABLE_H + 0.002
    objs = f"""    <body name="carton" pos="{cx} {cy} {carton_z}">
      <freejoint name="carton_free"/>
      {_carton_walls()}
    </body>
    <body name="item_g1" pos="{i1x} {i1y} {iz}">
      <freejoint name="item_g1_free"/>
      <geom type="box" size="{hx} {hy} {hz}" material="mat_green" mass="0.04" friction="1.0 0.02 0.001"/>
    </body>
    <body name="item_g2" pos="{i2x} {i2y} {iz}">
      <freejoint name="item_g2_free"/>
      <geom type="box" size="{hx} {hy} {hz}" material="mat_emerald" mass="0.04" friction="1.0 0.02 0.001"/>
    </body>
{_shelf()}"""
    welds = (
        '    <weld name="grasp_weld_L_item_g1" body1="hand_L" body2="item_g1" '
        'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>\n'
        '    <weld name="grasp_weld_R_item_g2" body1="hand_R" body2="item_g2" '
        'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>\n'
        # carton 由右臂抓顶部提手单臂搬运（双臂同步夹薄侧壁不稳，改单臂抓提手）
        '    <weld name="grasp_weld_R_carton" body1="hand_R" body2="carton" '
        'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>\n'
        # item 入箱后与 carton 焊接（物理上已被装入箱中），随箱被整体搬运不掉出
        '    <weld name="pack_weld_g1_carton" body1="carton" body2="item_g1" '
        'active="false" solimp="0.95 0.99 0.001" solref="0.01 1"/>\n'
        '    <weld name="pack_weld_g2_carton" body1="carton" body2="item_g2" '
        'active="false" solimp="0.95 0.99 0.001" solref="0.01 1"/>')
    return dc.make_dual_xml("exp4_storage", objs, welds)


if __name__ == "__main__":
    import mujoco
    xml = build_scene_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"[ok] storage 编译成功 nq={m.nq} nv={m.nv} neq={m.neq}")
    print(f"  CARTON_XY={CARTON_XY} CARTON_TOP_Z={CARTON_TOP_Z:.3f}")
    print(f"  SHELF_XY={SHELF_XY} SHELF_DECK_Z={SHELF_DECK_Z:.3f}")
    print(f"  ITEM1_XY={ITEM1_XY} ITEM2_XY={ITEM2_XY}")
