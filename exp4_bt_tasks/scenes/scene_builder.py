"""
exp4 五个 BT 长程任务的 MuJoCo 场景生成器
==========================================
对应论文 Fig.（5 个 manipulation 任务）：
  (1) cover     单 Franka —— 桌面有锅/碗、面包、白盘、红球；把盖子盖到锅上
  (2) blocks    单 Franka —— 蓝球 + 黄/绿积木；把积木堆叠
  (3) pour      双 Franka —— 左臂持红色罐倒入桌面杯子（右臂扶杯）
  (4) handover  双 Franka —— 左臂抓绿盒交给右臂（手手交接）
  (5) storage   双 Franka —— 双臂把绿色物体放进纸箱

设计约定：
  - 桌子：固定 box，桌面高度 TABLE_H=0.40，臂基座钉在桌后缘。
  - 单臂：基座在 (-0.05, 0, 0.40) 面向 +x（桌子在前方）。
  - 双臂：左臂基座 (0,+0.45,0.40) 面向 -y 偏 +x；右臂 (0,-0.45,0.40) 面向 +y 偏 +x。
    （两臂隔着桌子相对而立，便于 pour/handover/storage 协作）
  - 物体：可抓取物体用 freejoint，目标容器（锅/纸箱/杯）固定或带 freejoint 视任务定。
  - 相机：每个场景一个面向桌子的 overview 相机，构图对齐论文图。
"""

import os
from panda_body import (panda_body, panda_asset_block, panda_default_block,
                        panda_tendon_equality_actuator)

HERE = os.path.dirname(os.path.abspath(__file__))
MESHDIR = os.path.join(os.path.dirname(HERE), "assets", "panda_assets")

TABLE_H = 0.40          # 桌面高度
TABLE_HALF = (0.45, 0.55, 0.02)        # 单臂桌板半尺寸 (x,y,z)
TABLE_HALF_DUAL = (0.45, 0.925, 0.02)   # 双臂桌板半尺寸：y 方向加长，让两臂离得更远（全长 1.85m）


def _common_assets():
    return """    <texture type="skybox" builtin="gradient" rgb1="0.45 0.62 0.78" rgb2="0.30 0.45 0.62" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.36 0.52 0.66" rgb2="0.42 0.58 0.72" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
    <material name="pedestal" rgba="0.30 0.32 0.38 1"/>
    <material name="table" rgba="0.55 0.57 0.60 1"/>
    <material name="table_dark" rgba="0.33 0.35 0.39 1"/>
    <material class="panda" name="white" rgba="1 1 1 1"/>
    <material class="panda" name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
    <material class="panda" name="black" rgba="0.25 0.25 0.25 1"/>
    <material class="panda" name="green" rgba="0 1 0 1"/>
    <material class="panda" name="light_blue" rgba="0.039216 0.541176 0.780392 1"/>
    <material name="mat_red" rgba="0.85 0.18 0.15 1"/>
    <material name="mat_green" rgba="0.30 0.70 0.25 1"/>
    <material name="mat_blue" rgba="0.15 0.35 0.85 1"/>
    <material name="mat_yellow" rgba="0.92 0.82 0.16 1"/>
    <material name="mat_emerald" rgba="0.10 0.62 0.40 1"/>
    <material name="mat_bread" rgba="0.78 0.60 0.34 1"/>
    <material name="mat_plate" rgba="0.93 0.93 0.95 1"/>
    <material name="mat_pot" rgba="0.18 0.18 0.20 1"/>
    <material name="mat_lid" rgba="0.40 0.42 0.45 1"/>
    <material name="mat_cup" rgba="0.88 0.88 0.92 0.6"/>
    <material name="mat_carton" rgba="0.72 0.55 0.34 1"/>
    <material name="mat_carton_in" rgba="0.60 0.45 0.27 1"/>"""


def _table_and_world(camera, table_half=None):
    """地面 + 桌子 + 灯光 + 相机。table_half 不传则用单臂默认尺寸。"""
    tx, ty, tz = table_half or TABLE_HALF
    legs = ""
    for lx in (tx - 0.06, -(tx - 0.06)):
        for ly in (ty - 0.06, -(ty - 0.06)):
            legs += (f'    <geom type="box" size="0.025 0.025 {TABLE_H/2}" '
                     f'pos="{lx} {ly} {TABLE_H/2 - tz}" material="table_dark"/>\n')
    return f"""    <light name="key" pos="0.6 -0.6 2.2" dir="-0.25 0.25 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="-0.6 0.6 1.8" dir="0.25 -0.25 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
    {camera}
    <body name="table" pos="0.30 0 {TABLE_H - tz}">
      <geom type="box" size="{tx} {ty} {tz}" material="table"/>
    </body>
{legs}"""


def _wrap(assets_extra, world_bodies, tendons, equalities, actuators, ndof_note=""):
    """拼成完整 MuJoCo XML 文档。"""
    return f"""<mujoco model="exp4_bt_task">
  <compiler angle="radian" meshdir="{MESHDIR}" autolimits="true"/>
  <option integrator="implicitfast" gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>
  </visual>

  <default>
{panda_default_block()}
  </default>

  <asset>
{_common_assets()}
{panda_asset_block()}
{assets_extra}
  </asset>

  <worldbody>
{world_bodies}
  </worldbody>

  <tendon>
{tendons}
  </tendon>

  <equality>
{equalities}
  </equality>

  <actuator>
{actuators}
  </actuator>
</mujoco>"""


# ============================================================ 单臂基座
SINGLE_BASE = (-0.18, 0.0, TABLE_H)        # 面向 +x（桌子前方），后移让臂完整可见
SINGLE_QUAT = "1 0 0 0"
# ============================================================ 双臂基座（隔桌相对）
# 左臂在 +y 侧，需转 -90° 面向 -y 方向的桌中心区；右臂在 -y 侧转 +90°
import math
def _yaw_quat(yaw):
    return f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}"
LEFT_BASE = (0.30, 0.855, TABLE_H)
LEFT_QUAT = _yaw_quat(-math.pi/2)          # 面向 -y（桌中心）
RIGHT_BASE = (0.30, -0.855, TABLE_H)
RIGHT_QUAT = _yaw_quat(math.pi/2)          # 面向 +y（桌中心）


def _overview_cam():
    # 抬高、拉远，俯角更大，容纳完整双臂；构图对齐论文（桌子在中下、臂在上方）
    return ('<camera name="overview" pos="1.95 -1.75 1.75" '
            'xyaxes="0.66 0.75 0 -0.42 0.37 0.83"/>')


def _overview_cam_dual():
    # 双臂桌子（y 向 1.85m），相机拉远以容纳两臂
    return ('<camera name="overview" pos="2.48 -2.28 2.14" '
            'xyaxes="0.66 0.75 0 -0.42 0.37 0.83"/>')


# ============================================================ 物体片段
def _free_box(name, pos, half, material, mass=0.05):
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_free"/>
      <geom type="box" size="{half[0]} {half[1]} {half[2]}" material="{material}" mass="{mass}"/>
    </body>"""


def _free_cyl(name, pos, r, hh, material, mass=0.05):
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_free"/>
      <geom type="cylinder" size="{r} {hh}" material="{material}" mass="{mass}"/>
    </body>"""


def _free_sphere(name, pos, r, material, mass=0.05):
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_free"/>
      <geom type="sphere" size="{r}" material="{material}" mass="{mass}"/>
    </body>"""


# ============================================================ 五个任务场景
def build_cover():
    """(1) cover 单臂：桌上锅(固定) + 锅盖(可抓) + 面包 + 白盘 + 红球。盖盖子。"""
    arm = panda_body("", SINGLE_BASE, SINGLE_QUAT)
    td, eq, ac = panda_tendon_equality_actuator("")
    z = TABLE_H + 0.001
    objs = []
    # 黑锅（固定容器，开口向上的薄壁，这里用一个矮圆柱代锅体 + 锅盖单独可抓）
    objs.append(f"""    <body name="pot" pos="0.42 0.16 {z+0.03}">
      <geom type="cylinder" size="0.07 0.03" material="mat_pot"/>
    </body>""")
    # 锅盖（可抓物体，初始放在旁边）
    objs.append(f"""    <body name="lid" pos="0.42 -0.05 {z+0.012}">
      <freejoint name="lid_free"/>
      <geom type="cylinder" size="0.065 0.012" material="mat_lid" mass="0.05"/>
      <geom type="box" size="0.012 0.012 0.02" pos="0 0 0.025" material="mat_lid" mass="0.01"/>
    </body>""")
    # 面包（黄褐色圆块）
    objs.append(_free_cyl("bread", (0.30, 0.0, z + 0.02), 0.04, 0.02, "mat_bread", 0.03))
    # 白盘（固定矮圆盘）
    objs.append(f"""    <body name="plate" pos="0.50 -0.18 {z+0.006}">
      <geom type="cylinder" size="0.08 0.006" material="mat_plate"/>
    </body>""")
    # 红球
    objs.append(_free_sphere("red_ball", (0.24, -0.20, z + 0.03), 0.03, "mat_red", 0.04))
    world = _table_and_world(_overview_cam()) + "\n" + arm + "\n" + "\n".join(objs)
    return _wrap("", world, td, eq, ac)


def build_blocks():
    """(2) blocks 单臂：蓝球 + 黄块 + 绿块。把方块堆叠。"""
    arm = panda_body("", SINGLE_BASE, SINGLE_QUAT)
    td, eq, ac = panda_tendon_equality_actuator("")
    z = TABLE_H + 0.001
    objs = [
        _free_sphere("blue_ball", (0.30, 0.16, z + 0.03), 0.03, "mat_blue", 0.04),
        _free_box("yellow_block", (0.42, -0.14, z + 0.025), (0.025, 0.025, 0.025), "mat_yellow", 0.04),
        _free_box("green_block", (0.46, -0.06, z + 0.025), (0.025, 0.025, 0.025), "mat_green", 0.04),
    ]
    world = _table_and_world(_overview_cam()) + "\n" + arm + "\n" + "\n".join(objs)
    return _wrap("", world, td, eq, ac)


def build_pour():
    """(3) pour 双臂：左臂持红罐，右臂扶杯。把罐内（红块代示）倒入杯中。"""
    armL = panda_body("_L", LEFT_BASE, LEFT_QUAT)
    armR = panda_body("_R", RIGHT_BASE, RIGHT_QUAT)
    tdL, eqL, acL = panda_tendon_equality_actuator("_L")
    tdR, eqR, acR = panda_tendon_equality_actuator("_R")
    z = TABLE_H + 0.001
    objs = [
        # 红色罐（可抓，左臂区）
        _free_cyl("can", (0.34, 0.38, z + 0.06), 0.04, 0.06, "mat_red", 0.06),
        # 白杯（可抓/扶，右臂区）
        f"""    <body name="cup" pos="0.30 -0.38 {z+0.055}">
      <freejoint name="cup_free"/>
      <geom type="cylinder" size="0.045 0.055" material="mat_cup" mass="0.05"/>
    </body>""",
    ]
    world = (_table_and_world(_overview_cam_dual(), TABLE_HALF_DUAL) + "\n" + armL + "\n" + armR
             + "\n" + "\n".join(objs))
    return _wrap("", world, tdL + "\n" + tdR, eqL + "\n" + eqR, acL + "\n" + acR)


def build_handover():
    """(4) handover 双臂：左臂抓绿盒，抬到中间交给右臂。"""
    armL = panda_body("_L", LEFT_BASE, LEFT_QUAT)
    armR = panda_body("_R", RIGHT_BASE, RIGHT_QUAT)
    tdL, eqL, acL = panda_tendon_equality_actuator("_L")
    tdR, eqR, acR = panda_tendon_equality_actuator("_R")
    z = TABLE_H + 0.001
    objs = [
        # 绿盒（左臂先抓）
        _free_box("green_box", (0.30, 0.44, z + 0.035), (0.03, 0.045, 0.035), "mat_emerald", 0.05),
        # 桌面另放两个绿色物体做点缀（论文图右侧）
        _free_box("green_box2", (0.40, -0.34, z + 0.035), (0.03, 0.045, 0.035), "mat_green", 0.05),
        _free_cyl("green_btl", (0.48, -0.52, z + 0.06), 0.03, 0.06, "mat_emerald", 0.05),
    ]
    world = (_table_and_world(_overview_cam_dual(), TABLE_HALF_DUAL) + "\n" + armL + "\n" + armR
             + "\n" + "\n".join(objs))
    return _wrap("", world, tdL + "\n" + tdR, eqL + "\n" + eqR, acL + "\n" + acR)


def build_storage():
    """(5) storage 双臂：纸箱(固定) + 绿色物体若干。双臂把物体放进纸箱。"""
    armL = panda_body("_L", LEFT_BASE, LEFT_QUAT)
    armR = panda_body("_R", RIGHT_BASE, RIGHT_QUAT)
    tdL, eqL, acL = panda_tendon_equality_actuator("_L")
    tdR, eqR, acR = panda_tendon_equality_actuator("_R")
    z = TABLE_H + 0.001
    # 纸箱：四壁 + 底，固定在桌中
    carton = f"""    <body name="carton" pos="0.42 0.0 {z}">
      <geom type="box" size="0.10 0.10 0.004" pos="0 0 0.004" material="mat_carton_in"/>
      <geom type="box" size="0.10 0.006 0.05" pos="0 0.10 0.05" material="mat_carton"/>
      <geom type="box" size="0.10 0.006 0.05" pos="0 -0.10 0.05" material="mat_carton"/>
      <geom type="box" size="0.006 0.10 0.05" pos="0.10 0 0.05" material="mat_carton"/>
      <geom type="box" size="0.006 0.10 0.05" pos="-0.10 0 0.05" material="mat_carton"/>
    </body>"""
    objs = [
        carton,
        _free_box("item_g1", (0.26, 0.51, z + 0.025), (0.025, 0.025, 0.025), "mat_green", 0.04),
        _free_box("item_g2", (0.26, -0.51, z + 0.025), (0.025, 0.025, 0.025), "mat_emerald", 0.04),
    ]
    world = (_table_and_world(_overview_cam_dual(), TABLE_HALF_DUAL) + "\n" + armL + "\n" + armR
             + "\n" + "\n".join(objs))
    return _wrap("", world, tdL + "\n" + tdR, eqL + "\n" + eqR, acL + "\n" + acR)


BUILDERS = {
    "cover": build_cover,
    "blocks": build_blocks,
    "pour": build_pour,
    "handover": build_handover,
    "storage": build_storage,
}

ARMS = {"cover": 1, "blocks": 1, "pour": 2, "handover": 2, "storage": 2}


def write_all(out_dir=None):
    out_dir = out_dir or HERE
    paths = {}
    for name, fn in BUILDERS.items():
        xml = fn()
        p = os.path.join(out_dir, f"scene_{name}.xml")
        with open(p, "w") as f:
            f.write(xml)
        paths[name] = p
    return paths


if __name__ == "__main__":
    paths = write_all()
    for n, p in paths.items():
        print(f"{n:10s} ({ARMS[n]} arm) -> {p}")
