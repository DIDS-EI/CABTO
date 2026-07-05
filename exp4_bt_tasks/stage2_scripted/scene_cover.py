"""cover 场景生成器（单臂）—— 真实规格：3 个物体分别盖/放到对应的 3 个目标上面。
================================================================================
- 3 个固定目标 base（矮圆柱底座，无 freejoint），分散在臂可达区，各带颜色标记。
- 3 个可抓 lid（带把手的彩色圆盘，freejoint），分散在底座前方，颜色与对应 base 一致。
- 任务：把每个 lid 抓起盖到「同色」base 顶面上（3 对配对放置）。
- 另放少量点缀物（不参与抓取）。
- 每个 lid 一个 inactive weld：grasp_weld_lid_r / grasp_weld_lid_g / grasp_weld_lid_b。
- 成功：3 个 lid 都盖到对应 base 中心上方（xy 对齐 + 落在 base 顶面附近）。
"""
import os, sys

_THIS = os.path.dirname(os.path.abspath(__file__))
for _c in [_THIS, "/Users/yishuaicai/mywork/CABTO/exp4_bt_tasks/scenes"]:
    if os.path.isfile(os.path.join(_c, "panda_body.py")):
        if _c not in sys.path:
            sys.path.insert(0, os.path.abspath(_c))
        break
from panda_body import (panda_body, panda_asset_block, panda_default_block,
                        panda_tendon_equality_actuator)

MESHDIR = "/Users/yishuaicai/mywork/CABTO/exp4_bt_tasks/assets/panda_assets"
TABLE_H = 0.40
TABLE_HALF = (0.45, 0.55, 0.02)
SINGLE_BASE = (-0.18, 0.0, TABLE_H)
SINGLE_QUAT = "1 0 0 0"

# ---- 3 个固定目标底座（矮圆柱），顶面 = TABLE_H + 2*BASE_HH ----
BASE_R = 0.038                       # 缩小：单臂可达区窄，3 对物体需紧凑排布
BASE_HH = 0.020                      # 底座半高
BASE_TOP = TABLE_H + 2 * BASE_HH     # 底座顶面 z（盖盖落点）
# ---- 3 个可抓盖（彩色圆盘 + 把手），与对应底座同色 ----
LID_R = 0.032
LID_HH = 0.010
HANDLE_HH = 0.022                    # 把手半高

# (key, base_xy, lid_xy, material) —— base 在后排(x大)，对应 lid 在前排(x小)
# 全部收进单臂舒适可达区 x∈[0.45,0.54], y∈[-0.10,+0.10]（diag 实测 pe<0.02）
# y 间距 0.09：足够避免夹爪蹭倒相邻 lid，又都在舒适可达区内（y=±0.11 时 lid_r 已到边缘）。
# 顺序：先抓中间 g（最稳），再抓两侧 r/b——避免先放好的盖挡在中间盖的下探路径上。
PAIRS = [
    ("g", (0.52,  0.00), (0.43,  0.00), "mat_green"),
    ("r", (0.52,  0.09), (0.43,  0.09), "mat_red"),
    ("b", (0.52, -0.09), (0.43, -0.09), "mat_blue"),
]


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
    <material name="mat_red" rgba="0.85 0.20 0.18 1"/>
    <material name="mat_green" rgba="0.22 0.72 0.28 1"/>
    <material name="mat_blue" rgba="0.18 0.40 0.85 1"/>
    <material name="mat_base_r" rgba="0.55 0.16 0.14 1"/>
    <material name="mat_base_g" rgba="0.16 0.48 0.20 1"/>
    <material name="mat_base_b" rgba="0.14 0.28 0.58 1"/>
    <material name="mat_plate" rgba="0.93 0.93 0.95 1"/>"""


def _table_and_world():
    tx, ty, tz = TABLE_HALF
    legs = ""
    for lx in (tx - 0.06, -(tx - 0.06)):
        for ly in (ty - 0.06, -(ty - 0.06)):
            legs += (f'    <geom type="box" size="0.025 0.025 {TABLE_H/2}" '
                     f'pos="{lx} {ly} {TABLE_H/2 - tz}" material="table_dark"/>\n')
    return f"""    <light name="key" pos="0.6 -0.6 2.2" dir="-0.25 0.25 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="-0.6 0.6 1.8" dir="0.25 -0.25 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
    <body name="table" pos="0.30 0 {TABLE_H - tz}">
      <geom type="box" size="{tx} {ty} {tz}" material="table"/>
    </body>
{legs}"""


def _cameras():
    overview = ('    <camera name="overview" pos="1.10 -0.74 0.98" '
                'xyaxes="0.58 0.81 0 -0.44 0.31 0.84" fovy="44"/>')
    front = ('    <camera name="front" pos="1.45 0.0 0.80" '
             'xyaxes="0 1 0 -0.45 0 0.89" fovy="50"/>')
    return overview + "\n" + front


def build_scene_xml():
    arm = panda_body("", SINGLE_BASE, SINGLE_QUAT)
    tendon, equality, actuator = panda_tendon_equality_actuator("")

    obj_lines, weld_lines = [], []
    for key, (bx, by), (lx, ly), mat in PAIRS:
        # 固定目标底座
        obj_lines.append(
            f'    <body name="base_{key}" pos="{bx} {by} {TABLE_H}">\n'
            f'      <geom type="cylinder" size="{BASE_R} {BASE_HH}" pos="0 0 {BASE_HH}" '
            f'material="mat_base_{key}"/>\n'
            f'    </body>')
        # 可抓盖（圆盘 + 把手）
        lid_z = TABLE_H + LID_HH + 0.002
        obj_lines.append(
            f'    <body name="lid_{key}" pos="{lx} {ly} {lid_z}">\n'
            f'      <freejoint name="lid_{key}_free"/>\n'
            f'      <geom type="cylinder" size="{LID_R} {LID_HH}" material="{mat}" '
            f'mass="0.035" friction="1.0 0.02 0.001"/>\n'
            f'      <geom type="box" size="0.016 0.016 {HANDLE_HH}" '
            f'pos="0 0 {LID_HH+HANDLE_HH}" material="{mat}" mass="0.008"/>\n'
            f'    </body>')
        weld_lines.append(
            f'    <weld name="grasp_weld_lid_{key}" body1="hand" body2="lid_{key}" '
            f'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>')
    # 点缀（白盘）
    obj_lines.append(
        f'    <body name="plate" pos="0.30 0.32 {TABLE_H+0.006}">\n'
        f'      <geom type="cylinder" size="0.07 0.006" material="mat_plate"/>\n'
        f'    </body>')

    objs = "\n".join(obj_lines)
    weld = "\n".join(weld_lines)
    world = _table_and_world() + "\n" + _cameras() + "\n" + arm + "\n" + objs
    return f"""<mujoco model="exp4_cover">
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
  </asset>
  <worldbody>
{world}
  </worldbody>
  <tendon>
{tendon}
  </tendon>
  <equality>
{equality}
{weld}
  </equality>
  <actuator>
{actuator}
  </actuator>
</mujoco>"""


if __name__ == "__main__":
    import mujoco
    xml = build_scene_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"[ok] cover 编译成功 nq={m.nq} nv={m.nv} neq={m.neq}")
    for key, b, l, _ in PAIRS:
        print(f"  pair {key}: base@{b} lid@{l}")
    print(f"  BASE_TOP={BASE_TOP:.3f}")
