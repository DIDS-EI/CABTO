"""stack 场景生成器（单臂）—— 真实规格：4 个物体堆叠在一起。
================================================================
- 4 个可抓 box（红/绿/蓝/黄，freejoint），初始分散在单臂可达区（不堆叠）。
- 任务：把 4 个 box 依次堆成一摞（base block 不动，其余 3 个逐个叠上去）。
- 每个 box 一个 inactive weld：grasp_weld_{name}，运行时按抓谁激活。
- 成功：4 个 box 沿 z 依次堆叠（每个 block 底面≈下方 block 顶面、xy 对齐）。

复用 stage3 已调通的堆叠运动本质（pick→move_above 补偿抓偏→纯竖直下放→
手动解 weld 防弹飞），但用 stage2 通用层 exp4_env/arm_skills 表达，并扩展到 4 物体。
"""
import os, sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_SCENES = os.path.abspath(os.path.join(_THIS, "..", "scenes"))
for _c in [_THIS, _SCENES]:
    if os.path.isfile(os.path.join(_c, "panda_body.py")):
        if _c not in sys.path:
            sys.path.insert(0, os.path.abspath(_c))
        break
from panda_body import (panda_body, panda_asset_block, panda_default_block,
                        panda_tendon_equality_actuator)

MESHDIR = os.path.abspath(os.path.join(_THIS, "..", "assets", "panda_assets"))
TABLE_H = 0.40
TABLE_HALF = (0.45, 0.55, 0.02)
SINGLE_BASE = (-0.18, 0.0, TABLE_H)
SINGLE_QUAT = "1 0 0 0"

BLOCK_HALF = 0.025                # box 半尺寸 -> 全高 0.05
STACK_XY = (0.54, 0.0)           # 堆叠目标位（远端，避免前臂扫到塔）

# 4 个 box：base(蓝) 在堆叠位不动，其余 3 个分散在「近端」可达区，依次叠上去。
# (name, material, init_xy) —— 堆叠顺序按列表顺序（base 在最前）
# 【关键几何】单臂从 -x 伸出，前臂扫近端(小 x)空间。若堆叠塔在近端，抓取后续块时
# 前臂会压在塔顶导致 descend 卡死（实测 ee_z 冻结在 0.55 下不去）。故把堆叠塔放
# 远端 x=0.54，待抓块全部放在近端 x∈[0.42,0.44]——抓近端块时前臂不会扫到远端塔，
# descend 实测全部到底（ee_z≈0.437）。
BLOCKS = [
    ("blue_block",   "mat_blue",   (0.54,  0.00)),    # base（堆叠位，远端）
    ("green_block",  "mat_green",  (0.42,  0.00)),
    ("yellow_block", "mat_yellow", (0.42,  0.11)),
    ("red_block",    "mat_red",    (0.42, -0.11)),
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
    <material name="mat_yellow" rgba="0.92 0.82 0.16 1"/>"""


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
    overview = ('    <camera name="overview" pos="1.05 -0.70 0.94" '
                'xyaxes="0.55 0.84 0 -0.46 0.30 0.84" fovy="44"/>')
    front = ('    <camera name="front" pos="1.45 0.0 0.80" '
             'xyaxes="0 1 0 -0.45 0 0.89" fovy="50"/>')
    return overview + "\n" + front


def _free_box(name, material, xy):
    x, y = xy
    z = TABLE_H + BLOCK_HALF + 0.002
    return f"""    <body name="{name}" pos="{x} {y} {z}">
      <freejoint name="{name}_free"/>
      <geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" material="{material}" mass="0.05"
            friction="1.0 0.02 0.001"/>
    </body>"""


def build_scene_xml():
    arm = panda_body("", SINGLE_BASE, SINGLE_QUAT)
    tendon, equality, actuator = panda_tendon_equality_actuator("")
    objs = "\n".join(_free_box(n, m, xy) for n, m, xy in BLOCKS)
    welds = "\n".join(
        f'    <weld name="grasp_weld_{n}" body1="hand" body2="{n}" '
        f'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>'
        for n, _, _ in BLOCKS)
    world = _table_and_world() + "\n" + _cameras() + "\n" + arm + "\n" + objs
    return f"""<mujoco model="exp4_stack">
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
{welds}
  </equality>
  <actuator>
{actuator}
  </actuator>
</mujoco>"""


if __name__ == "__main__":
    import mujoco
    xml = build_scene_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"[ok] stack 编译成功 nq={m.nq} nv={m.nv} neq={m.neq}")
    for n, _, xy in BLOCKS:
        print(f"  {n}: init@{xy}")
    print(f"  STACK_XY={STACK_XY} BLOCK_HALF={BLOCK_HALF}")
