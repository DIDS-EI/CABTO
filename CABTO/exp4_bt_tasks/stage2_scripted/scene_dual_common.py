"""双臂场景公共件（仿 stage3 blocks_scene 模式，含 grasp_weld）。
================================================================
关键修正：把双臂基座间距缩到 y=±0.55（原 scene_builder 是 ±0.855，跨度 1.85m，
两臂工作区完全不重叠，协作任务无法物理完成）。±0.55 时两臂工作区在桌中(y≈0)
重叠，pour/handover/storage 才可达。

提供：
- LEFT_BASE/RIGHT_BASE/朝向四元数；
- _common_assets / _table_and_world / _dual_cameras；
- make_dual_xml(objs_xml, welds_xml)：拼出完整双臂 XML。
"""
import os, sys, math

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
TABLE_HALF_DUAL = (0.45, 0.70, 0.02)        # y 半长 0.70（全长 1.4m），容两臂+重叠工作区

LEFT_Y = 0.55
RIGHT_Y = -0.55
LEFT_BASE = (0.30, LEFT_Y, TABLE_H)
RIGHT_BASE = (0.30, RIGHT_Y, TABLE_H)
LEFT_YAW = -math.pi / 2                      # 面向 -y（桌中心）
RIGHT_YAW = math.pi / 2                      # 面向 +y（桌中心）


def _yaw_quat(yaw):
    return f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}"


LEFT_QUAT = _yaw_quat(LEFT_YAW)
RIGHT_QUAT = _yaw_quat(RIGHT_YAW)


def common_assets():
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
    <material name="mat_cup" rgba="0.86 0.90 0.95 0.7"/>
    <material name="mat_carton" rgba="0.72 0.55 0.34 1"/>
    <material name="mat_carton_in" rgba="0.60 0.45 0.27 1"/>
    <material name="mat_content" rgba="0.20 0.55 0.95 1"/>"""


def table_and_world():
    tx, ty, tz = TABLE_HALF_DUAL
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


def dual_cameras():
    overview = ('    <camera name="overview" pos="1.85 -1.55 1.45" '
                'xyaxes="0.64 0.77 0 -0.42 0.35 0.84" fovy="46"/>')
    front = ('    <camera name="front" pos="1.65 0.0 0.86" '
             'xyaxes="0 1 0 -0.42 0 0.91" fovy="52"/>')
    return overview + "\n" + front


def make_dual_xml(model_name, objs_xml, welds_xml):
    armL = panda_body("_L", LEFT_BASE, LEFT_QUAT)
    armR = panda_body("_R", RIGHT_BASE, RIGHT_QUAT)
    tdL, eqL, acL = panda_tendon_equality_actuator("_L")
    tdR, eqR, acR = panda_tendon_equality_actuator("_R")
    world = (table_and_world() + "\n" + dual_cameras() + "\n" + armL + "\n" + armR
             + "\n" + objs_xml)
    return f"""<mujoco model="{model_name}">
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
{common_assets()}
{panda_asset_block()}
  </asset>
  <worldbody>
{world}
  </worldbody>
  <tendon>
{tdL}
{tdR}
  </tendon>
  <equality>
{eqL}
{eqR}
{welds_xml}
  </equality>
  <actuator>
{acL}
{acR}
  </actuator>
</mujoco>"""
