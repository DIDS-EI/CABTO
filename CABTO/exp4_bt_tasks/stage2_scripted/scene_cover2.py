"""cover2 场景生成器（单臂）—— 两排错色布局：前排 3 个盖，后排 3 个底座，颜色交错。
================================================================================
与 scene_cover.py 的区别：原版每个 lid 就在同色 base 正前方（同 y），搬运是纯 +x 直线；
本版两排颜色故意错开，每个 lid 必须斜向搬到同色 base 上，路径互相交叉，对 refine_above
的 xy 对位和 move_above 的高位避障要求更高。

布局（俯视，x 向右为远离机械臂，y 向上为 +y）。相对上一版紧凑布局
(lid x=0.38 / base x=0.50 / y=±0.11) 间距约 3×：同排 y 严格 0.33（0.11×3），
两排 x 取 0.32（0.12×3=0.36 会把中心盖逼到内侧奇异区抓不稳，故收到 0.32）。

                       y=+0.33   y=0.00   y=-0.33
    后排 base  x=0.52     黄        绿        红      （固定底座）
    前排 lid   x=0.20     绿        红        黄      （可抓盖）

配对：绿 lid(y=+0.33) -> 绿 base(y=0.00)
      红 lid(y= 0.00) -> 红 base(y=-0.33)
      黄 lid(y=-0.33) -> 黄 base(y=+0.33)   跨整排，路径最长（dy=0.66）

可达性（实测，Panda 从基座 (-0.18,0) 极限约 0.855 m 水平）：后排外侧底座
(0.52,±0.33) 水平 0.774、搬到 HIGH=0.72 时 3D 距离 0.837，在极限内；前排中心盖
(0.20,0) 离臂根 0.38（严格 3× 时是 0.14/0.32，会落进内侧奇异区，完整 rollout 里
中心盖必抓空）。三对搬运误差见 result.json。

- 每个 lid 一个 inactive weld：grasp_weld_lid_g / _r / _y。
- 成功：3 个 lid 都盖到同色 base 中心上方（xy 对齐 + 落在 base 顶面附近）。
- run_cover2.py 按 STAGE2.md 经验 3 保留确定性 snap 兜底，并在 result.json 里同时
  记录纯闭环误差(xy_off_closedloop)与 snap 后误差(xy_off)；SNAP=0 可关掉 snap。
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
# 3× 后物件摊到 x∈[0.14,0.50]、y=±0.33。桌中心仍 (0.30,0)，半尺寸 (0.45,0.52) 即可
# 兜住（桌面 x∈[-0.15,0.75]，y∈[-0.52,0.52]），比双臂场景的 0.925 还窄。
TABLE_HALF = (0.45, 0.52, 0.02)
SINGLE_BASE = (-0.18, 0.0, TABLE_H)
SINGLE_QUAT = "1 0 0 0"

# ---- 固定目标底座（矮圆柱），顶面 = TABLE_H + 2*BASE_HH ----
BASE_R = 0.038
BASE_HH = 0.020
BASE_TOP = TABLE_H + 2 * BASE_HH
# ---- 可抓盖（彩色圆盘 + 把手），与对应底座同色 ----
LID_R = 0.032
LID_HH = 0.010
HANDLE_HH = 0.022

# ---- 两排错色布局（相对紧凑版约 3×） ----
# 紧凑版：lid x=0.38、base x=0.50、y=±0.11。间距约乘 3：
#   同排 y：0.11 → 0.33（严格 3×；相邻底座净空 0.33-2*0.038=0.254）
#   两排 x：0.12 → 0.32（两排净空 0.32-0.038-0.032=0.250）
# 两排 x 没取满 0.36：严格 3× 时 lid=0.14，中心盖 (0.14,0) 离臂根仅 0.32 落进内侧奇异
# 区，完整 rollout 里必抓空（实测 grab_dz=-0.31 空夹）。收到 lid=0.20 后中心盖 r=0.38
# 稳抓；后排 base=0.52 时外侧 (0.52,±0.33) 搬运 3D 距离 0.837 仍在 0.855 极限内。
LID_ROW_X = 0.20                    # 前排（靠机械臂）：可抓盖
BASE_ROW_X = 0.52                   # 后排（远离机械臂）：固定底座
ROW_Y = (0.33, 0.00, -0.33)         # 同排三个槽位，从 +y 到 -y
FRONT_ORDER = ("g", "r", "y")       # 前排颜色：绿 红 黄
BACK_ORDER = ("y", "g", "r")        # 后排颜色：黄 绿 红
PICK_ORDER = ("g", "r", "y")        # 执行顺序：中间 base 先盖

MATERIAL = {"g": "mat_green", "r": "mat_red", "y": "mat_yellow"}

_LID_Y = {k: y for k, y in zip(FRONT_ORDER, ROW_Y)}
_BASE_Y = {k: y for k, y in zip(BACK_ORDER, ROW_Y)}

# (key, base_xy, lid_xy, material)
PAIRS = [
    (k, (BASE_ROW_X, _BASE_Y[k]), (LID_ROW_X, _LID_Y[k]), MATERIAL[k])
    for k in PICK_ORDER
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
    <material name="mat_yellow" rgba="0.93 0.82 0.16 1"/>
    <material name="mat_base_r" rgba="0.55 0.16 0.14 1"/>
    <material name="mat_base_g" rgba="0.16 0.48 0.20 1"/>
    <material name="mat_base_y" rgba="0.60 0.52 0.10 1"/>
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
    # 3× 后簇跨度约 0.36 × 0.66，相机适度拉远。
    overview = ('    <camera name="overview" pos="1.55 -1.05 1.35" '
                'xyaxes="0.58 0.81 0 -0.44 0.31 0.84" fovy="44"/>')
    front = ('    <camera name="front" pos="1.95 0.0 1.05" '
             'xyaxes="0 1 0 -0.45 0 0.89" fovy="54"/>')
    # 俯视对准两排中点 x=(0.20+0.52)/2=0.36；z=1.50 时 fovy=42 半视野
    # 1.50*tan(21°)=0.576 > 0.33，两侧和后排都进画。
    top = ('    <camera name="top" pos="0.36 0.0 1.50" '
           'xyaxes="0 1 0 -1 0 0" fovy="42"/>')
    return overview + "\n" + front + "\n" + top


def build_scene_xml():
    arm = panda_body("", SINGLE_BASE, SINGLE_QUAT)
    tendon, equality, actuator = panda_tendon_equality_actuator("")

    obj_lines, weld_lines = [], []
    for key, (bx, by), (lx, ly), mat in PAIRS:
        obj_lines.append(
            f'    <body name="base_{key}" pos="{bx} {by} {TABLE_H}">\n'
            f'      <geom type="cylinder" size="{BASE_R} {BASE_HH}" pos="0 0 {BASE_HH}" '
            f'material="mat_base_{key}"/>\n'
            f'    </body>')
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
    # 点缀白盘放到桌前空处，避开 6 个任务物件（最近的是 lid_y(0.14,-0.33)/base_r(0.50,-0.33)）。
    obj_lines.append(
        f'    <body name="plate" pos="0.30 -0.44 {TABLE_H+0.006}">\n'
        f'      <geom type="cylinder" size="0.07 0.006" material="mat_plate"/>\n'
        f'    </body>')

    objs = "\n".join(obj_lines)
    weld = "\n".join(weld_lines)
    world = _table_and_world() + "\n" + _cameras() + "\n" + arm + "\n" + objs
    return f"""<mujoco model="exp4_cover2">
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
    print(f"[ok] cover2 编译成功 nq={m.nq} nv={m.nv} neq={m.neq}")
    print(f"  前排 lid  x={LID_ROW_X}: " +
          "  ".join(f"{k}@y={_LID_Y[k]:+.2f}" for k in FRONT_ORDER))
    print(f"  后排 base x={BASE_ROW_X}: " +
          "  ".join(f"{k}@y={_BASE_Y[k]:+.2f}" for k in BACK_ORDER))
    for key, b, l, _ in PAIRS:
        print(f"  pair {key}: lid@{l} -> base@{b}  "
              f"(dx={b[0]-l[0]:+.2f} dy={b[1]-l[1]:+.2f})")
    print(f"  BASE_TOP={BASE_TOP:.3f}")
