"""stage3 blocks 场景 XML 生成器
================================
exp4 stage3（CABTO LLM 规划 + 代码生成 + VLM 校验闭环）的地基场景：
单臂 Franka Panda（固定基座）+ 桌子 + 地面 + 灯光 + 3 个可抓 box 物体。

设计要点
--------
- 复用 scenes/panda_body.py 的 asset/default/actuator 块（与 exp2 同一套 Panda），
  保证 IK/伺服常量（HOME_QPOS / GRASP_QUAT / EE_SERVO_BIAS）原样可用。
- 单臂基座 SINGLE_BASE=(-0.18,0,0.40) 面向 +x，与 scene_builder 一致。
- 3 个 freejoint box（半尺寸 0.025 -> 全高 0.05），分散摆放在臂可达区
  （x∈[0.45,0.6], y 分散），不堆叠。每个 body 含 freejoint `{name}_free`。
- 关键：MuJoCo weld 在编译期 body 固定，无法运行时改 body2。因此为每个物体
  各建一个 inactive weld：grasp_weld_green / grasp_weld_yellow / grasp_weld_blue
  （body1=hand, body2={block}），运行时按抓哪个激活哪个。
- 相机：overview（俯视看三块相对位置+堆叠）、front（正视，供两视角三角化）。
- meshdir 用 CABTO assets 绝对路径。

对外接口
--------
- build_blocks_scene_xml() -> str
- write_scene(out_path) -> str  （写文件并返回 path）
"""

import os
import sys

# 复用 scenes/panda_body.py（不复制，直接 import）
_THIS = os.path.dirname(os.path.abspath(__file__))
# stage3 真实归属目录（运行期可能在 /tmp，故按 CABTO 固定路径找 panda_body）
_SCENES_CANDIDATES = [
    os.path.join(_THIS, "..", "scenes"),                         # 部署后位置
    "/Users/yishuaicai/mywork/CABTO/exp4_bt_tasks/scenes",       # 固定回退
]
for _c in _SCENES_CANDIDATES:
    if os.path.isfile(os.path.join(_c, "panda_body.py")):
        if _c not in sys.path:
            sys.path.insert(0, os.path.abspath(_c))
        break

from panda_body import (panda_body, panda_asset_block, panda_default_block,
                        panda_tendon_equality_actuator)

# meshdir：CABTO panda 网格绝对路径
MESHDIR = "/Users/yishuaicai/mywork/CABTO/exp4_bt_tasks/assets/panda_assets"

TABLE_H = 0.40                      # 桌面顶面高度（= TABLE_TOP_Z）
TABLE_HALF = (0.45, 0.55, 0.02)     # 桌板半尺寸
SINGLE_BASE = (-0.18, 0.0, TABLE_H)
SINGLE_QUAT = "1 0 0 0"

BLOCK_HALF = 0.025                  # box 半尺寸 -> 全高 0.05

# 三物体初始位姿（桌面上方一点点落稳），分散不堆叠，均在工作空间内
BLOCKS = [
    # name,          color material, (x, y)
    ("green_block",  "mat_green",  (0.50,  0.12)),
    ("yellow_block", "mat_yellow", (0.50, -0.12)),
    ("blue_block",   "mat_blue",   (0.58,  0.00)),
]


def _common_assets():
    """场景级材质（含 panda 用到的 white/off_white/black/green/light_blue）。"""
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
    <material name="mat_green" rgba="0.20 0.78 0.25 1"/>
    <material name="mat_blue" rgba="0.15 0.35 0.85 1"/>
    <material name="mat_yellow" rgba="0.92 0.82 0.16 1"/>"""


def _table_and_world():
    """地面 + 桌子 + 桌腿 + 灯光。"""
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
    """overview 俯视 + front 正视（两视角，可供 VLM 三角化）。"""
    # overview：从桌子前上方较高俯视物体区（聚焦三块所在 x≈0.5），看清相对位置与堆叠
    overview = ('    <camera name="overview" pos="1.05 -0.70 0.92" '
                'xyaxes="0.55 0.84 0 -0.46 0.30 0.84" fovy="42"/>')
    # front：正前方略俯视，沿 -x 看向桌中心，用作第二视角
    front = ('    <camera name="front" pos="1.45 0.0 0.78" '
             'xyaxes="0 1 0 -0.45 0 0.89" fovy="48"/>')
    return overview + "\n" + front


def _free_box(name, material, xy):
    x, y = xy
    z = TABLE_H + BLOCK_HALF + 0.002   # 底面贴桌
    return f"""    <body name="{name}" pos="{x} {y} {z}">
      <freejoint name="{name}_free"/>
      <geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" material="{material}" mass="0.05"
            friction="1.0 0.02 0.001"/>
    </body>"""


def _grasp_welds():
    """每个物体一个 inactive weld（body1=hand, body2={block}）。
    active="false"：默认不激活，运行时按抓取目标动态激活对应 weld。
    relpose 在运行时由 env._set_grasp_weld 按当前相对位姿写入 eq_data。"""
    lines = []
    for name, _, _ in BLOCKS:
        suffix = name.split("_")[0]   # green/yellow/blue
        lines.append(
            f'    <weld name="grasp_weld_{suffix}" body1="hand" body2="{name}" '
            f'active="false" solimp="0.95 0.99 0.001" solref="0.005 1"/>')
    return "\n".join(lines)


def build_blocks_scene_xml() -> str:
    arm = panda_body("", SINGLE_BASE, SINGLE_QUAT)
    tendon, equality, actuator = panda_tendon_equality_actuator("")

    objs = "\n".join(_free_box(name, mat, xy) for name, mat, xy in BLOCKS)
    world = (_table_and_world() + "\n" + _cameras() + "\n" + arm + "\n" + objs)

    return f"""<mujoco model="exp4_stage3_blocks">
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
{_grasp_welds()}
  </equality>

  <actuator>
{actuator}
  </actuator>
</mujoco>"""


def write_scene(out_path) -> str:
    xml = build_blocks_scene_xml()
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


if __name__ == "__main__":
    import mujoco
    out = "/tmp/exp4_stage3/scene_blocks.xml"
    write_scene(out)
    m = mujoco.MjModel.from_xml_path(out)
    print(f"[ok] 场景编译成功: nq={m.nq} nv={m.nv} nbody={m.nbody} neq={m.neq}")
    for name in ("green_block", "yellow_block", "blue_block"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"  body {name}: id={bid}")
    for w in ("grasp_weld_green", "grasp_weld_yellow", "grasp_weld_blue"):
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, w)
        print(f"  weld {w}: id={eid}")
