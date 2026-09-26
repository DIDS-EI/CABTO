"""桌面待抓取物体注册表（MuJoCo 版）
=====================================
与 Genesis 版 dp_objects 对应，但物体以 **MuJoCo freejoint body 的 XML 片段**
形式注入场景（MuJoCo 模型一旦编译就不能增删 body，因此每种物体在 build 时
单独把对应片段写进 panda_grasp_base.xml 的 <!--OBJECT_BODY--> 占位符，再用
mujoco.MjModel.from_xml_string 编译）。

为降低任务难度、保证真实接触动力学下可稳定抓起，MuJoCo 版只用**几何基元**
（方块 / 圆柱 / 球），命名沿用家用小物语义，与 Genesis 版的 primitive 组一致。

每个 ObjectSpec 提供：
    name          : 唯一名字
    kind          : "primitive"
    bottom_offset : 原点到最低点距离（落桌用）
    top_offset    : 原点到最高点距离（专家算抓取高度用）
    half_width    : 横向半宽（夹爪开合参考）
    body_xml(pos, quat) : 生成 <body>…</body> 注入片段（含 freejoint + geom）

放置范围支持环境变量 DP_EASY=1 收窄到桌面正中一小块、几乎不旋转。
"""

from __future__ import annotations

import os as _os

import numpy as np


# --------------------------------------------------------------------------- #
# 桌面随机采样范围（相对桌面中心，单位 m），与 Genesis 版保持一致
# --------------------------------------------------------------------------- #
def _spawn_ranges():
    """运行时读取 DP_EASY（而非 import 时固定），保证脚本里后设的环境变量生效。"""
    if _os.environ.get("DP_EASY", "0") == "1":
        # 简单模式：物体只落在桌面正中一小块、且几乎不旋转
        return (0.53, 0.59), (-0.06, 0.06), (-0.30, 0.30)
    return (0.47, 0.63), (-0.16, 0.16), (-np.pi, np.pi)


# 向后兼容的模块级常量（import 时快照；优先用 _spawn_ranges() 取实时值）
SPAWN_X_RANGE, SPAWN_Y_RANGE, SPAWN_YAW_RANGE = _spawn_ranges()


def sample_object_pose(rng, table_top_z, bottom_offset):
    """在桌面随机采样物体位姿，使其底部贴桌面 + 2mm 间隙。

    返回 (pos(3,), quat_wxyz(4,))。pos[2] = table + bottom_offset + 间隙。
    """
    x_range, y_range, yaw_range = _spawn_ranges()   # 运行时读取，尊重后设的 DP_EASY
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    z = table_top_z + bottom_offset + 0.002
    yaw = rng.uniform(*yaw_range)
    quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)
    return np.array([x, y, z], dtype=float), quat


# --------------------------------------------------------------------------- #
# 物体条目
# --------------------------------------------------------------------------- #
def _fmt(v):
    return " ".join(f"{float(x):.6f}" for x in v)


class ObjectSpec:
    def __init__(self, name, prim_shape, scale, color, mass):
        self.name = name
        self.kind = "primitive"
        self._prim_shape = prim_shape   # "box" | "cylinder" | "sphere"
        self.scale = float(scale)
        self.color = color
        self.mass = float(mass)
        self._geom = None

    # ---- 几何：返回 (bottom_offset, top_offset, half_width) ---- #
    @property
    def geometry(self):
        if self._geom is None:
            s = self.scale
            if self._prim_shape == "box":
                self._geom = (s / 2, s / 2, s / 2)
            elif self._prim_shape == "cylinder":
                h = s * 1.2
                self._geom = (h / 2, h / 2, s * 0.5)
            elif self._prim_shape == "sphere":
                self._geom = (s * 0.5, s * 0.5, s * 0.5)
            else:
                self._geom = (s / 2, s / 2, s / 2)
        return self._geom

    @property
    def bottom_offset(self):
        return self.geometry[0]

    @property
    def top_offset(self):
        return self.geometry[1]

    @property
    def half_width(self):
        return self.geometry[2]

    # ---- 生成 MuJoCo body 注入片段 ---- #
    def _geom_attrs(self):
        """返回该形状的 type/size 属性字符串。"""
        s = self.scale
        if self._prim_shape == "box":
            half = s / 2
            return f'type="box" size="{half:.6f} {half:.6f} {half:.6f}"'
        if self._prim_shape == "cylinder":
            r = s * 0.5
            halfh = s * 1.2 / 2
            return f'type="cylinder" size="{r:.6f} {halfh:.6f}"'
        if self._prim_shape == "sphere":
            r = s * 0.5
            return f'type="sphere" size="{r:.6f}"'
        half = s / 2
        return f'type="box" size="{half:.6f} {half:.6f} {half:.6f}"'

    def body_xml(self, pos, quat):
        """生成可注入 worldbody 的 <body> 片段（含 freejoint + 带摩擦的 geom）。

        condim=4 + 较高摩擦，保证夹爪与物体之间有足够切向摩擦不打滑。
        """
        rgba = f"{self.color[0]:.3f} {self.color[1]:.3f} {self.color[2]:.3f} 1"
        return f'''<body name="object" pos="{_fmt(pos)}" quat="{_fmt(quat)}">
      <freejoint name="obj_free"/>
      <geom name="obj_geom" {self._geom_attrs()} rgba="{rgba}"
            mass="{self.mass:.4f}" condim="4" friction="1.2 0.05 0.0002"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>'''


# 几何基元（家用小物语义命名），尺寸与 Genesis primitive 组一致
_PRIMITIVES = [
    ObjectSpec("dice_cube",   "box",      0.045, (0.90, 0.35, 0.20), mass=0.05),
    ObjectSpec("soda_can",    "cylinder", 0.045, (0.85, 0.20, 0.20), mass=0.04),
    ObjectSpec("rubber_ball", "sphere",   0.050, (0.20, 0.55, 0.85), mass=0.05),
    ObjectSpec("wood_block",  "box",      0.050, (0.72, 0.55, 0.36), mass=0.06),
    ObjectSpec("tea_tin",     "cylinder", 0.050, (0.30, 0.65, 0.45), mass=0.05),
]


def get_registry(group="all"):
    # MuJoCo 版仅含 primitive；mesh 组退化为 primitive，保证接口兼容
    return list(_PRIMITIVES)


def sample_object(rng, group="all"):
    reg = get_registry(group)
    return reg[rng.integers(len(reg))]


def get_object_by_name(name):
    for spec in _PRIMITIVES:
        if spec.name == name:
            return spec
    raise KeyError(f"未知物体: {name}. 可选: {[s.name for s in _PRIMITIVES]}")
