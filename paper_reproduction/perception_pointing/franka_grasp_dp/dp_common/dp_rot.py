"""旋转 / 位姿数学工具
====================
Diffusion Policy 采集的动作是 **末端 6D 位姿的 delta**：
    action = [dx, dy, dz,  d_rx, d_ry, d_rz,  grip]
其中：
- (dx, dy, dz)        : 末端平移增量（世界坐标，单位 m）
- (d_rx, d_ry, d_rz)  : 末端姿态增量，用 **轴角(rotation vector)** 表示（单位 rad）
- grip                : 夹爪目标开度（0=完全闭合, 1=完全张开），不是 delta

本模块提供：
- quat <-> 旋转矩阵 <-> 轴角(rotvec) 的互转
- 由当前姿态 + 姿态 delta 合成下一个目标姿态
- (可选) 6D 连续旋转表示，便于神经网络回归（这里动作直接用 rotvec，更紧凑）

约定：四元数统一用 (w, x, y, z) 顺序在对外接口暴露（与 MuJoCo 一致），
内部转 scipy 时按 (x, y, z, w) 处理。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
# 四元数顺序转换：MuJoCo/Genesis 用 (w, x, y, z)，scipy 用 (x, y, z, w)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


# --------------------------------------------------------------------------- #
# 四元数 <-> 轴角(rotvec)
# --------------------------------------------------------------------------- #
def quat_to_rotvec(q_wxyz: np.ndarray) -> np.ndarray:
    """四元数(w,x,y,z) -> 轴角向量(3,)，模长为旋转角(rad)。"""
    r = R.from_quat(quat_wxyz_to_xyzw(q_wxyz))
    return r.as_rotvec()


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """轴角向量(3,) -> 四元数(w,x,y,z)。"""
    r = R.from_rotvec(np.asarray(rotvec, dtype=float).reshape(3))
    return quat_xyzw_to_wxyz(r.as_quat())


# --------------------------------------------------------------------------- #
# 姿态增量合成
# --------------------------------------------------------------------------- #
def delta_rotvec_between(q_from_wxyz: np.ndarray, q_to_wxyz: np.ndarray) -> np.ndarray:
    """计算从 q_from 旋转到 q_to 所需的相对旋转，用轴角表示（世界坐标系左乘）。

    满足:  q_to = compose(delta, q_from)
    即  R_to = R_delta @ R_from  ->  R_delta = R_to @ R_from^{-1}
    """
    r_from = R.from_quat(quat_wxyz_to_xyzw(q_from_wxyz))
    r_to = R.from_quat(quat_wxyz_to_xyzw(q_to_wxyz))
    r_delta = r_to * r_from.inv()
    return r_delta.as_rotvec()


def apply_delta_rotvec(q_wxyz: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """在世界坐标系下，对姿态 q 施加一个轴角增量，返回新的四元数(w,x,y,z)。

    R_new = R_delta @ R_old
    """
    r_old = R.from_quat(quat_wxyz_to_xyzw(q_wxyz))
    r_delta = R.from_rotvec(np.asarray(delta_rotvec, dtype=float).reshape(3))
    r_new = r_delta * r_old
    return quat_xyzw_to_wxyz(r_new.as_quat())


def clip_rotvec(rotvec: np.ndarray, max_angle: float) -> np.ndarray:
    """限制单步姿态增量的最大旋转角，避免 IK 跳变。"""
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-9 or angle <= max_angle:
        return rotvec
    return rotvec * (max_angle / angle)


# --------------------------------------------------------------------------- #
# 6D 连续旋转表示（可选，留作扩展；当前动作用 rotvec）
# --------------------------------------------------------------------------- #
def quat_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    """四元数 -> 6D 表示（旋转矩阵前两列），适合网络回归。"""
    m = R.from_quat(quat_wxyz_to_xyzw(q_wxyz)).as_matrix()
    return m[:, :2].reshape(6)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """6D 表示 -> 四元数（Gram-Schmidt 正交化）。"""
    x = np.asarray(rot6d, dtype=float).reshape(3, 2)
    a1, a2 = x[:, 0], x[:, 1]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    b3 = np.cross(b1, b2)
    m = np.stack([b1, b2, b3], axis=1)
    return quat_xyzw_to_wxyz(R.from_matrix(m).as_quat())
