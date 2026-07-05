"""相机几何工具：MuJoCo 相机内外参 + 2D 像素 → 3D 世界点反投影。
==================================================================

实验2（代码生成 + VLM 打点）的核心定位模块。VLM 在某个相机图上返回一个像素
点 (u, v)，本模块负责把它转成机械臂可用的三维世界坐标 p_world。

提供两条反投影路径：
1. deproject_with_depth(cam, u, v)
   - 渲染该相机的深度图，读出 (u,v) 处的真实深度，按针孔模型反投影到相机系，
     再用相机外参变换到世界系。**单相机即可，最稳。**
2. triangulate(cam_a, ua, va, cam_b, ub, vb)
   - 两个相机各出一条视线射线，最小二乘求两射线最近点。**无深度时的备选**，
     也用于「至少两个不同视角」的交叉校验。

约定
----
- 像素坐标 (u, v)：u 为列(x，向右)，v 为行(y，向下)，原点在图像左上角。
- 内参用 fovy（垂直视场角，度）。MuJoCo 相机默认方形像素，
  fx = fy = (H/2) / tan(fovy/2)，cx = W/2, cy = H/2。
- 相机外参：用 d.cam_xpos / d.cam_xmat（相机在世界系的位姿）。
  MuJoCo 相机坐标系：+x 右，+y 上，**-z 朝向场景**（OpenGL 约定）。
"""

from __future__ import annotations

import numpy as np
import mujoco


def camera_intrinsics(model, img_w, img_h, cam_name):
    """返回针孔内参 (fx, fy, cx, cy)。MuJoCo 用 fovy 定义垂直视场。"""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = float(model.cam_fovy[cid])  # 度
    f = (img_h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy = f
    cx = img_w / 2.0
    cy = img_h / 2.0
    return fx, fy, cx, cy


def camera_extrinsics(data, model, cam_name):
    """返回相机在世界系的位姿 (R(3,3), t(3,))。
    R 的列是相机坐标轴在世界系的方向：R[:,0]=+x右, R[:,1]=+y上, R[:,2]=+z(背向场景)。
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    t = data.cam_xpos[cid].copy()
    R = data.cam_xmat[cid].reshape(3, 3).copy()
    return R, t


def pixel_ray(model, data, img_w, img_h, cam_name, u, v):
    """像素 (u,v) → 世界系视线射线 (origin(3,), dir(3,) 单位向量)。"""
    fx, fy, cx, cy = camera_intrinsics(model, img_w, img_h, cam_name)
    R, t = camera_extrinsics(data, model, cam_name)
    # 相机系方向（OpenGL：+x右 +y上 -z前）。像素 v 向下，故 y 取负。
    x = (u - cx) / fx
    y = -(v - cy) / fy
    d_cam = np.array([x, y, -1.0])
    d_world = R @ d_cam
    d_world = d_world / (np.linalg.norm(d_world) + 1e-9)
    return t, d_world


def deproject_with_depth(model, data, renderer, img_w, img_h, cam_name, u, v,
                         depth_win=2):
    """用深度图把像素 (u,v) 反投影到世界系 3D 点。

    renderer 必须已 enable_depth_rendering()。取 (u,v) 邻域 depth_win 半径内
    的有效深度中位数，抗噪。返回 world point (3,) 或 None（深度无效）。
    """
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render()  # (H, W) float32，单位 m，相机系沿 -z 的距离
    H, W = depth.shape[:2]
    uu = int(round(u)); vv = int(round(v))
    uu = max(0, min(W - 1, uu)); vv = max(0, min(H - 1, vv))
    lo_v = max(0, vv - depth_win); hi_v = min(H, vv + depth_win + 1)
    lo_u = max(0, uu - depth_win); hi_u = min(W, uu + depth_win + 1)
    patch = depth[lo_v:hi_v, lo_u:hi_u].reshape(-1)
    patch = patch[(patch > 1e-4) & (patch < 50.0)]
    if patch.size == 0:
        return None
    z = float(np.median(patch))

    fx, fy, cx, cy = camera_intrinsics(model, img_w, img_h, cam_name)
    R, t = camera_extrinsics(data, model, cam_name)
    # 相机系点：x右 y上 z= -depth（场景在 -z）
    x = (uu - cx) / fx * z
    y = -(vv - cy) / fy * z
    p_cam = np.array([x, y, -z])
    p_world = R @ p_cam + t
    return p_world


def triangulate(model, data, img_w, img_h, cam_a, ua, va, cam_b, ub, vb):
    """两视角射线三角化：求两条视线的最近公共点（最小二乘）。
    无深度时的备选定位，也用于多视角交叉校验。
    """
    o1, d1 = pixel_ray(model, data, img_w, img_h, cam_a, ua, va)
    o2, d2 = pixel_ray(model, data, img_w, img_h, cam_b, ub, vb)
    # 解 min || (o1 + s d1) - (o2 + t d2) ||^2
    d1d1 = d1 @ d1
    d2d2 = d2 @ d2
    d1d2 = d1 @ d2
    r = o1 - o2
    denom = d1d1 * d2d2 - d1d2 * d1d2
    if abs(denom) < 1e-9:
        return 0.5 * (o1 + o2)
    s = (d1d2 * (d2 @ r) - d2d2 * (d1 @ r)) / denom
    t = (d1d1 * (d2 @ r) - d1d2 * (d1 @ r)) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return 0.5 * (p1 + p2)


def project_point(model, data, img_w, img_h, cam_name, p_world):
    """世界点 → 像素 (u,v)（用于在图上画 ground-truth 点 / 叠加可视化）。"""
    fx, fy, cx, cy = camera_intrinsics(model, img_w, img_h, cam_name)
    R, t = camera_extrinsics(data, model, cam_name)
    p_cam = R.T @ (np.asarray(p_world) - t)  # 世界→相机
    x, y, z = p_cam
    if z >= -1e-6:  # 在相机背后
        return None
    u = cx + fx * (x / -z)
    v = cy - fy * (y / -z)
    return float(u), float(v)
