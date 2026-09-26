"""代码生成的 action：把 VLM 打点 + 反投影 + 原语串成一个技能
================================================================

这就是实验2 里 "low-level policy sampling" 的新语义：
**不是写死的状态机脚本，而是「用代码把现有原语接口串接起来」实现一个 action。**

下面 `pick(...)` 函数体本身就是「代码生成器」会产出的程序——它调用一组可组合的
接口（vlm_point / deproject / move_to / close_gripper / lift），任何 LLM 代码生成
模块都可以照此模式拼装出 pick / place / push 等技能。我们手写出这段「被生成的
代码」作为参考实现，并用它驱动 MuJoCo 验证动作模型 h=⟨pre,add,del⟩ 是否一致。

返回一个 dict，含：
- success / holding / obj_lifted
- target_px：VLM 在各相机打的点
- target_xyz：反投影得到的 3D 目标
- prim_log：原语执行日志
- traj：(obs, action) 轨迹（供录像/DP 数据）
"""

from __future__ import annotations

import numpy as np
import mujoco

from dp_common.franka_dp_env import TABLE_TOP_Z
from . import camera as C
from .primitives import PrimitiveRunner


def _make_oracle_project_fn(env, img_wh):
    """给 oracle 后端用：把物体真值「顶面中心」投影到指定相机像素。"""
    W, H = img_wh

    def fn(cam_name):
        opos, _ = env.get_object_pose()
        top = opos.copy()
        top[2] = TABLE_TOP_Z + env._cur_top  # 物体顶面
        return C.project_point(env.m, env.d, W, H, cam_name, top)
    return fn


def _robust_fuse(cands, inlier_tol=0.03):
    """从多个 3D 估计里挑互相靠近的内点求 xy 均值（z 用桌面几何另算，这里只管 xy）。
    cands: [(name, p3d(3,)), ...]。返回 (3,) 点（取内点 xy 均值 + 第一个候选的 z）。
    思路：对每个候选统计「与其它候选 xy 距离 < tol 的个数」，取支持度最高者所在的内点簇。
    单候选时直接返回；全员互不靠近时退回第一个（通常是三角化）。
    """
    pts = [np.asarray(p, float) for _, p in cands]
    if len(pts) == 1:
        return pts[0].copy()
    best_i, best_inliers = 0, [pts[0]]
    for i, pi in enumerate(pts):
        inl = [pj for pj in pts if np.linalg.norm((pi - pj)[:2]) < inlier_tol]
        if len(inl) > len(best_inliers):
            best_i, best_inliers = i, inl
    xy = np.mean([p[:2] for p in best_inliers], axis=0)
    out = pts[0].copy()       # z 占位（后续被几何 grasp_z 覆盖）
    out[0], out[1] = xy[0], xy[1]
    return out


def _point(pointer, img, instruction, cam_name):
    """统一打点入口：oracle 后端需要知道当前相机。"""
    if hasattr(pointer, "set_camera"):
        pointer.set_camera(cam_name)
    return pointer.point(img, instruction)


def pick(env, pointer, depth_renderer, img_wh,
         point_cam="overview", verify_cam="front",
         instruction="the object on the table",
         approach_h=0.12, log=None):
    """代码生成的 pick(obj) 技能。

    参数
    ----
    env            : FrankaDPEnv（已 build + reset）
    pointer        : VLM 打点后端（vlm_pointer.PointerBackend）
    depth_renderer : 已 enable_depth_rendering() 的 mujoco.Renderer
    img_wh         : (W, H) 渲染分辨率
    point_cam      : 主打点相机
    verify_cam     : 第二视角相机（交叉校验 / 可视化）

    流程（被生成的代码）
    -------------------
    1. 取主相机图 → VLM 打点 (u,v)
    2. 深度反投影 (u,v) → 3D 目标点 p3d
    3. （第二视角同样打点 + 反投影，做交叉校验，提升鲁棒性）
    4. move_to(p3d 上方) → descend → close_gripper → lift
    """
    W, H = img_wh
    log = log if log is not None else []
    runner = PrimitiveRunner(env, log=log)
    record = {"point_cam": point_cam, "verify_cam": verify_cam,
              "instruction": instruction, "target_px": {}, "deproj": {}}

    # ---- 1. 两个不同视角各自打点（至少两个相机）---- #
    img_main = env._grab_rgb(point_cam)
    uv_main = _point(pointer, img_main, instruction, point_cam)
    if uv_main is None:
        record["error"] = f"VLM 在 {point_cam} 未返回有效点"
        record["success"] = False
        return record
    record["target_px"][point_cam] = [float(uv_main[0]), float(uv_main[1])]

    img_v = env._grab_rgb(verify_cam)
    uv_v = _point(pointer, img_v, instruction, verify_cam)
    if uv_v is not None:
        record["target_px"][verify_cam] = [float(uv_v[0]), float(uv_v[1])]

    # ---- 2. 2D→3D 定位 ---- #
    # 首选：两视角视线三角化（不依赖深度缓冲，抗 GL 深度精度偏置，xy 最准）。
    # 退化：仅单视角时用深度反投影。
    # 真实 VLM 打点会有噪声，单一视角偶尔严重偏（如 front 视角细高罐被指到罐身边缘）。
    # 这里做**多估计鲁棒融合**：同时拿到 三角化点 + 两视角各自深度反投影点，
    # 取彼此互相靠近的「内点」求均值；都不一致时退回三角化。全程只用 VLM 点，不偷看真值。
    p_md = C.deproject_with_depth(env.m, env.d, depth_renderer, W, H,
                                  point_cam, uv_main[0], uv_main[1])
    if p_md is not None:
        record["deproj"][point_cam] = p_md.tolist()
    p3d = None
    if uv_v is not None:
        p_tri = C.triangulate(env.m, env.d, W, H,
                              point_cam, uv_main[0], uv_main[1],
                              verify_cam, uv_v[0], uv_v[1])
        record["triangulated"] = p_tri.tolist()
        p_vd = C.deproject_with_depth(env.m, env.d, depth_renderer, W, H,
                                      verify_cam, uv_v[0], uv_v[1])
        if p_vd is not None:
            record["deproj"][verify_cam] = p_vd.tolist()
        if p_md is not None and p_vd is not None:
            record["cross_view_disagreement"] = float(np.linalg.norm(p_md[:2] - p_vd[:2]))
        # —— 鲁棒融合：在 {三角化, 主视深度, 验证视深度} 里找互相靠近(<3cm)的最大内点集 ——
        cands = [("tri", p_tri)]
        if p_md is not None:
            cands.append(("md", p_md))
        if p_vd is not None:
            cands.append(("vd", p_vd))
        p3d = _robust_fuse(cands, inlier_tol=0.03)
        record["fused_xy_from"] = [n for n, _ in cands]
    else:
        if p_md is None:
            record["error"] = f"{point_cam} 深度反投影失败且无第二视角"
            record["success"] = False
            return record
        p3d = p_md.copy()

    # 工作空间夹取：VLM 把点打到桌外/机械臂够不到时，钳到可达范围，避免 IK 发散乱抓。
    p3d[0] = float(np.clip(p3d[0], 0.32, 0.73))
    p3d[1] = float(np.clip(p3d[1], -0.28, 0.28))

    # 抓取高度：指尖落到「物体腰部（略低于几何中心）」，保证两指跨在物体两侧。
    # 物体中心高度 = TABLE_TOP_Z + _cur_bottom；顶面 = 中心 + _cur_top。
    # 与已验证的 ScriptedExpert 同一公式：从顶面往下 0.6×全高 = 略低于中心。
    # 不依赖深度缓冲（GL 精度偏置），用已知桌面+物体几何兜底，最稳。
    obj_center_z = TABLE_TOP_Z + env._cur_bottom
    obj_top_z = obj_center_z + env._cur_top
    obj_bottom_z = obj_center_z - env._cur_bottom
    grasp_z = obj_top_z - 0.6 * (env._cur_top + env._cur_bottom)
    grasp_z = max(grasp_z, obj_bottom_z + 0.006)
    p3d[2] = grasp_z
    record["target_xyz"] = p3d.tolist()

    # ---- 4. 串接原语执行 pick（servo 目标始终是 VLM 反投影点 p3d，不偷看真值）---- #
    runner.approach(p3d, height=approach_h, grip=1.0)
    runner.open_gripper(hold=2)
    runner.descend_to(p3d, grip=1.0, tol=0.010)
    runner.align_xy(p3d, grip=1.0, horiz_tol=0.006, max_steps=25)
    runner.close_gripper_following(p3d, hold=22, grip=0.0)
    runner.lift(height=0.30, grip=0.0)

    record["success"] = bool(env.check_success())
    record["holding"] = bool(env.is_holding())
    obj_z = float(env.d.qpos[env.obj_qadr + 2])
    record["obj_z"] = obj_z
    record["obj_lifted"] = bool(obj_z > TABLE_TOP_Z + env._cur_bottom + 0.02)
    record["prim_log"] = list(log)
    record["traj"] = runner.traj
    return record
