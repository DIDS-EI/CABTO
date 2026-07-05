"""A. VLM 打点 + 2D→3D 定位（stage3 自含轻量版）
====================================================
对应 CABTO 执行层的感知前端：VLM 在 overview 图上对目标物体「打点」得像素
(u,v)，再用相机深度反投影成 3D 世界坐标，交给原语驱动机械臂。

- 相机几何（内参/外参/反投影/三角化）内联在本文件，独立于 exp2/camera.py。
- Pointer(backend="qwen"|"oracle")：
    point_2d(image, instr) -> (u,v)        VLM 打点 + 鲁棒解析
    locate_3d(instr_or_obj) -> np.ndarray  深度反投影；失败退化两视角 triangulate
    locate_3d_oracle(obj)   -> np.ndarray  直接 env 真值（对照/兜底）
  backend="oracle" 时 locate_3d 直接走真值，便于先把 ②③⑤ 跑通再验证 ④。
"""

from __future__ import annotations

import json
import re

import numpy as np
import mujoco

from blocks_env import BLOCK_HALF


# --------------------------------------------------------------------------- #
# 相机几何（内联，自含）
# --------------------------------------------------------------------------- #
def _cam_intrinsics(model, img_w, img_h, cam_name):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = float(model.cam_fovy[cid])
    f = (img_h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    return f, f, img_w / 2.0, img_h / 2.0


def _cam_extrinsics(data, model, cam_name):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    t = data.cam_xpos[cid].copy()
    Rm = data.cam_xmat[cid].reshape(3, 3).copy()
    return Rm, t


def _pixel_ray(model, data, img_w, img_h, cam_name, u, v):
    fx, fy, cx, cy = _cam_intrinsics(model, img_w, img_h, cam_name)
    Rm, t = _cam_extrinsics(data, model, cam_name)
    x = (u - cx) / fx
    y = -(v - cy) / fy
    d_cam = np.array([x, y, -1.0])
    d_world = Rm @ d_cam
    d_world = d_world / (np.linalg.norm(d_world) + 1e-9)
    return t, d_world


def _deproject_with_depth(model, data, depth, img_w, img_h, cam_name, u, v, win=2):
    """用已渲染好的深度图 depth(H,W) 把像素 (u,v) 反投影到世界 3D。无效返回 None。"""
    H, W = depth.shape[:2]
    uu = max(0, min(W - 1, int(round(u))))
    vv = max(0, min(H - 1, int(round(v))))
    lo_v, hi_v = max(0, vv - win), min(H, vv + win + 1)
    lo_u, hi_u = max(0, uu - win), min(W, uu + win + 1)
    patch = depth[lo_v:hi_v, lo_u:hi_u].reshape(-1)
    patch = patch[(patch > 1e-4) & (patch < 50.0)]
    if patch.size == 0:
        return None
    z = float(np.median(patch))
    fx, fy, cx, cy = _cam_intrinsics(model, img_w, img_h, cam_name)
    Rm, t = _cam_extrinsics(data, model, cam_name)
    x = (uu - cx) / fx * z
    y = -(vv - cy) / fy * z
    p_cam = np.array([x, y, -z])
    return Rm @ p_cam + t


def _triangulate(model, data, img_w, img_h, cam_a, ua, va, cam_b, ub, vb):
    o1, d1 = _pixel_ray(model, data, img_w, img_h, cam_a, ua, va)
    o2, d2 = _pixel_ray(model, data, img_w, img_h, cam_b, ub, vb)
    d1d1, d2d2, d1d2 = d1 @ d1, d2 @ d2, d1 @ d2
    r = o1 - o2
    denom = d1d1 * d2d2 - d1d2 * d1d2
    if abs(denom) < 1e-9:
        return 0.5 * (o1 + o2)
    s = (d1d2 * (d2 @ r) - d2d2 * (d1 @ r)) / denom
    t = (d1d1 * (d2 @ r) - d1d2 * (d1 @ r)) / denom
    return 0.5 * ((o1 + s * d1) + (o2 + t * d2))


# --------------------------------------------------------------------------- #
# VLM 打点解析（照搬 exp2 _parse_point 思路，兼容多格式）
# --------------------------------------------------------------------------- #
def _maybe_denorm(x, y, w, h):
    if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > w or y > h):
        x = x / 1000.0 * w
        y = y / 1000.0 * h
    return float(x), float(y)


def _first_bracket_block(s):
    for op, cl in (("[", "]"), ("{", "}")):
        i = s.find(op)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == op:
                depth += 1
            elif s[j] == cl:
                depth -= 1
                if depth == 0:
                    return s[i:j + 1]
    return None


def _dig_point(obj):
    if isinstance(obj, dict):
        for k in ("point_2d", "point", "point_3d", "coordinate", "xy"):
            if k in obj and isinstance(obj[k], (list, tuple)) and len(obj[k]) >= 2:
                return float(obj[k][0]), float(obj[k][1])
        for v in obj.values():
            r = _dig_point(v)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(isinstance(e, (int, float)) for e in obj[:2]):
            return float(obj[0]), float(obj[1])
        for e in obj:
            r = _dig_point(e)
            if r is not None:
                return r
    return None


def parse_point(text, w, h):
    """从 VLM 输出抠出像素 (u,v)，兼容 point_2d/point/裸数对/bbox/0-1000 归一化。"""
    if not text:
        return None
    mkey = re.search(r'"point(?:_2d|_3d)?"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)', text)
    if mkey:
        return _maybe_denorm(float(mkey.group(1)), float(mkey.group(2)), w, h)
    cleaned = re.sub(r'```(?:json)?|```', '', text).strip()
    for cand in (cleaned, _first_bracket_block(cleaned)):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        pt = _dig_point(obj)
        if pt is not None:
            return _maybe_denorm(pt[0], pt[1], w, h)
    mpair = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*,\s*([\d.]+))?\s*\]', text)
    if mpair:
        x1, y1 = float(mpair.group(1)), float(mpair.group(2))
        if mpair.group(3) is not None:
            x2, y2 = float(mpair.group(3)), float(mpair.group(4))
            return _maybe_denorm((x1 + x2) / 2, (y1 + y2) / 2, w, h)
        return _maybe_denorm(x1, y1, w, h)
    nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]
    if len(nums) >= 2:
        return _maybe_denorm(nums[0], nums[1], w, h)
    return None


# 物体 → 自然语言短语（给 VLM 打点用）
def obj_phrase(name):
    color = name.split("_")[0]
    return f"the {color} block"


# --------------------------------------------------------------------------- #
# Pointer
# --------------------------------------------------------------------------- #
class Pointer:
    POINT_PROMPT = "Point to {instr}. Output the pixel coordinate."

    def __init__(self, vlm_backend, env, backend="qwen"):
        self.vlm = vlm_backend
        self.env = env
        self.backend = backend.lower()

    # ---- 2D 打点 ---- #
    def point_2d(self, image, instruction):
        h, w = image.shape[:2]
        prompt = self.POINT_PROMPT.format(instr=instruction)
        text = self.vlm.chat_vision(prompt, [image], max_tokens=128, temperature=0.0)
        pt = parse_point(text, w, h)
        return pt  # (u,v) 或 None

    # ---- 真值定位（oracle / 兜底）：物体顶面中心 ---- #
    def locate_3d_oracle(self, obj_name):
        pos, _ = self.env.get_object_pose(obj_name)
        p = pos.copy()
        p[2] = pos[2] + BLOCK_HALF  # 顶面
        return np.asarray(p, float)

    # ---- VLM 打点 → 3D ---- #
    def locate_3d(self, instruction, obj_name=None):
        """instruction 用于 VLM 打点的自然语言；obj_name 给 oracle/兜底用真值。
        backend='oracle' 直接返回真值。否则 VLM 打点 + 深度反投影，失败退化 triangulate。"""
        if obj_name is None:
            obj_name = _guess_obj_from_instr(instruction)
        if self.backend == "oracle":
            return self.locate_3d_oracle(obj_name)

        env = self.env
        img = env.render_cam("overview")
        h, w = img.shape[:2]
        uv = self.point_2d(img, instruction)
        if uv is None:
            print(f"[Pointer] VLM 未给出像素点，退化 oracle: {obj_name}")
            return self.locate_3d_oracle(obj_name)
        u, v = uv

        # 1) 深度反投影（overview 深度）
        depth = env.render_depth("overview")
        p = _deproject_with_depth(env.m, env.d, depth, w, h, "overview", u, v)
        if p is not None:
            # 反投影命中的是物体顶面附近；抬升半块高度以贴近顶面中心
            self._last_uv = (u, v)
            self._last_src = "depth"
            return np.asarray(p, float)

        # 2) 深度无效：用 overview + front 两视角 triangulate
        img2 = env.render_cam("front")
        uv2 = self.point_2d(img2, instruction)
        if uv2 is not None:
            p = _triangulate(env.m, env.d, w, h, "overview", u, v,
                             "front", uv2[0], uv2[1])
            self._last_uv = (u, v)
            self._last_src = "triangulate"
            return np.asarray(p, float)

        print(f"[Pointer] 深度+三角化均失败，退化 oracle: {obj_name}")
        return self.locate_3d_oracle(obj_name)


def _guess_obj_from_instr(instr):
    s = instr.lower()
    for c in ("green", "yellow", "blue"):
        if c in s:
            return f"{c}_block"
    return "green_block"


if __name__ == "__main__":
    from blocks_env import BlocksEnv
    env = BlocksEnv(render=True)
    env.reset()
    ptr = Pointer(None, env, backend="oracle")
    for o in ("green_block", "yellow_block", "blue_block"):
        print(o, ptr.locate_3d_oracle(o), "| locate_3d:", ptr.locate_3d(obj_phrase(o), o))
