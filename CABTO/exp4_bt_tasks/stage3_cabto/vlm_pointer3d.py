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

from blocks_env import BLOCK_HALF, TABLE_TOP_Z


# --------------------------------------------------------------------------- #
# 落点校验（on-surface check）
# --------------------------------------------------------------------------- #
# 为什么需要：VLM 打点可能偏出目标物体、落到桌面上。此时深度图仍然"有效"，
# 反投影会忠实地把那个桌面点还原成 3D —— 得到一个几何上正确、语义上错误的坐标。
# 实测案例（见 outputs/xbias_diagnosis.md）：VLM 点偏 39 px 落到桌面，
# 深度读到 0.9900（桌面）而非 0.9065（块顶面），反投影 z=0.4006 ≈ 桌面高度 0.40，
# 导致 place_on 把物体放到目标块之外，任务失败。
#
# 判据：块顶面 z = TABLE_TOP_Z + BLOCK_HALF = 0.4250，桌面 z = 0.40，相差 25 mm。
# 取中点作阈值，信号充足。已在 diag_xbias.py PART 3 实测 5/5 判定正确。
ON_SURFACE_Z_MIN = TABLE_TOP_Z + BLOCK_HALF * 0.5      # 0.4125


def is_on_object_top(p, z_min=ON_SURFACE_Z_MIN):
    """判断反投影点是否落在物体顶面上（而非桌面/物体外）。

    p: 反投影得到的世界 3D 坐标。返回 (ok, reason)。
    注意这是一个**不依赖模型意见**的几何判据——深度图不会撒谎，
    与本项目"世界真值判据兜底"的思路一致。
    """
    if p is None:
        return False, "反投影失败"
    z = float(p[2])
    if z <= z_min:
        return False, ("落点 z=%.4f <= 阈值 %.4f，判定点在桌面/物体外"
                       "（块顶面应约 %.4f）" % (z, z_min, TABLE_TOP_Z + BLOCK_HALF))
    return True, "落点在物体顶面 (z=%.4f)" % z


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
    def locate_3d(self, instruction, obj_name=None, retry_point=1):
        """instruction 用于 VLM 打点的自然语言；obj_name 给 oracle/兜底用真值。
        backend='oracle' 直接返回真值。否则 VLM 打点 + 深度反投影，
        **经落点校验**后返回；校验不过则依次退化：重新打点 → 两视角三角化 → oracle 真值。

        retry_point: 落点校验失败时，允许 VLM 重新打点的次数。
        """
        if obj_name is None:
            obj_name = _guess_obj_from_instr(instruction)
        if self.backend == "oracle":
            return self.locate_3d_oracle(obj_name)

        env = self.env
        img = env.render_cam("overview")
        h, w = img.shape[:2]
        depth = env.render_depth("overview")
        self.last_check = None

        # 1) 深度反投影 + 落点校验（允许重新打点若干次）
        for attempt in range(retry_point + 1):
            uv = self.point_2d(img, instruction)
            if uv is None:
                print("[Pointer] VLM 未给出像素点 (attempt %d)" % attempt)
                continue
            u, v = uv
            p = _deproject_with_depth(env.m, env.d, depth, w, h, "overview", u, v)
            ok, reason = is_on_object_top(p)
            self.last_check = dict(attempt=attempt, uv=(u, v), ok=ok, reason=reason,
                                   p=None if p is None else list(map(float, p)))
            if ok:
                self._last_uv = (u, v)
                self._last_src = "depth"
                return np.asarray(p, float)
            print("[Pointer] 落点校验未通过 (attempt %d, uv=(%.1f,%.1f))：%s"
                  % (attempt, u, v, reason))

        # 2) 深度路径不可信：用 overview + front 两视角 triangulate
        uv = self._last_uv if getattr(self, "_last_uv", None) else None
        if uv is None and self.last_check and self.last_check.get("uv"):
            uv = self.last_check["uv"]
        if uv is not None:
            img2 = env.render_cam("front")
            uv2 = self.point_2d(img2, instruction)
            if uv2 is not None:
                p = _triangulate(env.m, env.d, w, h, "overview", uv[0], uv[1],
                                 "front", uv2[0], uv2[1])
                ok, reason = is_on_object_top(p)
                if ok:
                    self._last_src = "triangulate"
                    print("[Pointer] 退化三角化通过校验：%s" % reason)
                    return np.asarray(p, float)
                print("[Pointer] 三角化结果同样未通过校验：%s" % reason)

        # 3) 全部失败：退化 oracle 真值
        print("[Pointer] 深度+三角化均未通过落点校验，退化 oracle: %s" % obj_name)
        self._last_src = "oracle_fallback"
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
