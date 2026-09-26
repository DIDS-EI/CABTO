"""VLM 视觉打点接口（pointing）
================================

实验2「代码生成 + VLM 打点」的感知前端。给定一张相机图像和一句自然语言描述
（要打的点，例如 "the top center of the red cube to grasp"），VLM 在图上返回一个
像素点 (u, v)。该点随后交给 camera.deproject_with_depth 转成 3D 世界坐标，驱动机械臂。

后端可插拔（--backend）
-----------------------
- "qwen"  : Qwen2.5-VL（Apple Silicon 本地 mlx 推理，**首选**）。原生支持
            grounding，提示其输出 JSON {"point":[x,y]}（0-1000 归一化或像素）。
- "molmo" : Molmo（原生 pointing 训练，输出 <point x=".." y=".."> 标签）。
            mlx / transformers 后端，依赖较重，作为对比项。
- "oracle": 几何 oracle——用物体真值世界坐标投影到该相机的像素点。
            无需模型、确定性，用于打通全链路与作为 VLM 失败时的回退/对照基线。

统一返回 (u, v) 像素坐标（左上原点，u 列 v 行），坐标系与 camera.py 一致。
"""

from __future__ import annotations

import json
import re
import numpy as np


# --------------------------------------------------------------------------- #
# 抽象后端
# --------------------------------------------------------------------------- #
class PointerBackend:
    name = "base"

    def point(self, image_rgb: np.ndarray, instruction: str) -> tuple[float, float] | None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Oracle：几何真值投影（确定性，用于打通链路 / 对照基线）
# --------------------------------------------------------------------------- #
class OraclePointer(PointerBackend):
    name = "oracle"

    def __init__(self, project_fn):
        """project_fn(cam_name) -> (u, v)：把目标真值投影到指定相机的像素。
        codegen_action 在每次打点前用 set_camera() 告知当前相机。
        """
        self._project_fn = project_fn
        self._cam = None

    def set_camera(self, cam_name):
        self._cam = cam_name

    def point(self, image_rgb, instruction):
        return self._project_fn(self._cam)


# --------------------------------------------------------------------------- #
# Qwen2.5-VL（mlx 本地推理）
# --------------------------------------------------------------------------- #
class QwenVLPointer(PointerBackend):
    name = "qwen"

    # 标定结论（exp2_codegen/results/qwen_prompt_calib.json）：
    # Qwen2.5-VL 对**简洁**的 pointing 指令最准（median ~20px @512 ≈ 1cm）；
    # 一旦把 prompt 写啰嗦、强加 JSON schema 或出现 "grasp" 等词，误差显著变大。
    # 它原生输出形如 ```json [{"point_2d":[x,y], "label":...}] ```，绝对像素，
    # 无需归一化。因此这里用最简指令，解析交给 _parse_point 兼容多种 key。
    POINT_PROMPT = "Point to {instr}. Output the pixel coordinate."

    def __init__(self, model_path="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                 max_tokens=128, temperature=0.0):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self._load = load
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_path = model_path

    def point(self, image_rgb, instruction):
        import tempfile, os
        import imageio.v2 as imageio
        from mlx_vlm import generate, apply_chat_template
        h, w = image_rgb.shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        imageio.imwrite(tmp, image_rgb)
        try:
            prompt_txt = self.POINT_PROMPT.format(w=w, h=h, instr=instruction)
            prompt = apply_chat_template(self.processor, self.config, prompt_txt,
                                         num_images=1)
            out = generate(self.model, self.processor, prompt, image=[tmp],
                           max_tokens=self.max_tokens, temperature=self.temperature,
                           verbose=False)
            text = out.text if hasattr(out, "text") else str(out)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return _parse_point(text, w, h)


# --------------------------------------------------------------------------- #
# Molmo（mlx 本地推理，原生 pointing）
# --------------------------------------------------------------------------- #
class MolmoPointer(PointerBackend):
    name = "molmo"

    def __init__(self, model_path="mlx-community/Molmo-7B-D-0924-4bit",
                 max_tokens=128):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        self.max_tokens = max_tokens
        self.model_path = model_path

    def point(self, image_rgb, instruction):
        import tempfile, os
        import imageio.v2 as imageio
        from mlx_vlm import generate, apply_chat_template
        h, w = image_rgb.shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        imageio.imwrite(tmp, image_rgb)
        try:
            # Molmo 原生 pointing 指令
            prompt_txt = f"Point to {instruction}"
            prompt = apply_chat_template(self.processor, self.config, prompt_txt,
                                         num_images=1)
            out = generate(self.model, self.processor, prompt, image=[tmp],
                           max_tokens=self.max_tokens, verbose=False)
            text = out.text if hasattr(out, "text") else str(out)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        # Molmo 输出 <point x="34.5" y="50.0" .../>，x/y 是 0-100 百分比
        m = re.search(r'x="?([\d.]+)"?\s+y="?([\d.]+)"?', text)
        if m:
            x = float(m.group(1)) / 100.0 * w
            y = float(m.group(2)) / 100.0 * h
            return x, y
        return _parse_point(text, w, h)


# --------------------------------------------------------------------------- #
# 解析工具：从模型输出文本里抠出 (u, v)，兼容多种格式
# --------------------------------------------------------------------------- #
def _parse_point(text: str, w: int, h: int):
    """从 Qwen/通用 VLM 输出里抠出像素点 (u, v)，兼容实测的多种格式：
      - ```json [{"point_2d":[x,y], "label":...}] ```   (Qwen2.5-VL 最常见)
      - {"point_3d":[x,y]} / {"point":[x,y]} / {"point_2d":[x,y]}
      - 裸坐标 [x, y] 或 [x1,y1,x2,y2]（bbox 取中心）
      - Molmo <points x1=".." y1=".."> 由各后端自行处理，这里兜底数字提取。
    Qwen2.5-VL 输出为输入图分辨率下的**绝对像素**，无需归一化。
    """
    if not text:
        return None
    # 1) 直接抓 "point" 系列 key 后面的两个数字（point / point_2d / point_3d）
    mkey = re.search(r'"point(?:_2d|_3d)?"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)', text)
    if mkey:
        return _maybe_denorm(float(mkey.group(1)), float(mkey.group(2)), w, h)
    # 2) 尝试解析 JSON（去掉 markdown code fence），支持 list/dict 嵌套
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
    # 3) 兜底：抓第一个 [a, b] 数对；或 4 数 bbox 取中心
    mpair = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*,\s*([\d.]+))?\s*\]', text)
    if mpair:
        x1, y1 = float(mpair.group(1)), float(mpair.group(2))
        if mpair.group(3) is not None:  # bbox
            x2, y2 = float(mpair.group(3)), float(mpair.group(4))
            return _maybe_denorm((x1 + x2) / 2, (y1 + y2) / 2, w, h)
        return _maybe_denorm(x1, y1, w, h)
    nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]
    if len(nums) >= 2:
        return _maybe_denorm(nums[0], nums[1], w, h)
    return None


def _dig_point(obj):
    """递归在 dict/list 里找 point/point_2d/point_3d 或第一个 2 元数对。"""
    if isinstance(obj, dict):
        for k in ("point_2d", "point", "point_3d", "coordinate", "xy"):
            if k in obj and isinstance(obj[k], (list, tuple)) and len(obj[k]) >= 2:
                return float(obj[k][0]), float(obj[k][1])
        for v in obj.values():
            r = _dig_point(v)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        # [x, y] 数对
        if len(obj) >= 2 and all(isinstance(e, (int, float)) for e in obj[:2]):
            return float(obj[0]), float(obj[1])
        for e in obj:
            r = _dig_point(e)
            if r is not None:
                return r
    return None


def _first_bracket_block(s):
    """提取第一个完整的 [...] 或 {...} 文本块（应对尾部被截断的输出）。"""
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


def _maybe_denorm(x, y, w, h):
    """若坐标看起来是 0-1000 归一化（Qwen 习惯），按图像尺寸还原到像素。"""
    if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > w or y > h):
        x = x / 1000.0 * w
        y = y / 1000.0 * h
    return float(x), float(y)


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def make_pointer(backend: str, **kwargs) -> PointerBackend:
    backend = backend.lower()
    if backend == "oracle":
        return OraclePointer(kwargs["project_fn"])
    if backend == "qwen":
        return QwenVLPointer(**{k: v for k, v in kwargs.items()
                                if k in ("model_path", "max_tokens", "temperature")})
    if backend == "molmo":
        return MolmoPointer(**{k: v for k, v in kwargs.items()
                               if k in ("model_path", "max_tokens")})
    raise ValueError(f"未知 VLM 后端: {backend}")
