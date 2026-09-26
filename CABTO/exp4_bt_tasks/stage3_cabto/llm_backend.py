"""可插拔 LLM/VLM 后端
======================

exp4 阶段3 闭环里 ①规划 ②codegen ④VLM校验 ⑤反馈 都要调"大模型"。本模块把
"模型后端"抽象成统一接口，方便在 **本地 Qwen2.5-VL（mlx-vlm，零依赖）** 与
**云端 OpenAI 兼容 API（填 key 即用）** 之间切换。

两类能力：
  - chat(messages)          : 纯文本对话（规划 / codegen / 反馈）。
  - chat_vision(text, imgs) : 带图对话（VLM 看 before/after 校验效应）。

后端
----
- LocalQwenVL : 本地 mlx-vlm 加载 mlx-community/Qwen2.5-VL-3B-Instruct-4bit，
                文本和带图都用同一个多模态模型。零 API 依赖，Apple Silicon 本地跑。
- CloudOpenAI : 预留——OpenAI 兼容 /v1/chat/completions（deepseek/通义/openai/claude
                兼容网关均可）。需要 base_url + api_key + model。当前未配置 key，
                实例化时会明确报错提示，但接口已就位，填上即用。

环境变量（可选，用于一键切云端）：
  EXP4_LLM_BACKEND = local | cloud         （默认 local）
  EXP4_LLM_BASE_URL / EXP4_LLM_API_KEY / EXP4_LLM_MODEL
  EXP4_VLM_BACKEND = local                  （VLM 校验默认强制本地，要看图）
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


# --------------------------------------------------------------------------- #
# 抽象后端
# --------------------------------------------------------------------------- #
class LLMBackend:
    name = "base"

    def chat(self, messages: list[dict], max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        """纯文本对话。messages = [{"role": "system/user/assistant", "content": str}]。"""
        raise NotImplementedError

    def chat_vision(self, text: str, images: list[np.ndarray],
                    max_tokens: int = 512, temperature: float = 0.0) -> str:
        """带图对话。images 为 HxWx3 uint8 RGB 数组列表。"""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 本地 Qwen2.5-VL（mlx-vlm）—— 文本 + 视觉同一模型
# --------------------------------------------------------------------------- #
class LocalQwenVL(LLMBackend):
    name = "local_qwen_vl"

    _shared = None   # 进程内单例缓存（模型很大，避免重复加载）

    def __init__(self, model_path: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"):
        self.model_path = model_path
        if LocalQwenVL._shared is None:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            model, processor = load(model_path)
            config = load_config(model_path)
            LocalQwenVL._shared = (model, processor, config)
        self.model, self.processor, self.config = LocalQwenVL._shared

    # ----- 文本：保留 system/user/assistant 角色与反馈轮次，不扁平化 ----- #
    def chat(self, messages, max_tokens=1024, temperature=0.0):
        from mlx_vlm import generate, apply_chat_template
        # Installed mlx-vlm supports a list of role/content messages. Flattening
        # history into one user message hides the actual correction turn structure.
        prompt = apply_chat_template(self.processor, self.config, messages,
                                     num_images=0)
        out = generate(self.model, self.processor, prompt,
                       max_tokens=max_tokens, temperature=temperature,
                       verbose=False)
        return out.text if hasattr(out, "text") else str(out)

    # ----- 带图 ----- #
    def chat_vision(self, text, images, max_tokens=512, temperature=0.0):
        import imageio.v2 as imageio
        from mlx_vlm import generate, apply_chat_template
        tmps = []
        try:
            for im in images:
                tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tf.close()
                imageio.imwrite(tf.name, np.asarray(im, dtype=np.uint8))
                tmps.append(tf.name)
            prompt = apply_chat_template(self.processor, self.config, text,
                                         num_images=len(tmps))
            out = generate(self.model, self.processor, prompt, image=tmps,
                           max_tokens=max_tokens, temperature=temperature,
                           verbose=False)
            return out.text if hasattr(out, "text") else str(out)
        finally:
            for p in tmps:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# 云端 OpenAI 兼容（预留接口，填 key 即用）
# --------------------------------------------------------------------------- #
class CloudOpenAI(LLMBackend):
    name = "cloud_openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.base_url = base_url or os.environ.get("EXP4_LLM_BASE_URL")
        self.api_key = api_key or os.environ.get("EXP4_LLM_API_KEY")
        self.model = model or os.environ.get("EXP4_LLM_MODEL")
        if not (self.base_url and self.api_key and self.model):
            raise RuntimeError(
                "云端 LLM 后端未配置。请设置环境变量 EXP4_LLM_BASE_URL / "
                "EXP4_LLM_API_KEY / EXP4_LLM_MODEL，或在构造时传入。"
                "接口已就位，填上凭证即可切换到云端强模型做规划/codegen。"
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "需要 openai 包：pip install openai。云端后端依赖 OpenAI 兼容客户端。"
            ) from e
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages, max_tokens=1024, temperature=0.0):
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content

    def chat_vision(self, text, images, max_tokens=512, temperature=0.0):
        # OpenAI 兼容多模态：image_url + base64。多数兼容网关支持。
        import base64
        import io
        import imageio.v2 as imageio
        content = [{"type": "text", "text": text}]
        for im in images:
            buf = io.BytesIO()
            imageio.imwrite(buf, np.asarray(im, dtype=np.uint8), format="png")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def make_llm_backend(kind: str | None = None, **kwargs) -> LLMBackend:
    """规划/codegen 用的 LLM 后端。kind=None 时读环境变量 EXP4_LLM_BACKEND（默认 local）。"""
    kind = (kind or os.environ.get("EXP4_LLM_BACKEND", "local")).lower()
    if kind in ("local", "local_qwen", "qwen"):
        return LocalQwenVL(**{k: v for k, v in kwargs.items() if k == "model_path"})
    if kind in ("cloud", "openai", "cloud_openai"):
        return CloudOpenAI(**{k: v for k, v in kwargs.items()
                              if k in ("base_url", "api_key", "model")})
    raise ValueError(f"未知 LLM 后端: {kind}")


def make_vlm_backend(kind: str | None = None, **kwargs) -> LLMBackend:
    """VLM 校验用的后端。默认本地（要看图，本地 Qwen2.5-VL 原生多模态）。"""
    kind = (kind or os.environ.get("EXP4_VLM_BACKEND", "local")).lower()
    if kind in ("local", "local_qwen", "qwen"):
        return LocalQwenVL(**{k: v for k, v in kwargs.items() if k == "model_path"})
    if kind in ("cloud", "openai", "cloud_openai"):
        return CloudOpenAI(**{k: v for k, v in kwargs.items()
                              if k in ("base_url", "api_key", "model")})
    raise ValueError(f"未知 VLM 后端: {kind}")


def _messages_to_text(messages: list[dict]) -> str:
    """把 OpenAI 风格 messages 压成单段文本（本地模型无 system 角色时的兜底拼接）。"""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(f"[User]\n{content}")
    return "\n\n".join(parts)
