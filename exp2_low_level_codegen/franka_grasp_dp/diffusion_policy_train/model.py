"""Diffusion Policy 模型（image-based, 自包含）
==================================================
参考 Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
(image-based 变体) 的核心结构，从零实现（不依赖 diffusers）：

    [双相机 RGB + 本体状态]  --视觉编码-->  观测条件向量 (global_cond)
                                                    |
    噪声动作序列 a_noisy (B, Ta, 7) + 扩散步 k  --条件 1D UNet-->  预测噪声 ε

训练：DDPM 噪声预测损失（MSE）。
推理：从纯噪声出发，DDPM 反向去噪若干步，得到动作序列。

组件
----
- VisionEncoder      : 小型 CNN（类 ResNet-block 简化版），编码单张相机图。
- ObsEncoder         : 双相机 + agent_pos -> global_cond。
- ConditionalUNet1D  : 沿「时间维(动作序列长度)」做 1D 卷积的条件 UNet，
                       用 FiLM 把 (扩散步嵌入 + 观测条件) 注入每个残差块。
- DDPMScheduler      : 余弦/线性 beta 调度 + 加噪 / 去噪。
- DiffusionPolicy    : 顶层封装，提供 compute_loss / predict_action。

为兼顾 CPU 训练，网络规模偏小（通道数可调）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. 视觉编码器：小型 CNN
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class VisionEncoder(nn.Module):
    """输入 (B,3,H,W) -> 输出 (B, feat_dim)。H,W 一般为 96。"""

    def __init__(self, feat_dim=128, in_ch=3, width=32):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            ConvBlock(width, width, stride=2),       # 96->48
            ConvBlock(width, width * 2, stride=2),   # 48->24
            ConvBlock(width * 2, width * 4, stride=2),  # 24->12
            ConvBlock(width * 4, width * 4, stride=2),  # 12->6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 4, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------- #
# 2. 观测编码：双相机 + 本体状态 -> 全局条件向量
# --------------------------------------------------------------------------- #
class ObsEncoder(nn.Module):
    def __init__(self, agent_dim=10, vis_feat=128, n_obs_steps=2, cond_dim=256):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.front_enc = VisionEncoder(vis_feat)
        self.wrist_enc = VisionEncoder(vis_feat)
        in_dim = (vis_feat * 2 + agent_dim) * n_obs_steps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_dim = cond_dim

    def forward(self, front, wrist, agent_pos):
        """front/wrist: (B, To, 3, H, W); agent_pos: (B, To, agent_dim)."""
        B, To = front.shape[:2]
        f = self.front_enc(front.reshape(B * To, *front.shape[2:]))
        w = self.wrist_enc(wrist.reshape(B * To, *wrist.shape[2:]))
        f = f.reshape(B, To, -1)
        w = w.reshape(B, To, -1)
        feat = torch.cat([f, w, agent_pos], dim=-1)   # (B, To, *)
        feat = feat.reshape(B, -1)                    # 拼接观测步
        return self.mlp(feat)                         # (B, cond_dim)


# --------------------------------------------------------------------------- #
# 3. 扩散步的正弦位置嵌入
# --------------------------------------------------------------------------- #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --------------------------------------------------------------------------- #
# 4. 条件 1D UNet（沿动作序列时间维卷积，FiLM 注入条件）
# --------------------------------------------------------------------------- #
class CondResBlock1D(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad)
        self.norm1 = nn.GroupNorm(min(8, c_out), c_out)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.SiLU()
        # FiLM：从条件生成 (scale, bias)
        self.cond_proj = nn.Linear(cond_dim, c_out * 2)
        self.res_conv = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, cond):
        # x: (B, C, T) ; cond: (B, cond_dim)
        h = self.act(self.norm1(self.conv1(x)))
        scale, bias = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)   # FiLM
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.res_conv(x)


class ConditionalUNet1D(nn.Module):
    """动作序列去噪网络。输入 (B, Ta, action_dim)，输出同形噪声预测。"""

    def __init__(self, action_dim=7, cond_dim=256, diff_emb_dim=128,
                 channels=(64, 128, 256)):
        super().__init__()
        self.diff_emb = nn.Sequential(
            SinusoidalPosEmb(diff_emb_dim),
            nn.Linear(diff_emb_dim, diff_emb_dim),
            nn.SiLU(),
            nn.Linear(diff_emb_dim, diff_emb_dim),
        )
        # 总条件 = 观测条件 + 扩散步嵌入
        self.cond_dim = cond_dim + diff_emb_dim

        c0, c1, c2 = channels
        self.in_proj = nn.Conv1d(action_dim, c0, 1)

        # 下采样路径（沿 T 维 stride=2）
        self.down1 = CondResBlock1D(c0, c0, self.cond_dim)
        self.down2 = CondResBlock1D(c0, c1, self.cond_dim)
        self.pool = nn.AvgPool1d(2)
        self.down3 = CondResBlock1D(c1, c2, self.cond_dim)

        # 中间
        self.mid = CondResBlock1D(c2, c2, self.cond_dim)

        # 上采样路径
        self.up1 = CondResBlock1D(c2 + c1, c1, self.cond_dim)
        self.up2 = CondResBlock1D(c1 + c0, c0, self.cond_dim)
        self.out_proj = nn.Sequential(
            nn.GroupNorm(min(8, c0), c0),
            nn.SiLU(),
            nn.Conv1d(c0, action_dim, 1),
        )

    def forward(self, x, diffusion_t, global_cond):
        """x: (B, Ta, action_dim) ; diffusion_t: (B,) ; global_cond: (B, cond_dim)"""
        t_emb = self.diff_emb(diffusion_t)
        cond = torch.cat([global_cond, t_emb], dim=-1)

        h = x.transpose(1, 2)            # (B, action_dim, T)
        h = self.in_proj(h)

        d1 = self.down1(h, cond)         # (B, c0, T)
        d2 = self.down2(d1, cond)        # (B, c1, T)
        d2p = self.pool(d2)              # (B, c1, T/2)
        d3 = self.down3(d2p, cond)       # (B, c2, T/2)

        m = self.mid(d3, cond)           # (B, c2, T/2)

        u1 = F.interpolate(m, size=d2.shape[-1], mode="nearest")
        u1 = self.up1(torch.cat([u1, d2], dim=1), cond)   # (B, c1, T)
        u2 = self.up2(torch.cat([u1, d1], dim=1), cond)   # (B, c0, T)
        out = self.out_proj(u2)          # (B, action_dim, T)
        return out.transpose(1, 2)       # (B, T, action_dim)


# --------------------------------------------------------------------------- #
# 5. DDPM 调度器
# --------------------------------------------------------------------------- #
class DDPMScheduler:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.device = device

    def add_noise(self, x0, noise, t):
        """前向加噪: x_t = sqrt(acp)·x0 + sqrt(1-acp)·noise"""
        acp = self.alphas_cumprod[t].view(-1, 1, 1)
        return torch.sqrt(acp) * x0 + torch.sqrt(1 - acp) * noise

    @torch.no_grad()
    def step(self, model_eps, t, x_t):
        """单步反向去噪 (DDPM ancestral sampling)。"""
        beta = self.betas[t]
        alpha = self.alphas[t]
        acp = self.alphas_cumprod[t]
        coef = beta / torch.sqrt(1 - acp)
        mean = (x_t - coef * model_eps) / torch.sqrt(alpha)
        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta) * noise
        return mean


# --------------------------------------------------------------------------- #
# 6. 顶层 Diffusion Policy
# --------------------------------------------------------------------------- #
class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=7, agent_dim=10, n_obs_steps=2, horizon=16,
                 num_diffusion_steps=100, cond_dim=256, device="cpu"):
        super().__init__()
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.device = device

        self.obs_encoder = ObsEncoder(
            agent_dim=agent_dim, n_obs_steps=n_obs_steps, cond_dim=cond_dim)
        self.unet = ConditionalUNet1D(action_dim=action_dim, cond_dim=cond_dim)
        self.scheduler = DDPMScheduler(num_steps=num_diffusion_steps, device=device)

    # ---- 训练损失 ---- #
    def compute_loss(self, batch):
        front = batch["front_rgb"]      # (B,To,3,H,W) float[0,1]
        wrist = batch["wrist_rgb"]
        agent = batch["agent_pos"]      # (B,To,agent_dim)
        actions = batch["action"]       # (B,Ta,action_dim) 已归一化

        B = actions.shape[0]
        cond = self.obs_encoder(front, wrist, agent)

        t = torch.randint(0, self.scheduler.num_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        noisy = self.scheduler.add_noise(actions, noise, t)
        pred = self.unet(noisy, t, cond)
        return F.mse_loss(pred, noise)

    # ---- 推理：生成动作序列 ---- #
    @torch.no_grad()
    def predict_action(self, front, wrist, agent):
        """输入单帧/多帧观测 (1,To,...)，返回归一化动作序列 (1, horizon, action_dim)。"""
        cond = self.obs_encoder(front, wrist, agent)
        x = torch.randn(cond.shape[0], self.horizon, self.action_dim,
                        device=cond.device)
        for k in reversed(range(self.scheduler.num_steps)):
            t = torch.full((cond.shape[0],), k, device=cond.device, dtype=torch.long)
            eps = self.unet(x, t, cond)
            x = self.scheduler.step(eps, k, x)
        return x
