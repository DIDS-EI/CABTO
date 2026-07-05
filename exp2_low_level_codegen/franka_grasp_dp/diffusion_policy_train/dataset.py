"""Diffusion Policy 数据集
=========================
把采集的 episode_*.npz 转成训练样本。每个样本是一个时间窗口：

    观测:  最近 n_obs_steps 帧 (front_rgb, wrist_rgb, agent_pos)
    标签:  从当前帧起的 horizon 步动作序列 (action)

与 image-based Diffusion Policy 一致：观测取窗口末端对齐，动作预测未来 horizon 步。
不足 horizon 的尾部用最后一帧动作 padding（动作为"保持闭合"，语义合理）。

归一化:
    - 图像: uint8 -> float32 /255，并转成 (C,H,W)。
    - agent_pos / action: 按训练集统计做 [-1,1] 线性归一化（min-max）。
      归一化统计存入 stats，训练后随权重保存，评估时复用。
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_stats(arrays):
    """对一组 (N_i, D) 数组按列求 min/max。"""
    cat = np.concatenate(arrays, axis=0)
    lo = cat.min(axis=0)
    hi = cat.max(axis=0)
    # 防止除零：范围过小的维度强制范围为 1
    rng = hi - lo
    rng[rng < 1e-6] = 1.0
    return lo.astype(np.float32), (lo + rng).astype(np.float32)


def normalize(x, lo, hi):
    """线性映射到 [-1, 1]。"""
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def unnormalize(x, lo, hi):
    return (x + 1.0) / 2.0 * (hi - lo) + lo


class GraspSequenceDataset(Dataset):
    def __init__(self, data_dir, n_obs_steps=2, horizon=16, stats=None):
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon

        files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        if not files:
            raise FileNotFoundError(f"未找到 episode_*.npz 于 {data_dir}")

        self.episodes = []
        all_agent, all_action = [], []
        for f in files:
            d = np.load(f)
            ep = {
                "front_rgb": d["front_rgb"],   # (T,H,W,3) uint8
                "wrist_rgb": d["wrist_rgb"],
                "agent_pos": d["agent_pos"].astype(np.float32),  # (T,10)
                "action": d["action"].astype(np.float32),        # (T,7)
            }
            self.episodes.append(ep)
            all_agent.append(ep["agent_pos"])
            all_action.append(ep["action"])

        # 归一化统计
        if stats is None:
            a_lo, a_hi = compute_stats(all_agent)
            act_lo, act_hi = compute_stats(all_action)
            self.stats = {
                "agent_lo": a_lo, "agent_hi": a_hi,
                "action_lo": act_lo, "action_hi": act_hi,
            }
        else:
            self.stats = stats

        # 构建 (episode_idx, start_t) 索引
        self.index = []
        for ei, ep in enumerate(self.episodes):
            T = ep["action"].shape[0]
            for t in range(T):
                self.index.append((ei, t))

    def __len__(self):
        return len(self.index)

    def _get_obs_window(self, ep, t):
        """取末端对齐到 t 的 n_obs_steps 帧（不足则重复首帧）。"""
        idxs = [max(0, t - (self.n_obs_steps - 1) + i) for i in range(self.n_obs_steps)]
        front = ep["front_rgb"][idxs].astype(np.float32) / 255.0   # (To,H,W,3)
        wrist = ep["wrist_rgb"][idxs].astype(np.float32) / 255.0
        front = np.transpose(front, (0, 3, 1, 2))                  # (To,3,H,W)
        wrist = np.transpose(wrist, (0, 3, 1, 2))
        agent = ep["agent_pos"][idxs]                              # (To,10)
        agent = normalize(agent, self.stats["agent_lo"], self.stats["agent_hi"])
        return front, wrist, agent.astype(np.float32)

    def _get_action_horizon(self, ep, t):
        T = ep["action"].shape[0]
        idxs = [min(T - 1, t + i) for i in range(self.horizon)]
        act = ep["action"][idxs]                                   # (Ta,7)
        act = normalize(act, self.stats["action_lo"], self.stats["action_hi"])
        return act.astype(np.float32)

    def __getitem__(self, i):
        ei, t = self.index[i]
        ep = self.episodes[ei]
        front, wrist, agent = self._get_obs_window(ep, t)
        action = self._get_action_horizon(ep, t)
        return {
            "front_rgb": torch.from_numpy(front),
            "wrist_rgb": torch.from_numpy(wrist),
            "agent_pos": torch.from_numpy(agent),
            "action": torch.from_numpy(action),
        }


def stats_to_numpy_dict(stats):
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
