"""Diffusion Policy 端到端训练
==============================
读取 dp_data_collection/data 下的 episode_*.npz，训练一个 image-based
Diffusion Policy（双相机 + 本体状态 -> 末端 6D delta 动作序列）。

用法:
    python train.py --data_dir ../dp_data_collection/data --epochs 100
    python train.py --epochs 200 --batch_size 64 --horizon 16 --n_obs_steps 2

输出:
    checkpoints/dp_latest.pt   每轮覆盖保存（含模型权重 + 归一化统计 + 超参）
    checkpoints/dp_best.pt     训练损失最优
    checkpoints/train_log.json 损失曲线

设备: 自动选择 cuda > mps > cpu。CPU 上请用较小 batch / epochs 先跑通。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))   # 让 diffusion_policy_train 可作为包导入

from diffusion_policy_train.model import DiffusionPolicy          # noqa: E402
from diffusion_policy_train.dataset import (                      # noqa: E402
    GraspSequenceDataset, stats_to_numpy_dict)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="训练 Diffusion Policy")
    p.add_argument("--data_dir", type=str,
                   default=os.path.join(_THIS, "..", "dp_data_collection", "data"))
    p.add_argument("--out_dir", type=str, default=os.path.join(_THIS, "checkpoints"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=16, help="预测动作序列长度")
    p.add_argument("--n_obs_steps", type=int, default=2, help="观测历史帧数")
    p.add_argument("--num_diffusion_steps", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ema_decay", type=float, default=0.995, help="EMA 权重衰减")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--resume", type=str, default=None,
                   help="从指定 ckpt 续训（恢复 model/ema/optimizer/scheduler/epoch）")
    return p.parse_args()


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b)


def main():
    args = parse_args()
    device = args.device or pick_device()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f">> 设备: {device}")

    # ---- 数据 ---- #
    dataset = GraspSequenceDataset(
        args.data_dir, n_obs_steps=args.n_obs_steps, horizon=args.horizon)
    print(f">> 样本数: {len(dataset)} (来自 {len(dataset.episodes)} 集)")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True)

    # 推断维度
    sample = dataset[0]
    agent_dim = sample["agent_pos"].shape[-1]
    action_dim = sample["action"].shape[-1]

    # ---- 模型 ---- #
    model = DiffusionPolicy(
        action_dim=action_dim, agent_dim=agent_dim,
        n_obs_steps=args.n_obs_steps, horizon=args.horizon,
        num_diffusion_steps=args.num_diffusion_steps, device=device,
    ).to(device)
    # 重新把 scheduler 张量放到 device
    model.scheduler = type(model.scheduler)(
        num_steps=args.num_diffusion_steps, device=device)

    ema_model = DiffusionPolicy(
        action_dim=action_dim, agent_dim=agent_dim,
        n_obs_steps=args.n_obs_steps, horizon=args.horizon,
        num_diffusion_steps=args.num_diffusion_steps, device=device,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f">> 模型参数量: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = {"epoch_loss": [], "args": vars(args)}
    best = float("inf")
    start_epoch = 0

    # ---- 续训：恢复 model/ema/opt/sched/epoch/best ---- #
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck["ema_model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0)
        best = ck.get("best", float("inf"))
        if "epoch_loss" in ck:
            log["epoch_loss"] = list(ck["epoch_loss"])
        print(f">> 续训: 从 {args.resume} 恢复，已完成 {start_epoch} 轮，best={best:.5f}")

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        nb = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.compute_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema_model, model, args.ema_decay)
            running += loss.item()
            nb += 1
        sched.step()
        ep_loss = running / max(nb, 1)
        log["epoch_loss"].append(ep_loss)

        if epoch % args.log_every == 0:
            print(f"[{epoch+1:3d}/{args.epochs}] loss={ep_loss:.5f} "
                  f"lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s")

        ckpt = {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch + 1,
            "best": min(best, ep_loss),
            "epoch_loss": log["epoch_loss"],
            "stats": stats_to_numpy_dict(dataset.stats),
            "config": {
                "action_dim": action_dim, "agent_dim": agent_dim,
                "n_obs_steps": args.n_obs_steps, "horizon": args.horizon,
                "num_diffusion_steps": args.num_diffusion_steps,
            },
        }
        # 原子保存：先写 .tmp 再 os.replace（rename 原子，SIGTERM 打断
        # 只会留下未完成的 .tmp，不会损坏正式 ckpt，保证续训安全）
        latest = os.path.join(args.out_dir, "dp_latest.pt")
        torch.save(ckpt, latest + ".tmp")
        os.replace(latest + ".tmp", latest)
        if ep_loss < best:
            best = ep_loss
            bestp = os.path.join(args.out_dir, "dp_best.pt")
            torch.save(ckpt, bestp + ".tmp")
            os.replace(bestp + ".tmp", bestp)

    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n训练完成: best_loss={best:.5f}, 用时 {time.time()-t0:.0f}s")
    print(f"权重: {os.path.join(args.out_dir, 'dp_best.pt')}")


if __name__ == "__main__":
    main()
