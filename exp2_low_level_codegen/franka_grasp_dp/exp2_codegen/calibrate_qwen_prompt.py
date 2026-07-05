"""标定 Qwen2.5-VL 打点 prompt
=================================

目标：在真实渲染图上，用「oracle 几何真值投影点」作为 ground truth，
比较若干 prompt 变体 + 解析策略，给出像素误差 px_err，选出最佳组合。

跑法（cabto 环境）：
  cd mujoco_workspace/franka_grasp_dp
  PYTHONPATH=. <cabto python> exp2_codegen/calibrate_qwen_prompt.py
"""
from __future__ import annotations

import os, sys, json, re, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp_common.franka_dp_env import FrankaDPEnv, TABLE_TOP_Z
from exp2_codegen import camera as C

IMG = 512                     # 方形渲染（env.renderer 用 img_size×img_size）
W = H = IMG
CAMS = ["overview", "front"]
SEEDS = [0, 1, 2, 3]          # 4 个布局
MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

# ---- prompt 变体 ---- #
PROMPTS = {
    "terse_json": (
        'Point to the object on the table. '
        'Respond ONLY with JSON: {"point": [x, y]} in pixel coordinates.'
    ),
    "qwen_native": (
        'Locate the object on the table. Output its center as a point.'
    ),
    "grasp_terse": (
        'Point to the center of the object on the table to grasp it. '
        'Answer ONLY as {"point": [x, y]} in pixels.'
    ),
}


def parse_point(text, w, h):
    if not text:
        return None
    # Qwen2.5-VL pointing 常见: <points x1="100" y1="200">obj</points>
    m = re.search(r'x1?="?([\d.]+)"?\s+y1?="?([\d.]+)"?', text)
    if m:
        return _denorm(float(m.group(1)), float(m.group(2)), w, h)
    # JSON {"point":[x,y]} 或 [[x,y]]
    mj = re.search(r'\[\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)', text)
    if mj:
        return _denorm(float(mj.group(1)), float(mj.group(2)), w, h)
    nums = [float(n) for n in re.findall(r'\d+\.?\d*', text)]
    if len(nums) >= 2:
        return _denorm(nums[0], nums[1], w, h)
    return None


def _denorm(x, y, w, h):
    # Qwen2.5-VL 输出绝对像素（基于输入分辨率）。若疑似 0-1000 归一化则还原。
    if (x > w or y > h) and x <= 1000 and y <= 1000:
        x = x / 1000.0 * w
        y = y / 1000.0 * h
    return float(x), float(y)


def main():
    from mlx_vlm import load, generate, apply_chat_template
    from mlx_vlm.utils import load_config
    import imageio.v2 as imageio

    print("[load] Qwen2.5-VL ...", flush=True)
    model, processor = load(MODEL)
    config = load_config(MODEL)

    def ask(img_rgb, prompt_txt):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        imageio.imwrite(tmp, img_rgb)
        try:
            pr = apply_chat_template(processor, config, prompt_txt, num_images=1)
            out = generate(model, processor, pr, image=[tmp],
                           max_tokens=64, temperature=0.0, verbose=False)
            return out.text if hasattr(out, "text") else str(out)
        finally:
            try: os.unlink(tmp)
            except OSError: pass

    env = FrankaDPEnv(img_size=IMG, render_enabled=True, seed=0)
    env.build()

    agg = {k: [] for k in PROMPTS}
    samples = []
    for seed in SEEDS:
        env.reset(seed=seed)
        opos, _ = env.get_object_pose()
        top = opos.copy(); top[2] = TABLE_TOP_Z + env._cur_top
        for cam in CAMS:
            gt = C.project_point(env.m, env.d, W, H, cam, top)
            img = env._grab_rgb(cam)
            for pname, ptext in PROMPTS.items():
                txt = ask(img, ptext)
                uv = parse_point(txt, W, H)
                if uv is None:
                    err = float("nan")
                else:
                    err = float(np.hypot(uv[0]-gt[0], uv[1]-gt[1]))
                agg[pname].append(err)
                samples.append({"seed": seed, "cam": cam, "prompt": pname,
                                "gt": [round(gt[0],1), round(gt[1],1)],
                                "pred": None if uv is None else [round(uv[0],1), round(uv[1],1)],
                                "px_err": None if uv != uv else round(err,1),
                                "raw": txt.strip()[:120]})
                print(f"seed{seed} {cam:8s} {pname:12s} gt=({gt[0]:.0f},{gt[1]:.0f}) "
                      f"pred={uv} err={err:.1f}  raw={txt.strip()[:60]!r}", flush=True)

    print("\n==== 汇总（中位/均值像素误差，越小越好）====")
    summary = {}
    for pname, errs in agg.items():
        valid = [e for e in errs if e == e]
        med = float(np.median(valid)) if valid else float("nan")
        mean = float(np.mean(valid)) if valid else float("nan")
        nfail = len(errs) - len(valid)
        summary[pname] = {"median_px": round(med,1), "mean_px": round(mean,1),
                          "n_fail_parse": nfail, "n": len(errs)}
        print(f"  {pname:14s} median={med:6.1f}  mean={mean:6.1f}  fail={nfail}/{len(errs)}")

    out = {"summary": summary, "samples": samples, "img_wh": [W, H]}
    op = os.path.join(os.path.dirname(__file__), "results", "qwen_prompt_calib.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {op}")


if __name__ == "__main__":
    main()
