"""实验2（代码生成 + VLM 打点）可视化渲染
=============================================

产出三类可视化，落到 exp2_codegen/renders/：
1. pointing_overlay_<obj>.png  —— 在 overview / front 两视角图上叠加 VLM(或oracle)
   打的像素点（红色十字）+ 反投影/三角化得到的 3D 目标重投影点（绿色圆），
   直观展示「描述→打点→2D 转 3D」的感知链路。
2. rollout_<obj>.mp4           —— 用代码生成的 pick() 执行 rollout，逐 step 抓取
   overview 视角帧，串成抓取过程视频。
3. multiview_<obj>.png         —— 同一时刻 overview/front/wrist 三视角拼图。

跑法（cabto 环境）：
  cd mujoco_workspace/franka_grasp_dp
  PYTHONPATH=. <cabto python> exp2_codegen/render_viz.py --backend oracle
  PYTHONPATH=. <cabto python> exp2_codegen/render_viz.py --backend qwen
"""
from __future__ import annotations

import os, sys, json, argparse, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import imageio.v2 as imageio

from dp_common.franka_dp_env import FrankaDPEnv, TABLE_TOP_Z
from exp2_codegen import camera as C
from exp2_codegen.primitives import PrimitiveRunner
from exp2_codegen.vlm_pointer import make_pointer
from exp2_codegen import codegen_action as CG

IMG = 512
W = H = IMG
OUT = os.path.join(os.path.dirname(__file__), "renders")
os.makedirs(OUT, exist_ok=True)

# 演示物体（覆盖 cube / cylinder / sphere 等形态）
DEMO_OBJECTS = ["dice_cube", "wood_block", "soda_can", "tea_tin"]


# --------------------------- 画图辅助（纯 numpy，无需 PIL/cv2） --------------------------- #
def draw_cross(img, u, v, color, size=10, thick=2):
    h, w = img.shape[:2]
    u, v = int(round(u)), int(round(v))
    for d in range(-size, size + 1):
        for t in range(-thick, thick + 1):
            for (x, y) in [(u + d, v + t), (u + t, v + d)]:
                if 0 <= x < w and 0 <= y < h:
                    img[y, x] = color


def draw_circle(img, u, v, color, r=8, thick=2):
    h, w = img.shape[:2]
    u, v = int(round(u)), int(round(v))
    for ang in np.linspace(0, 2 * np.pi, 120):
        for dr in range(thick):
            x = int(round(u + (r + dr) * np.cos(ang)))
            y = int(round(v + (r + dr) * np.sin(ang)))
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = color


def hstack_pad(imgs, pad=6, bg=255):
    h = max(im.shape[0] for im in imgs)
    cols = []
    for im in imgs:
        if im.shape[0] < h:
            im = np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)),
                        constant_values=bg)
        cols.append(im)
        cols.append(np.full((h, pad, 3), bg, np.uint8))
    return np.concatenate(cols[:-1], axis=1)


# --------------------------- 主流程 --------------------------- #
def render_for_object(obj_name, backend, seed=0, model_path=None, output_tag=None):
    env = FrankaDPEnv(img_size=IMG, fixed_object=obj_name,
                      render_enabled=True, seed=seed)
    env.build()
    env.reset(seed=seed)
    tag = output_tag or backend

    # pointer
    if backend == "oracle":
        pointer = make_pointer("oracle",
                               project_fn=CG._make_oracle_project_fn(env, (W, H)))
        instruction = "the top center of the object on the table to grasp"
    elif backend == "qwen":
        pointer = make_pointer("qwen", **({"model_path": model_path} if model_path else {}))
        instruction = "the object on the table"
    else:
        raise ValueError(backend)

    # 深度渲染器（反投影/交叉校验用）
    dep = mujoco.Renderer(env.m, IMG, IMG)
    dep.enable_depth_rendering()

    # ---------- (A) pointing overlay：打点 + 反投影重投影 ---------- #
    overlay_info = {"obj": obj_name, "backend": backend, "model_path": model_path}
    pts = {}
    for cam in ["overview", "front"]:
        if hasattr(pointer, "set_camera"):
            pointer.set_camera(cam)
        img = env._grab_rgb(cam).copy()
        uv = pointer.point(img, instruction)
        pts[cam] = uv
        if uv is not None:
            draw_cross(img, uv[0], uv[1], np.array([255, 40, 40], np.uint8))
        imageio.imwrite(os.path.join(OUT, f"point_{tag}_{obj_name}_{cam}.png"), img)

    # 三角化 3D 点 + 重投影回两视角（绿圈）验证一致性
    p3d = None
    if pts.get("overview") is not None and pts.get("front") is not None:
        p3d = C.triangulate(env.m, env.d, W, H,
                            "overview", pts["overview"][0], pts["overview"][1],
                            "front", pts["front"][0], pts["front"][1])
        overlay_info["triangulated_xyz"] = [round(float(x), 4) for x in p3d]

    overlay_imgs = []
    for cam in ["overview", "front"]:
        img = env._grab_rgb(cam).copy()
        if pts.get(cam) is not None:
            draw_cross(img, pts[cam][0], pts[cam][1], np.array([255, 40, 40], np.uint8))
        if p3d is not None:
            rp = C.project_point(env.m, env.d, W, H, cam, p3d)
            if rp is not None:
                draw_circle(img, rp[0], rp[1], np.array([40, 220, 60], np.uint8))
        overlay_imgs.append(img)
    overlay = hstack_pad(overlay_imgs)
    imageio.imwrite(os.path.join(OUT, f"pointing_overlay_{tag}_{obj_name}.png"), overlay)
    overlay_info["target_px"] = {k: (None if v is None else [round(v[0], 1), round(v[1], 1)])
                                 for k, v in pts.items()}

    # ---------- (B) rollout 视频：执行代码生成 pick，逐 step 录 overview ---------- #
    # 复用同一 env（已 reset），重新跑一遍 pick 并录帧
    env.reset(seed=seed)
    if backend == "oracle":
        pointer = make_pointer("oracle",
                               project_fn=CG._make_oracle_project_fn(env, (W, H)))

    frames = []
    runner = PrimitiveRunner(env, log=[])
    orig_apply = runner._apply

    def apply_and_record(dpos, grip, drot=(0, 0, 0)):
        out = orig_apply(dpos, grip, drot)
        if len(runner.traj) % 2 == 0:  # 每 2 步抓一帧，控制时长
            frames.append(env._grab_rgb("overview").copy())
        return out
    runner._apply = apply_and_record

    # —— 内联「被生成的代码」（与 codegen_action.pick 一致，但用本地 runner 录帧）——
    if hasattr(pointer, "set_camera"):
        pointer.set_camera("overview")
    uv_o = pointer.point(env._grab_rgb("overview"), instruction)
    if hasattr(pointer, "set_camera"):
        pointer.set_camera("front")
    uv_f = pointer.point(env._grab_rgb("front"), instruction)
    if uv_o is not None and uv_f is not None:
        p = C.triangulate(env.m, env.d, W, H, "overview", uv_o[0], uv_o[1],
                          "front", uv_f[0], uv_f[1])
    else:
        p = C.deproject_with_depth(env.m, env.d, dep, W, H, "overview",
                                   uv_o[0], uv_o[1]) if uv_o is not None else None
    success = False
    if p is not None:
        obj_center_z = TABLE_TOP_Z + env._cur_bottom
        obj_top_z = obj_center_z + env._cur_top
        obj_bottom_z = obj_center_z - env._cur_bottom
        grasp_z = obj_top_z - 0.6 * (env._cur_top + env._cur_bottom)
        p[2] = max(grasp_z, obj_bottom_z + 0.006)
        runner.approach(p, height=0.12, grip=1.0)
        runner.open_gripper(hold=2)
        runner.descend_to(p, grip=1.0, tol=0.010)
        runner.align_xy(p, grip=1.0, horiz_tol=0.006, max_steps=25)
        runner.close_gripper_following(p, hold=22, grip=0.0)
        runner.lift(height=0.30, grip=0.0)
        success = bool(env.check_success())
    overlay_info["rollout_success"] = success
    overlay_info["n_frames"] = len(frames)

    if frames:
        mp4 = os.path.join(OUT, f"rollout_{tag}_{obj_name}.mp4")
        imageio.mimsave(mp4, frames, fps=20, quality=8, macro_block_size=1)
        # 末帧做封面
        imageio.imwrite(os.path.join(OUT, f"rollout_{tag}_{obj_name}_last.png"),
                        frames[-1])

    # ---------- (C) 三视角拼图（抓取后状态）---------- #
    mv = hstack_pad([env._grab_rgb(c).copy() for c in ["overview", "front", "wrist"]])
    imageio.imwrite(os.path.join(OUT, f"multiview_{tag}_{obj_name}.png"), mv)

    dep.close()
    del env, dep, runner
    gc.collect()
    return overlay_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="oracle", choices=["oracle", "qwen"])
    ap.add_argument("--objects", nargs="*", default=DEMO_OBJECTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--result-tag", default=None)
    args = ap.parse_args()

    results = []
    for obj in args.objects:
        print(f"[render] {args.backend} :: {obj}", flush=True)
        try:
            info = render_for_object(obj, args.backend, seed=args.seed,
                                     model_path=args.model_path, output_tag=args.result_tag)
            print(f"   success={info.get('rollout_success')} "
                  f"px={info.get('target_px')} frames={info.get('n_frames')}", flush=True)
            results.append(info)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"obj": obj, "error": str(e)})

    summary_tag = args.result_tag or args.backend
    with open(os.path.join(OUT, f"viz_summary_{summary_tag}.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    n_ok = sum(1 for r in results if r.get("rollout_success"))
    print(f"\n[done] {args.backend}: rollout 成功 {n_ok}/{len(results)}；产物在 {OUT}")


if __name__ == "__main__":
    main()
