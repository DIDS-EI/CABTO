"""cover 任务 scripted expert（单臂）：把 3 个彩色盖分别盖到同色的 3 个底座上。
对每一对 (lid_k, base_k)：pick lid_k -> 移到 base_k 上方 -> descend_place_tracked 盖上。
真值判定 success：3 对都满足 lid 中心 xy 对准 base 中心(<0.04) 且 lid 落在 base 顶面附近。
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z
from arm_skills import ArmSkills
import scene_cover as sc

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp4_tasks/out_cover"
os.makedirs(OUT, exist_ok=True)


def cover_one(env, sk, arm, lid_name, base_xy, base_top):
    """抓 lid 盖到 base 顶面。返回 (covered, xy_off, z_gap, lid_final)。"""
    lid0 = env.get_object_pose(lid_name)[0]
    # 抓把手上部：把手顶 = lid_z + LID_HH + 2*HANDLE_HH
    handle_top = lid0[2] + sc.LID_HH + 2 * sc.HANDLE_HH
    grasp_z = handle_top - 0.30 * (2 * sc.HANDLE_HH)

    # 稳健抓取（每对前 rehome 到干净关节配置）
    holding, _ = sk.pick_secure(lid_name, grasp_z, lift_top=TABLE_TOP_Z + 0.32,
                                n_close=28, rehome_first=True, retries=0)
    print(f"  [pick {lid_name}] holding={holding} grab_dz={sk.grab_dz:.3f} "
          f"grip_w={arm.get_gripper_width():.4f}")

    p = np.array([base_xy[0], base_xy[1], base_top])
    sk.move_above(p, high_z=TABLE_TOP_Z + 0.32)
    sk.refine_above(base_xy, high_z=TABLE_TOP_Z + 0.32, obj_name=lid_name)
    # 盖盖高精度对位：边降边把 lid 实测 xy 拉向 base 中心
    sk.descend_place_tracked(base_xy, base_top, obj_name=lid_name, gap=0.003, max_steps=300)
    sk.place_release()
    sk.retreat(dz=0.16)
    sk.hold(grip=1.0, n=20)

    lid_f = env.get_object_pose(lid_name)[0]
    base_c = np.array(base_xy)
    xy_off = float(np.linalg.norm(lid_f[:2] - base_c))
    lid_bottom = lid_f[2] - sc.LID_HH
    z_gap = lid_bottom - base_top
    covered = (xy_off < 0.04) and (-0.02 < z_gap < 0.05)
    print(f"  [place {lid_name}] lid={np.round(lid_f,3)} xy_off={xy_off:.4f} "
          f"z_gap={z_gap:.3f} covered={covered}")
    return covered, xy_off, z_gap, lid_f


def main():
    obj_names = [f"lid_{k}" for k, *_ in sc.PAIRS]
    env = Exp4Env(sc.build_scene_xml(), arms=[("", 0.0)], obj_names=obj_names,
                  render=True, img_size=560)
    env.reset()
    arm = env.arms[0]
    arm.grasp_z_tol = 0.10          # 抓高把手：放宽吸附垂直容差
    sk = ArmSkills(env, arm, recorder=lambda: env.record_frame("overview"),
                   rec_every=2, block_half=sc.LID_HH)
    env.start_record()
    for _ in range(10):
        env.record_frame("overview")

    env.save_png(os.path.join(OUT, "cover_before.png"), "overview")

    results = {}
    for key, base_xy, lid_xy, mat in sc.PAIRS:
        lid_name = f"lid_{key}"
        print(f"[pair {key}] lid={lid_name} -> base_{key}@{base_xy}")
        covered, xy_off, z_gap, lid_f = cover_one(env, sk, arm, lid_name,
                                                  np.array(base_xy), sc.BASE_TOP)
        results[key] = {"covered": bool(covered), "xy_off": round(xy_off, 4),
                        "z_gap": round(z_gap, 4),
                        "lid_final": [round(float(v), 4) for v in lid_f]}

    all_covered = all(r["covered"] for r in results.values())
    print(f"[result] all_covered={all_covered}  detail=" +
          " ".join(f"{k}:{results[k]['covered']}" for k in results))

    env.save_png(os.path.join(OUT, "cover_after.png"), "overview")
    env.save_video(os.path.join(OUT, "rollout.mp4"), fps=30)
    res = {"task": "cover", "success": bool(all_covered),
           "n_pairs": len(sc.PAIRS), "pairs": results}
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[saved]", OUT)
    return all_covered


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
