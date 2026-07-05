"""stack 任务 scripted expert（单臂）：把 4 个 box 堆成一摞。
base(blue) 留在堆叠位不动，其余 3 个(green/yellow/red)依次抓起叠到摞顶。
真值判定 success：4 个 box 沿 z 依次堆叠（相邻 block 底面≈下方顶面、xy 对齐）。
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z
from arm_skills import ArmSkills
import scene_stack as st

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp4_tasks/out_stack"
os.makedirs(OUT, exist_ok=True)
BH = st.BLOCK_HALF


def stack_one(env, sk, arm, block, dst_xy, dst_top_z):
    """抓 block 叠到 dst_xy 处、顶面 dst_top_z 上。返回 block 最终 pos。"""
    p0 = env.get_object_pose(block)[0]
    grasp_z = p0[2] + BH - 0.22 * (2 * BH)        # 抓 block 更靠上部，抓偏更稳定
    # 稳健抓取（每块前 rehome 到干净关节配置；与 cover 同一手法）
    holding, _ = sk.pick_secure(block, grasp_z, lift_top=TABLE_TOP_Z + 0.40,
                                n_close=26, rehome_first=True, retries=0)
    print(f"  [pick {block}] holding={holding} grab_dz={sk.grab_dz:.3f} "
          f"grip_w={arm.get_gripper_width():.4f}")

    # 移到摞顶正上方（高于当前摞高），补偿抓偏，再竖直下放（descend_vertical 锁 xy）
    high_z = max(TABLE_TOP_Z + 0.40, dst_top_z + 0.22)
    sk.move_above([dst_xy[0], dst_xy[1]], high_z=high_z)
    sk.refine_above(dst_xy, high_z=high_z, obj_name=block)
    # 用 stack 专用放置：物体闭环 + 宽限幅 xy 修正，抵抗高 z IK 的 +x 系统漂移。
    # gap=0.012：block 底面落到下方顶面之上一点点再松手坐稳，避免撞飞下方块。
    sk.descend_stack(dst_xy, dst_top_z, obj_name=block, gap=0.012,
                     z_step=0.02, max_rounds=30)
    sk.place_release()
    sk.retreat(dz=0.20)
    sk.hold(grip=1.0, n=22)
    fin = env.get_object_pose(block)[0]
    print(f"  [place {block}] pos={np.round(fin,3)} (dst_top={dst_top_z:.3f})")
    return fin


def main():
    obj_names = [n for n, _, _ in st.BLOCKS]
    env = Exp4Env(st.build_scene_xml(), arms=[("", 0.0)], obj_names=obj_names,
                  render=True, img_size=560)
    env.reset()
    arm = env.arms[0]
    arm.grasp_z_tol = 0.07
    sk = ArmSkills(env, arm, recorder=lambda: env.record_frame("overview"),
                   rec_every=2, block_half=BH)
    env.start_record()
    for _ in range(10):
        env.record_frame("overview")
    env.save_png(os.path.join(OUT, "stack_before.png"), "overview")

    base_name = st.BLOCKS[0][0]
    top_name = base_name           # 当前塔顶那块（用其实测位置做目标，消除累积偏差）
    base_p = env.get_object_pose(base_name)[0]
    print(f"[init] base={base_name}@{np.round(base_p,3)}")

    for name, _, _ in st.BLOCKS[1:]:
        # 关键：每次以「当前塔顶块的实测 xy / 顶面 z」为放置目标——
        # base/塔块是 freejoint，放上一块时可能被轻微推动；固定用 base 初始 xy 会
        # 累积偏差。改为实时读取塔顶块当前位置，让每层都对准真实塔心。
        top_p = env.get_object_pose(top_name)[0]
        stack_xy = top_p[:2].copy()
        cur_top = top_p[2] + BH
        print(f"[stack {name}] onto {top_name} top_xy={np.round(stack_xy,3)} top_z={cur_top:.3f}")
        stack_one(env, sk, arm, name, stack_xy, cur_top)
        top_name = name            # 新塔顶

    # 收尾沉降
    sk.hold(grip=1.0, n=40)

    # ---- 判定：按 z 排序，相邻两块底/顶贴合且 xy 对齐 ----
    poses = {n: env.get_object_pose(n)[0] for n, _, _ in st.BLOCKS}
    order = sorted(poses, key=lambda n: poses[n][2])     # 由低到高
    print("[stacked order]", " < ".join(f"{n}({poses[n][2]:.3f})" for n in order))
    ok_pairs = 0
    details = []
    for i in range(len(order) - 1):
        lo, hi = poses[order[i]], poses[order[i + 1]]
        xy_off = float(np.linalg.norm(hi[:2] - lo[:2]))
        z_gap = (hi[2] - BH) - (lo[2] + BH)              # 上块底面 - 下块顶面
        good = xy_off < 0.035 and -0.02 < z_gap < 0.03
        ok_pairs += int(good)
        details.append({"upper": order[i + 1], "lower": order[i],
                        "xy_off": round(xy_off, 4), "z_gap": round(z_gap, 4),
                        "good": bool(good)})
        print(f"  {order[i+1]} on {order[i]}: xy_off={xy_off:.4f} z_gap={z_gap:.4f} good={good}")
    stacked = ok_pairs == len(order) - 1
    print(f"[result] stacked={stacked} ({ok_pairs}/{len(order)-1} 对贴合)")

    env.save_png(os.path.join(OUT, "stack_after.png"), "overview")
    env.save_video(os.path.join(OUT, "rollout.mp4"), fps=30)
    res = {"task": "stack", "success": bool(stacked), "n_blocks": len(st.BLOCKS),
           "order_low_to_high": order, "pairs": details}
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[saved]", OUT)
    return stacked


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
