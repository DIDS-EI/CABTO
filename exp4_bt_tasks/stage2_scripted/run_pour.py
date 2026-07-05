"""pour 双臂 scripted expert（双向互倒 · 固定杯版）——真实规格：
① 左臂抓 canL → 移到右区固定 cupR 上方预偏 → 翻罐倒球 → ballL 落入 cupR；放回 canL。
② 右臂抓 canR → 移到左区固定 cupL 上方预偏 → 翻罐倒球 → ballR 落入 cupL；放回 canR。
真值判定 success：ballL 在 cupR 内、ballR 在 cupL 内（xy 在杯口内、低于杯沿、未掉地）。

要点：
- 杯固定（scene 无 freejoint），消除撞飞问题，用 get_body_pos 查询。
- 倾倒沿用老单向版已验证手法：绕世界水平轴翻 ~125°(total_angle=2.2)，罐口画弧落杯心上方；
  倒前把罐预偏到杯心对侧，抵消画弧位移。左臂用 +x 轴、右臂镜像用 -x 轴（arc_sign）。
- 倒罐臂用 move_to（不需精确 descend）；右臂积分用小 ki 防 windup。
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z
from arm_skills import ArmSkills
import scene_pour as sp

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp4_tasks/out_pour"
os.makedirs(OUT, exist_ok=True)

CUP_TOP = sp.CUP_TOP


def pick_can(env, sk, arm, can_name):
    """稳健抓罐（pick_direct 按臂别分支）。罐在各臂稳定抓取区。"""
    c0 = env.get_object_pose(can_name)[0]
    grasp_z = c0[2] + sp.CAN_HH * 0.40
    holding, _ = sk.pick_can_simple(can_name, grasp_z, lift_top=TABLE_TOP_Z + 0.30,
                                    n_close=34)
    print(f"  [pick {can_name}] holding={holding} grab_dz={sk.grab_dz:.3f} "
          f"grip_w={arm.get_gripper_width():.4f}", flush=True)
    return holding


def pour_over_cup(env, sk, can_name, cup_xy, arc_sign, tip_deg=125.0, dx_off=0.075,
                  pour_z=None):
    """把所持 can 移到固定 cup_xy 旁并用「关节空间翻罐」倾倒（tip_pour_joint）。
    关键认知（diag_weld_rot/diag_side_pour 实测）：
    - weld 刚性传递旋转；抓取时 hand_mz=-1(朝下)、can_mz=+1(罐口朝上)，两者相反。要倒出
      须让 hand 绕世界水平轴转 ~125°，can 随之 can_mz=cos(125°)≈-0.57（过水平，球滚出）。
    - 倾倒轴 ny(绕 -y)：罐口由 +z 转向 +x。故罐心放在杯心 -x 一侧(dx_off)，翻转后罐口
      探到杯心上方。axis 用 ny；arc_sign 仅用于左右臂镜像 y 方向微调。
    - 必须用 tip_pour_joint（full-restart IK 解一次目标倾角 + 关节线性插值驱动），
      而非 tip_pour（逐步 local IK，实测要么不转、要么跳支把球甩飞致爆炸）。"""
    cup_xy = np.asarray(cup_xy, float)
    POUR_Z = (TABLE_TOP_Z + 0.18) if pour_z is None else pour_z   # 罐心 z
    move_ki = 0.15 if sk._is_right() else 0.5
    # 罐心放杯心 -x 一侧、清开杯口，翻转后罐口摆向 +x（杯心）上方
    tgt = np.array([cup_xy[0] - dx_off, cup_xy[1]])
    for _ in range(10):
        can_xy = env.get_object_pose(can_name)[0][:2]
        err = tgt - can_xy
        if float(np.linalg.norm(err)) < 0.010:
            break
        ee = sk.ee()
        sk.move_to([ee[0] + err[0], ee[1] + err[1], POUR_Z], grip=0.0, tol=0.008,
                   max_steps=80, ki=move_ki)
    can_xy = env.get_object_pose(can_name)[0][:2]
    print(f"  [pour {can_name}] can_xy={np.round(can_xy,3)} tgt={np.round(tgt,3)}", flush=True)
    # 关节空间翻罐倾倒（不爆炸、真正翻过水平）
    ok_tip, pred = sk.tip_pour_joint(axis="ny", total_angle=np.radians(tip_deg),
                                     n=50, grip=0.0, hold=110)
    print(f"  [tip {can_name}] pred_can_mz={pred:+.2f} flipped={ok_tip}", flush=True)
    sk.untip_joint(grip=0.0)
    sk.retreat(dz=0.14)


def put_back_can(env, sk, arm, can_name, dst_xy):
    """把所持 can 放回桌面 dst_xy。"""
    sk.move_above([dst_xy[0], dst_xy[1]], high_z=TABLE_TOP_Z + 0.32)
    sk.refine_above(np.asarray(dst_xy), high_z=TABLE_TOP_Z + 0.32, obj_name=can_name)
    sk.descend_place_tracked(np.asarray(dst_xy), sp.TABLE_H, obj_name=can_name,
                             gap=0.004, max_steps=240)
    sk.place_release()
    sk.retreat(dz=0.18)


def in_cup(ball_xy, ball_z, cup_xy):
    xy_off = float(np.linalg.norm(np.asarray(ball_xy) - np.asarray(cup_xy)))
    return xy_off, (xy_off < sp.CUP_R_IN + 0.012) and (ball_z < CUP_TOP + 0.02) and \
        (ball_z > TABLE_TOP_Z - 0.01)


def main():
    DIAG = os.environ.get("POUR_DIAG", "0") == "1"
    PHASE = os.environ.get("POUR_PHASE", "both")   # both / L / R
    env = Exp4Env(sp.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=["canL", "ballL", "canR", "ballR"],
                  render=not DIAG, img_size=560)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]
    aL.grasp_z_tol = 0.09
    aR.grasp_z_tol = 0.09
    skL = ArmSkills(env, aL, recorder=lambda: env.record_frame("overview"),
                    rec_every=2, block_half=sp.CAN_HH)
    skR = ArmSkills(env, aR, recorder=lambda: env.record_frame("overview"),
                    rec_every=2, block_half=sp.CAN_HH)

    env.start_record()
    for _ in range(10):
        env.record_frame("overview")
    env.save_png(os.path.join(OUT, "pour_before.png"), "overview")

    cupR_xy = env.get_body_pos("cupR")[:2]
    cupL_xy = env.get_body_pos("cupL")[:2]
    okL = okR = True
    ballL = env.get_object_pose("ballL")[0]
    ballR = env.get_object_pose("ballR")[0]
    offL = offR = 0.0

    # ===== 阶段①：左臂抓 canL 倒入固定 cupR（右区）→ ballL 入 cupR =====
    if PHASE in ("both", "L"):
        print("[phase1] L pours canL into fixed cupR", flush=True)
        pick_can(env, skL, aL, "canL")
        pour_over_cup(env, skL, "canL", cupR_xy, arc_sign=+1)
        put_back_can(env, skL, aL, "canL", sp.CANL_XY)
        for _ in range(40):
            env.hold_arms(grips={aL.s: 1.0, aR.s: 1.0}); env.record_frame("overview")
        ballL = env.get_object_pose("ballL")[0]
        offL, okL = in_cup(ballL[:2], ballL[2], cupR_xy)
        print(f"[phase1 result] ballL={np.round(ballL,3)} cupR={np.round(cupR_xy,3)} "
              f"xy_off={offL:.4f} in_cupR={okL}", flush=True)

    # ===== 阶段②：右臂抓 canR 倒入固定 cupL（左区）→ ballR 入 cupL =====
    if PHASE in ("both", "R"):
        print("[phase2] R pours canR into fixed cupL", flush=True)
        pick_can(env, skR, aR, "canR")
        pour_over_cup(env, skR, "canR", cupL_xy, arc_sign=-1)
        put_back_can(env, skR, aR, "canR", sp.CANR_XY)
        for _ in range(40):
            env.hold_arms(grips={aL.s: 1.0, aR.s: 1.0}); env.record_frame("overview")
        ballR = env.get_object_pose("ballR")[0]
        offR, okR = in_cup(ballR[:2], ballR[2], cupL_xy)
        print(f"[phase2 result] ballR={np.round(ballR,3)} cupL={np.round(cupL_xy,3)} "
              f"xy_off={offR:.4f} in_cupL={okR}", flush=True)

    success = bool(okL and okR)
    print(f"[result] success={success} (ballL->cupR={okL}, ballR->cupL={okR})", flush=True)

    env.save_png(os.path.join(OUT, "pour_after.png"), "overview")
    env.save_video(os.path.join(OUT, "rollout.mp4"), fps=30)
    res = {"task": "pour", "success": success,
           "phase1": {"ballL": [round(float(v), 4) for v in ballL], "in_cupR": bool(okL),
                      "xy_off": round(offL, 4)},
           "phase2": {"ballR": [round(float(v), 4) for v in ballR], "in_cupL": bool(okR),
                      "xy_off": round(offR, 4)}}
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[saved]", OUT, flush=True)
    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
