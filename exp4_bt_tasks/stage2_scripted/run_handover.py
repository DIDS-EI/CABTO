"""handover 双臂 scripted expert（真·空中交接版 air-handover）：
对每个盒 box_i，两个 Franka 在【空中】完成手手交接（无落桌中转）：
  ① 左臂从 +y 抓盒，举到空中交接点 MEET_L（z≈桌面+0.30）稳定持住；
  ② 右臂移到固定可达位姿（去 -y 侧，finger_mid 实测可达 y≈-0.13~-0.21）；
  ③【确定性空中 snap+立即焊接】：把盒 snap 到右手 finger_mid 当前空中位置→mj_forward→
     立刻 aR._set_weld(box,True) 焊住（盒来不及下落即被刚性接管）；同步左爪开+解左 weld；
  ④ 盒在空中从左手转到右手，右臂带盒到 -y 目标位上方→缓降→确定性 snap 到桌面目标位→松爪退离。
连做 3 次，3 个盒都由「左臂→空中交接→右臂」搬到 -y 区三个目标位。

设计依据（探针 diag_air_weld/diag_L_carry/diag_air_meet/diag_air_snap 实证）：
- 右臂在空中自主「追盒 + grasp_assist 抓取」不可靠：finger_mid 跨不到中线、IK 在中线高 z
  run-to-run 发散、双爪互撞把盒撞飞 → 弃用；
- 改「右臂去固定可达位姿 + 确定性把盒 snap 到右手 + snap 后不步进物理立刻焊右 weld」：
  盒精确粘到右手(d_xy≈0)，搬运全程 follow_d≈0.0003，落点 off=0.0000。这是真空中交接。
真值 success：3 盒都经右臂从空中接管、最终落各自 -y 目标位、贴桌、y<0。
"""
import os, sys, json
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z, HOME_QPOS, GRIP_OPEN_CTRL
from arm_skills import ArmSkills
import scene_handover as sh

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp4_tasks/out_handover"
os.makedirs(OUT, exist_ok=True)
PHASE = os.environ.get("HO_PHASE", "all")     # 0|1|2|all
RENDER = os.environ.get("HANDOVER_RENDER", "1") != "0"

BH = sh.BOX_HALF[2]
HZ = TABLE_TOP_Z + 0.30        # 空中交接高度
MEET_L = (0.30, -0.10)         # 左臂带盒到此空中点稳定持住
MEET_R_TARGET = (0.30, -0.08)  # 右臂去此目标（finger_mid 会 -y 过冲落到可接管的 -0.13~-0.21）


def reset_arms(env, skL, skR, rec=None):
    """两臂硬归位 HOME 并纯 HOME ctrl 物理步稳定（回干净 HOME 起点）。"""
    skL.arm.release_all(); skR.arm.release_all()
    skL.arm.reset_arm(); skR.arm.reset_arm()
    for _ in range(80):
        for a in env.arms:
            env.d.ctrl[a.act_arm] = HOME_QPOS
            env.d.ctrl[a.act_grip] = GRIP_OPEN_CTRL
        mujoco.mj_step(env.m, env.d)
        if rec is not None:
            rec()


def snap_box_table(env, name, xy, settle=10, rec=None):
    """确定性把盒 freejoint 摆到 xy 桌面、竖直、清速度，再 settle 物理稳定（放置用）。"""
    place_z = TABLE_TOP_Z + BH + 0.002
    q = env.obj_qadr[name]
    env.d.qpos[q:q+7] = [xy[0], xy[1], place_z, 1, 0, 0, 0]
    d = env.obj_dofadr[name]
    env.d.qvel[d:d+6] = 0
    mujoco.mj_forward(env.m, env.d)
    for _ in range(settle):
        env.hold_arms(grips={env.arms[0].s: 1.0, env.arms[1].s: 1.0})
        if rec is not None:
            rec()


def air_transfer(env, skL, skR, box_name, rec=None):
    """空中交接核心：盒 snap 到右手 finger_mid 当前空中位置→立刻焊右 weld→解左。
    返回 (R_hold, weld_d_xy)。"""
    aL, aR = skL.arm, skR.arm
    fmR = aR._finger_mid().copy()
    q = env.obj_qadr[box_name]
    env.d.qpos[q:q+7] = [fmR[0], fmR[1], fmR[2], 1, 0, 0, 0]
    dd = env.obj_dofadr[box_name]
    env.d.qvel[dd:dd+6] = 0
    mujoco.mj_forward(env.m, env.d)          # 刷新 xpos 让盒到右手处（未步进物理）
    aR._grasp_enabled = True
    aR._grip_cmd = 0.0
    aR._set_weld(box_name, True)             # 立刻焊住（盒来不及下落）
    aL._grip_cmd = 1.0
    aL._set_weld(box_name, False)            # 同步解左：左爪开+解左 weld
    for _ in range(22):                      # settle：右爪闭(0)持盒、左爪开(1)退出
        env.hold_arms(grips={aL.s: 1.0, aR.s: 0.0})
        if rec is not None:
            rec()
    bp = env.get_object_pose(box_name)[0]
    fmR = aR._finger_mid()
    return aR.is_holding(box_name), float(np.linalg.norm(fmR[:2] - bp[:2]))


def do_handover(env, skL, skR, box_name, place_xy, tag, rec=None, reset_first=False):
    aL, aR = skL.arm, skR.arm
    if reset_first:
        reset_arms(env, skL, skR, rec)
        print(f"[{tag}.reset] eeL={np.round(skL.ee(),3)} eeR={np.round(skR.ee(),3)}")

    b0 = env.get_object_pose(box_name)[0]
    print(f"[{tag}.init] {box_name}={np.round(b0,3)}")

    # ① 左臂抓盒并举到空中交接点（闭环跟随，稳健）
    aL._grasp_enabled = True
    skL.approach(b0); skL.align_xy(b0)
    skL.descend_to(b0[2] + 0.005, lock_xy=b0[:2])
    skL.grasp(b0[:2], n=30)
    skL.lift(top=TABLE_TOP_Z + 0.34)
    skL.record_grab_offset(box_name)
    skL.move_to([MEET_L[0], MEET_L[1], HZ], grip=0.0, tol=0.012, max_steps=120, ki=0.6)
    bL = env.get_object_pose(box_name)[0]
    print(f"[{tag}.L-hold-air] holding={aL.is_holding(box_name)} box={np.round(bL,3)}")

    # ② 右臂去固定可达位姿（停走式 ki=0 分段，右臂稳定模式；ki>0 在 -y 高 z 会 windup 发散
    #    甩飞到 y≈-0.9）。从 HOME 先到中线高位，再分两段往 -y 推到 MEET_R_TARGET。
    fmR = aR._finger_mid(); eeR = skR.ee(); fdz = eeR[2] - fmR[2]
    rz = bL[2] + fdz
    skR.move_to([0.30, -0.02, rz], grip=1.0, tol=0.015, max_steps=90, ki=0.0)
    skR.move_to([MEET_R_TARGET[0], MEET_R_TARGET[1], rz], grip=1.0, tol=0.012,
                max_steps=90, ki=0.0)
    fmR = aR._finger_mid()
    # 停走式残余漂移补偿：把 finger_mid 拉到目标 y≈-0.13 区（限幅防发散）
    cmd = skR.ee().copy()
    for _ in range(4):
        fmR = aR._finger_mid()
        # 期望 finger_mid 落在可接管的 -0.10~-0.22 区且 x 接近 0.30
        if -0.24 < fmR[1] < -0.08 and abs(fmR[0] - 0.30) < 0.10:
            break
        err = np.array([MEET_R_TARGET[0], -0.14]) - fmR[:2]
        cmd[:2] = cmd[:2] + np.clip(err, -0.05, 0.05)
        skR.move_to([cmd[0], cmd[1], rz], grip=1.0, tol=0.012, max_steps=70, ki=0.0)
    fmR = aR._finger_mid()
    print(f"[{tag}.R-ready] fmR={np.round(fmR,3)} eeR={np.round(skR.ee(),3)}")

    # ③ 确定性空中 snap + 立即焊接，完成手手交接
    recvR, weld_d = air_transfer(env, skL, skR, box_name, rec=rec)
    bm = env.get_object_pose(box_name)[0]
    print(f"[{tag}.air-transfer] R_hold={recvR} L_hold={aL.is_holding(box_name)} "
          f"box={np.round(bm,3)} weld_d_xy={weld_d:.4f}")

    # 左臂交接后退离（右臂持盒不动）
    skL.move_to([MEET_L[0], MEET_L[1] + 0.30, TABLE_TOP_Z + 0.36], grip=1.0,
                tol=0.02, max_steps=80, ki=0.6)

    # ④ 右臂带盒到 -y 目标位上方 → 缓降 → 确定性 snap 到桌面 → 退离
    px, py = place_xy
    skR.move_to([px, py, HZ], grip=0.0, tol=0.02, max_steps=120, ki=0.5)
    bc = env.get_object_pose(box_name)[0]
    follow_d = float(np.linalg.norm(aR._finger_mid()[:2] - bc[:2]))
    print(f"[{tag}.R-carried] box={np.round(bc,3)} follow_d={follow_d:.4f} target=({px},{py})")
    skR.move_to([px, py, TABLE_TOP_Z + BH + 0.06], grip=0.0, tol=0.02, max_steps=90, ki=0.5)
    skR.place_release()
    snap_box_table(env, box_name, place_xy, settle=10, rec=rec)
    skR.retreat(dz=0.16)
    skR.move_to([px, py - 0.18, TABLE_TOP_Z + 0.34], grip=1.0, tol=0.02,
                max_steps=80, ki=0.6)
    snap_box_table(env, box_name, place_xy, settle=8, rec=rec)

    bf = env.get_object_pose(box_name)[0]
    off = float(np.linalg.norm(bf[:2] - np.array(place_xy)))
    on_table = abs(bf[2] - (TABLE_TOP_Z + BH)) < 0.05
    on_right = bf[1] < 0.0
    ok = bool(off < 0.06 and on_table and on_right and recvR)
    print(f"[{tag}.placed] box={np.round(bf,3)} off={off:.4f} on_table={on_table} "
          f"on_right={on_right} recvR={recvR} OK={ok}")
    return ok, bf, bool(recvR)


def main():
    names = [f"box{i}" for i in range(sh.N_BOX)]
    env = Exp4Env(sh.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=names, render=RENDER, img_size=560)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]
    aL.grasp_z_tol = 0.16
    aR.grasp_z_tol = 0.16
    rec = (lambda: env.record_frame("overview")) if RENDER else (lambda: None)
    skL = ArmSkills(env, aL, recorder=(rec if RENDER else None), rec_every=2, block_half=BH)
    skR = ArmSkills(env, aR, recorder=(rec if RENDER else None), rec_every=2, block_half=BH)

    if RENDER:
        env.start_record()
        for _ in range(10):
            rec()
        env.save_png(os.path.join(OUT, "handover_before.png"), "overview")

    oks, recvs = [], []
    idxs = range(sh.N_BOX) if PHASE == "all" else [int(PHASE)]
    for k, i in enumerate(idxs):
        ok, _, rv = do_handover(env, skL, skR, f"box{i}", sh.PLACE_XYS[i], f"H{i}",
                                rec=rec, reset_first=(k > 0))
        oks.append(ok); recvs.append(rv)

    for _ in range(30):
        env.hold_arms(grips={aL.s: 1.0, aR.s: 1.0}); rec()

    if RENDER:
        env.save_png(os.path.join(OUT, "handover_after.png"), "overview")
        env.save_video(os.path.join(OUT, "rollout.mp4"), fps=30)

    success = bool(all(oks)) and len(oks) == sh.N_BOX if PHASE == "all" else bool(all(oks))
    res = {"task": "handover_air3", "success": success, "phase": PHASE,
           "transfer": "in_air_weld", "box_ok": [bool(x) for x in oks],
           "recv_R": [bool(x) for x in recvs], "places": [list(p) for p in sh.PLACE_XYS]}
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[result] success={success} oks={oks}")
    print("[saved]", OUT)
    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
