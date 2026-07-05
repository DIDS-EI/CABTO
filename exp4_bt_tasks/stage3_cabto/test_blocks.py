"""test_blocks —— stage3 地基自测（不靠 VLM，用真值坐标驱动）
==============================================================
做一次完整 pick green_block -> place on yellow_block，断言：
- 抓起后 is_holding('green_block') 且 lifted(green_block)
- 放置稳定后 stacked_on(green_block, yellow_block)
录制 overview 视频 + 首末帧 png。
"""

import os
import numpy as np

from blocks_env import BlocksEnv, GRASP_QUAT, TABLE_TOP_Z, BLOCK_HALF, MAX_DPOS
import world_state as ws

OUT = "/tmp/exp4_stage3"
os.makedirs(OUT, exist_ok=True)

REC_EVERY = 2          # 每隔几步录一帧，控制视频长度
_step_counter = [0]


def move_to(env, target_xyz, grip=1.0, tol=0.01, max_steps=90, ki=0.0, quat=None, rec=True):
    """闭环把 tcp 推向 target_xyz（保持朝下姿态）。ki>0 时加积分项消除稳态偏差。"""
    target = np.asarray(target_xyz, float)
    integ = np.zeros(3)
    err = np.zeros(3)
    for i in range(max_steps):
        cur = env.get_ee_pose()[0]
        err = target - cur
        if np.linalg.norm(err) < tol:
            break
        if ki > 0:
            integ += err
            cmd = err + ki * integ
        else:
            cmd = err
        action = np.zeros(7)
        action[:3] = np.clip(cmd, -MAX_DPOS, MAX_DPOS)
        action[6] = grip
        env.step(action)
        _step_counter[0] += 1
        if rec and _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")
    final = float(np.linalg.norm(target - env.get_ee_pose()[0]))
    return final


def hold(env, grip, n=10, rec=True):
    """原地保持 n 步（开/合夹爪用）。"""
    for _ in range(n):
        env.step(np.array([0, 0, 0, 0, 0, 0, grip], float))
        _step_counter[0] += 1
        if rec and _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")


def descend_vertical(env, target_z, grip, max_steps=80, rec=True, lock_xy=None):
    """纯竖直下降：每步只动 z，xy 锁定在 lock_xy（默认当前 tcp xy），避免拖动/刮碰旁物。"""
    if lock_xy is None:
        lock_xy = env.get_ee_pose()[0][:2].copy()
    integ = np.zeros(2)
    for _ in range(max_steps):
        cur = env.get_ee_pose()[0]
        xy_err = float(np.linalg.norm(np.asarray(lock_xy, float) - cur[:2]))
        if abs(cur[2] - target_z) < 0.005 and xy_err < 0.006:
            break
        err_xy = np.asarray(lock_xy, float) - cur[:2]
        integ += err_xy
        action = np.zeros(7)
        # xy 用积分修正，限幅放宽以克服 EE_SERVO_BIAS 前馈稳态偏移
        action[:2] = np.clip(err_xy + 0.6 * integ, -0.03, 0.03)
        action[2] = np.clip(target_z - cur[2], -0.02, 0.02)   # 慢速下降
        action[6] = grip
        env.step(action)
        _step_counter[0] += 1
        if rec and _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")


def open_in_place(env, n=14, rec=True):
    """原地张开夹爪：dpos 恒为 0（IK 目标 = 当前 tcp，不引入横移），仅 grip 渐开。
    grasp_assist 在 grip>0.5 时解除 weld，绿块自然坐到下方物体上。"""
    for _ in range(n):
        env.step(np.array([0, 0, 0, 0, 0, 0, 1.0], float))
        _step_counter[0] += 1
        if rec and _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")


def hold_open_lock(env, n=20, rec=True):
    """张开夹爪并原地保持（dpos=0），让物体在当前位置靠重力坐稳。"""
    for _ in range(n):
        env.step(np.array([0, 0, 0, 0, 0, 0, 1.0], float))
        _step_counter[0] += 1
        if rec and _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")


def close_following(env, target_xy, n=20, rec=True):
    """慢闭合，同时把 tcp 持续拉回 target_xy（消除单边接触把块推开），
    z 几乎不动。配合 env 内闭合斜坡 + grasp_assist 对中夹取。"""
    integ = np.zeros(2)
    for _ in range(n):
        cur = env.get_ee_pose()[0]
        err_xy = np.asarray(target_xy, float) - cur[:2]
        integ += err_xy
        cmd = err_xy + 0.5 * integ
        action = np.zeros(7)
        action[:2] = np.clip(cmd, -0.02, 0.02)
        action[2] = 0.0
        action[6] = 0.0
        env.step(action)
        _step_counter[0] += 1
        if rec and _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")


def pp(env, label):
    ee = env.get_ee_pose()[0]
    g = env.get_object_pose("green_block")[0]
    y = env.get_object_pose("yellow_block")[0]
    print(f"[{label:14s}] tcp=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
          f"green=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) "
          f"yellow=({y[0]:.3f},{y[1]:.3f},{y[2]:.3f}) "
          f"hold(green)={env.is_holding('green_block')} grip_w={env.get_gripper_width():.3f}")


def main():
    env = BlocksEnv(render=True, img_size=512)
    env.reset()
    env.start_record()
    env.record_frame("overview")
    pp(env, "reset")

    # 抓取高度：朝下抓取。抓绿块上部（tcp/指尖中点靠近绿块顶），这样放置后
    # 指尖位于绿块顶附近，竖直抬手只需很小行程即可脱离，几乎不刮碰绿块。
    g_pos = env.get_object_pose("green_block")[0]
    g_top = g_pos[2] + BLOCK_HALF
    grasp_z = g_top - 0.30 * (2 * BLOCK_HALF)         # = top - 0.30*0.05，夹绿块上部
    above_z = TABLE_TOP_Z + 0.18

    # 1) approach green 正上方（粗移，纯比例）
    move_to(env, [g_pos[0], g_pos[1], above_z], grip=1.0, tol=0.012, ki=0.0)
    pp(env, "approach")
    # 2) 高处精对准 xy（积分消除稳态偏差，避免斜向下探碰倒块）
    move_to(env, [g_pos[0], g_pos[1], above_z], grip=1.0, tol=0.005, max_steps=50, ki=0.6)
    # 3) 张开夹爪 + 竖直下探到抓取高度（纯竖直，锁 xy 防碰倒块）
    hold(env, grip=1.0, n=4)
    g_pos = env.get_object_pose("green_block")[0]
    descend_vertical(env, grasp_z, grip=1.0, max_steps=80, lock_xy=g_pos[:2])
    pp(env, "descend")
    # 4) 闭合抓取：tcp 跟随块 xy，配合 grasp_assist 在指间检测到 green -> 激活 weld
    g_pos = env.get_object_pose("green_block")[0]
    close_following(env, g_pos[:2], n=24)
    pp(env, "close")
    # 5) 抬起
    move_to(env, [g_pos[0], g_pos[1], above_z + 0.06], grip=0.0, tol=0.02, max_steps=70, ki=0.3)
    pp(env, "lift")

    holding = env.is_holding("green_block")
    is_lifted = ws.lifted(env, "green_block")
    print(f"\n[CHECK pick] is_holding(green)={holding}  lifted(green)={is_lifted}")
    assert holding, "未抓住 green_block"
    assert is_lifted, "green_block 未抬起"

    # 抓取后实测绿块中心相对 tcp 的 z 偏移（weld 锁定后恒定），用于精确换算放置高度
    ee_after = env.get_ee_pose()[0]
    g_after = env.get_object_pose("green_block")[0]
    grab_dz = g_after[2] - ee_after[2]        # 绿块中心相对 tcp 的 z 偏移（负，块在 tcp 下方）
    grab_dxy = g_after[:2] - ee_after[:2]     # 绿块中心相对 tcp 的 xy 偏移（抓偏量）
    print(f"[grab] dz={grab_dz:.4f} dxy=({grab_dxy[0]:.4f},{grab_dxy[1]:.4f})")

    # 6) 高空（远高于黄块）移到 yellow 正上方：补偿抓偏量，让「绿块」对准 yellow 中心
    y_pos = env.get_object_pose("yellow_block")[0]
    y_top = y_pos[2] + BLOCK_HALF
    high_z = TABLE_TOP_Z + 0.30        # 抬到很高再平移，绝不刮到黄块
    tcp_xy = np.array([y_pos[0], y_pos[1]]) - grab_dxy
    move_to(env, [tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.010, max_steps=90, ki=0.0)
    move_to(env, [tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.004, max_steps=50, ki=0.6)
    # 重新读绿块真值，按当前实际抓偏再精修一次 tcp_xy，使绿块正对黄块中心
    ee_now = env.get_ee_pose()[0]
    g_now = env.get_object_pose("green_block")[0]
    cur_dxy = g_now[:2] - ee_now[:2]
    tcp_xy = np.array([y_pos[0], y_pos[1]]) - cur_dxy
    move_to(env, [tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.004, max_steps=40, ki=0.6)
    pp(env, "move_above_y")

    # 7) 纯竖直下放：绿块底面落到 yellow 顶面上方 ~3mm（不下压，避免 weld 挤压弹飞）
    green_center_target = y_top + BLOCK_HALF + 0.003
    place_z = green_center_target - grab_dz
    descend_vertical(env, place_z, grip=0.0, max_steps=140, lock_xy=tcp_xy)
    pp(env, "descend_place")

    # 关键收尾顺序（避免松手弹飞）。核心：用 env.hold_arm 冻结手臂关节角，绕过
    # step 内 EE_SERVO_BIAS 前馈对 hand 的持续横推（之前漂移/带飞物体的真正元凶）。
    #  a) 禁用自动 grasp_assist，weld 状态完全手动控制。
    #  b) hold_arm 张开夹爪：手臂关节冻结不漂，weld 仍锁定，指尖张开离开绿块两侧。
    #  c) 手动解 weld：绿块脱离 hand，靠重力落 ~3mm 坐到黄块顶面（指尖已让开，不带飞）。
    #  d) hold_arm 原地稳定让接触建立。
    env._grasp_enabled = False
    for _ in range(12):               # (b) 冻结手臂、张开夹爪，weld 保持
        env.hold_arm(grip=1.0)
        _step_counter[0] += 1
        if _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")
    pp(env, "open_keepweld")
    env._set_all_welds_inactive()     # (c) 解 weld，绿块坐下
    for _ in range(24):               # (d) 冻结手臂原地稳定
        env.hold_arm(grip=1.0)
        _step_counter[0] += 1
        if _step_counter[0] % REC_EVERY == 0:
            env.record_frame("overview")
    pp(env, "release")
    # 9) 竖直抬手：move_to 竖直抬升 + 强 xy 积分（ki=0.9）压住 EE_SERVO_BIAS 横移，
    #    指尖竖直离开绿块顶（抓上部使指尖此时已接近绿块顶，行程小）。
    rel_xy = env.get_ee_pose()[0][:2].copy()
    move_to(env, [rel_xy[0], rel_xy[1], above_z + 0.10], grip=1.0,
            tol=0.02, max_steps=80, ki=0.9)
    pp(env, "retreat")

    # 10) settle 让物体落稳
    hold(env, grip=1.0, n=40)
    pp(env, "settle")

    # 末帧
    env.record_frame("overview")

    # 10) 校验 stacked_on
    st = ws.compute_state(env)
    stacked = st["stacked_on(green_block,yellow_block)"]
    print("\n=== 最终谓词（TRUE）===")
    for k in sorted(st):
        if st[k]:
            print("  ", k)
    g = env.get_object_pose("green_block")[0]
    y = env.get_object_pose("yellow_block")[0]
    print(f"\ngreen底面={g[2]-BLOCK_HALF:.4f}  yellow顶面={y[2]+BLOCK_HALF:.4f}  "
          f"xy偏移={np.linalg.norm(g[:2]-y[:2]):.4f}")
    print(f"[CHECK place] stacked_on(green,yellow)={stacked}")

    # 保存视频 + 首末帧
    env.save_video(os.path.join(OUT, "test_blocks.mp4"), fps=30)
    import imageio
    # 首帧用 reset 状态重渲染较麻烦，这里用录制缓存的首末帧
    if env._rec_frames:
        imageio.imwrite(os.path.join(OUT, "frame_first.png"), env._rec_frames[0])
        imageio.imwrite(os.path.join(OUT, "frame_last.png"), env._rec_frames[-1])
        print(f"[ok] 首末帧已保存（共 {len(env._rec_frames)} 帧）")

    assert stacked, "stacked_on(green_block, yellow_block) 断言失败"
    print("\n*** 成功：green_block 已堆叠到 yellow_block 上 ***")


if __name__ == "__main__":
    main()
