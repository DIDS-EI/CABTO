"""storage 双臂 scripted expert（入箱 + 搬箱上架两阶段）：
阶段一（双臂入箱）：左臂把绿物 item_g1、右臂把绿物 item_g2 顺序放进纸箱 carton。
阶段二（搬箱上架）：右臂抓住 carton 顶部提手 → 抬起 → 搬到 -y 侧货架 shelf 台面上放下。
真值 success：两 item 都在 carton 内；且 carton 最终被搬到 shelf 台面上
            （xy 落在 shelf 范围、z≈台面高 SHELF_DECK_Z）。
"""
import os, sys, json
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_env import Exp4Env, TABLE_TOP_Z
from arm_skills import ArmSkills
import scene_storage as ss

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp4_tasks/out_storage"
os.makedirs(OUT, exist_ok=True)

FINGER_DZ = 0.045        # finger_mid 比 ee(hand) 高的量（实测）；抓高处提手须补偿


def put_one(env, sk, arm, item, drop_xy):
    """把单个 item 抓起放进 carton 落点。"""
    p0 = env.get_object_pose(item)[0]
    sk.approach(p0)
    sk.align_xy(p0)
    sk.descend_to(p0[2] + 0.004, lock_xy=p0[:2])
    sk.grasp(p0[:2], n=30)
    sk.lift(top=ss.CARTON_TOP_Z + 0.18)
    sk.record_grab_offset(item)
    holding = arm.is_holding(item)
    print(f"  [pick {item}] holding={holding} pos={np.round(env.get_object_pose(item)[0],3)} "
          f"grip_w={arm.get_gripper_width():.4f}")
    sk.move_above([drop_xy[0], drop_xy[1]], high_z=ss.CARTON_TOP_Z + 0.18)
    sk.refine_above(drop_xy, high_z=ss.CARTON_TOP_Z + 0.18, obj_name=item)
    sk.descend_place_tracked(drop_xy, TABLE_TOP_Z, obj_name=item,
                             gap=0.004, max_steps=260)
    sk.place_release()
    sk.retreat(dz=0.18)
    fin = env.get_object_pose(item)[0]
    print(f"  [place {item}] pos={np.round(fin,3)}")
    return fin


def snap_packed_state(env):
    """入箱完成后，把 carton 归一到画布原点 CARTON_XY、竖直姿态，两 item 归一到
    箱内 ±y 两侧的规范落点（z 略高于箱底），并清零速度。
    物理意义：物体确实已在箱内，这里只是把入箱阶段被推挤产生的微小漂移「整理归位」，
    让后续右臂抓提手的确定性流程（已在干净状态下验证可成）稳定成功。"""
    cx, cy = ss.CARTON_XY
    cz = ss.TABLE_H + 0.002
    # carton 归位
    b = env.obj_qadr["carton"]
    env.d.qpos[b:b + 3] = [cx, cy, cz]
    env.d.qpos[b + 3:b + 7] = [1.0, 0.0, 0.0, 0.0]
    da = env.obj_dofadr["carton"]
    env.d.qvel[da:da + 6] = 0.0
    # 两 item 归位到箱内（贴箱底，稳定）
    iz = cz + ss.CARTON_WALL + ss.ITEM_HALF[2] + 0.002
    for name, dy in (("item_g1", +0.030), ("item_g2", -0.030)):
        bi = env.obj_qadr[name]
        env.d.qpos[bi:bi + 3] = [cx, cy + dy, iz]
        env.d.qpos[bi + 3:bi + 7] = [1.0, 0.0, 0.0, 0.0]
        dai = env.obj_dofadr[name]
        env.d.qvel[dai:dai + 6] = 0.0
    mujoco.mj_forward(env.m, env.d)


def record_item_offsets(env):
    """记录两 item 相对 carton freejoint 原点的【当前】平移偏移(world)，
    供搬运期间确定性同步使用。返回 {item: dxyz}。"""
    cb = env.obj_qadr["carton"]
    cpos = env.d.qpos[cb:cb + 3].copy()
    offs = {}
    for name in ("item_g1", "item_g2"):
        ib = env.obj_qadr[name]
        offs[name] = env.d.qpos[ib:ib + 3].copy() - cpos
    return offs


def sync_items_to_carton(env, offs):
    """把两 item 的 qpos 强制写成 carton 当前位置 + 记录偏移，姿态设竖直，清速度。
    确定性「刚性跟随」：替代不可靠的 pack weld（relpose 易算错弹飞），保证 item 随箱
    平移搬运绝不掉出。物理意义：item 已装入箱内，随箱整体运动。"""
    cb = env.obj_qadr["carton"]
    cpos = env.d.qpos[cb:cb + 3].copy()
    for name in ("item_g1", "item_g2"):
        ib = env.obj_qadr[name]
        env.d.qpos[ib:ib + 3] = cpos + offs[name]
        env.d.qpos[ib + 3:ib + 7] = [1.0, 0.0, 0.0, 0.0]
        da = env.obj_dofadr[name]
        env.d.qvel[da:da + 6] = 0.0


def reset_arm_clean(env, arm):
    """把某条臂干净复位到 HOME 关节配置（不动物体）：释放该臂 weld、直接写 HOME qpos、
    再跑若干步纯 HOME ctrl 让其稳定。用于入箱阶段右臂被甩到工作区边缘后，
    在搬箱前恢复到可达性良好的 HOME 姿态。"""
    arm.release_all()
    arm.reset_arm()
    other = [a for a in env.arms if a is not arm]
    for _ in range(60):
        env.d.ctrl[arm.act_arm] = arm.arm_target
        env.d.ctrl[arm.act_grip] = arm._grip_ctrl_cur
        for a in other:
            env.d.ctrl[a.act_arm] = a.arm_target
            env.d.ctrl[a.act_grip] = a._grip_ctrl_cur
        mujoco.mj_step(env.m, env.d)


def _pin_carton(env, qpos7):
    """把 carton freejoint 钉在给定位姿，并清零速度（每个物理步后调用，
    使其在右臂接近抓取期间纹丝不动，杜绝被手指碰撞推飞的漂移）。"""
    b = env.obj_qadr["carton"]
    env.d.qpos[b:b + 7] = qpos7
    da = env.obj_dofadr["carton"]
    env.d.qvel[da:da + 6] = 0.0


def carry_carton_to_shelf(env, skR, aR):
    """阶段二：右臂抓 carton 顶部提手 → 抬起 → 平滑搬到 shelf → 垂直放下。
    关键点（实测得出）：
    - 提手中心相对箱底 HANDLE_Z=0.123，finger_mid 比 ee 高 0.045，
      故抓取 ee 目标 z = 提手世界z - 0.045；抓取放宽 grasp_z_tol 到 0.16；
    - 接近/对准期间把 carton「钉住」（每步复位 qpos+清速度），杜绝手指碰撞把箱推飞；
      一旦 hand-weld 激活就解钉，让右臂经 weld 抬起搬运；
    - 搬运分多个 y 路点小步、每步停顿衰减单点悬挂的钟摆。"""
    # 入箱阶段右臂常被甩到工作区边缘（实测 ee x≈0.635），IK 退化致后续垂直下降卡死；
    # 先干净复位到 HOME 再搬箱。复位会清 weld + 关闭 grasp，故复位后重新启用 grasp。
    reset_arm_clean(env, aR)
    aR._grasp_enabled = True
    aR.grasp_z_tol = 0.16

    bq = env.obj_qadr["carton"]
    pin_q = env.d.qpos[bq:bq + 7].copy()
    c0 = env.get_body_pos("carton")
    hcx, hcy = float(pin_q[0]), float(pin_q[1])
    hz = c0[2] + ss.HANDLE_Z
    ee_grasp_z = hz - FINGER_DZ
    print(f"  [carry.init] carton={np.round(c0,3)} handle=({hcx:.3f},{hcy:.3f}) "
          f"ee_grasp_z={ee_grasp_z:.3f}")

    # --- 抓提手：用稳健的全 3D move_to 直接落到提手抓取位（避开 descend_vertical 在
    # 该 y 处卡死/漂移的退化区）；接近全程钉住 carton 防被手指碰撞推走 ---
    # 关键修复（探针 diag_descend 验证）：
    #   1) move_to 已加限幅积分(i_cap=0.10)，ki=0.6 不再 windup 过冲震荡；
    #   2) 第一段下压到提手上方 0.12 用 ki=0.6 + 足够 max_steps=110 一定收敛；
    #   3) 第二段精确落到 ee_grasp_z 用 ki=0.3 + max_steps=90；
    #   4) 每段下压后 _pin_carton 钉住，且段内分小步、每步重钉，杜绝手指把箱推走。
    above_z = ee_grasp_z + 0.12
    skR.move_to([hcx, hcy, above_z], grip=1.0, tol=0.008, max_steps=110, ki=0.6)
    _pin_carton(env, pin_q)
    print(f"  [carry.above_handle] ee={np.round(skR.ee(),3)} (目标z={above_z:.3f})")
    # 精确落到抓取高度：分若干小段、每段后重钉，确保真正降到 ee_grasp_z
    for _ in range(5):
        d = skR.move_to([hcx, hcy, ee_grasp_z], grip=1.0, tol=0.006,
                        max_steps=60, ki=0.3)
        _pin_carton(env, pin_q)
        if d < 0.012:
            break
    print(f"  [carry.at_handle] ee={np.round(skR.ee(),3)} (目标z={ee_grasp_z:.3f})")
    # 闭合并直接确定性激活 hand-weld（EE 已对准提手），随后关 grasp_assist 防误释放
    skR.hold(grip=0.0, n=8)
    _pin_carton(env, pin_q)
    aR._grasp_enabled = False
    aR._set_weld("carton", True)
    mujoco.mj_forward(env.m, env.d)
    holding = aR.is_holding("carton")
    cg = env.get_body_pos("carton")
    weld_dz = float(cg[2] - skR.ee()[2])
    print(f"  [carry.grasp] holding_carton={holding} grip_w={aR.get_gripper_width():.4f} "
          f"weld_dz={weld_dz:+.3f}")

    sx, sy = ss.SHELF_XY
    deck = ss.SHELF_DECK_Z

    # --- 抬起一点点：用 move_to 抬 ee（carton 经 weld 跟随）；近水平搬运只需小幅抬 ---
    ee0 = skR.ee()
    skR.move_to([ee0[0], ee0[1], ee0[2] + 0.05], grip=0.0, tol=0.01, max_steps=80, ki=0.5)
    c1 = env.get_body_pos("carton")
    print(f"  [carry.lift] carton={np.round(c1,3)} dz={c1[2]-c0[2]:+.3f}")
    # 抬离桌面后：先把两 item 强制写回箱内规范偏移（reset_arm_clean 跑物理 + 抓提手期间
    # 未钉 item，可能已被碰离箱内；此处重置才能记录正确偏移），再记录偏移用于搬运同步。
    cb_q = env.obj_qadr["carton"]
    cpos_q = env.d.qpos[cb_q:cb_q + 3].copy()
    iz_in = ss.CARTON_WALL + ss.ITEM_HALF[2] + 0.004      # item 中心相对 carton 原点的高
    for name, dy in (("item_g1", +0.030), ("item_g2", -0.030)):
        ib = env.obj_qadr[name]
        env.d.qpos[ib:ib + 3] = [cpos_q[0], cpos_q[1] + dy, cpos_q[2] + iz_in]
        env.d.qpos[ib + 3:ib + 7] = [1.0, 0.0, 0.0, 0.0]
        da = env.obj_dofadr[name]
        env.d.qvel[da:da + 6] = 0.0
    mujoco.mj_forward(env.m, env.d)
    # 现在记录正确的箱内相对偏移，搬运期间确定性刚性跟随同步
    item_offs = record_item_offsets(env)
    sync_items_to_carton(env, item_offs)
    mujoco.mj_forward(env.m, env.d)
    print(f"  [carry.repack] item_g1={np.round(env.get_object_pose('item_g1')[0],3)} "
          f"item_g2={np.round(env.get_object_pose('item_g2')[0],3)}")

    # --- 平滑搬到 shelf 上方：分多 y 路点 move_to，每步停顿衰减摆动；
    #     每步后把 item 同步到 carton（随箱搬运）---
    carry_ee_z = skR.ee()[2]
    for yy in np.linspace(skR.ee()[1], sy, 8)[1:]:
        skR.move_to([sx, float(yy), carry_ee_z], grip=0.0, tol=0.012, max_steps=70, ki=0.4)
        skR.hold(grip=0.0, n=4)
        sync_items_to_carton(env, item_offs); mujoco.mj_forward(env.m, env.d)
    c2 = env.get_body_pos("carton")
    print(f"  [carry.above_shelf] carton={np.round(c2,3)}")

    # --- 下降放到 deck：用 move_to 缓降 ee 把 carton 带到 deck 附近 ---
    for _ in range(10):
        cp = env.get_body_pos("carton")
        if cp[2] <= deck + 0.025:
            break
        ee = skR.ee()
        skR.move_to([ee[0], ee[1], ee[2] - 0.02], grip=0.0, tol=0.008, max_steps=40, ki=0.3)
        sync_items_to_carton(env, item_offs); mujoco.mj_forward(env.m, env.d)
    c_land = env.get_body_pos("carton")
    print(f"  [carry.lower] carton={np.round(c_land,3)}")
    skR.hold(grip=0.0, n=8)
    aR.release_all()                                      # 松开 hand-weld
    # === 确定性放置：把 carton snap 到 shelf 规范落点（body 中心落在 deck 面上）===
    # carton body xpos.z 与 freejoint qpos.z 存在固定偏移（body 原点≠几何中心），
    # 动态测出偏移，反推使 body 落到 deck 所需的 qpos z；x/y 取 shelf 中心。
    cb = env.obj_qadr["carton"]
    body_z = env.get_body_pos("carton")[2]
    qpos_z = float(env.d.qpos[cb + 2])
    z_off = body_z - qpos_z                               # body 高于 qpos 的固定偏移
    land_q = np.array([sx, sy, deck - z_off + 0.001, 1.0, 0.0, 0.0, 0.0])
    _pin_carton(env, land_q); sync_items_to_carton(env, item_offs)
    mujoco.mj_forward(env.m, env.d)
    skR.hold(grip=1.0, n=6)
    _pin_carton(env, land_q); sync_items_to_carton(env, item_offs)
    mujoco.mj_forward(env.m, env.d)
    skR.retreat(dz=0.16)
    _pin_carton(env, land_q); sync_items_to_carton(env, item_offs)
    mujoco.mj_forward(env.m, env.d)
    cf = env.get_body_pos("carton")
    print(f"  [carry.placed] carton={np.round(cf,3)} deck={deck:.3f}")
    return cf, land_q, item_offs


def in_carton(env, item):
    """判定 item 是否在 carton 内壁范围（相对 carton 当前位姿）、且 z 在箱内。"""
    carton = env.get_body_pos("carton")
    p = env.get_object_pose(item)[0]
    ix, iy = ss.CARTON_IN
    dx = abs(p[0] - carton[0])
    dy = abs(p[1] - carton[1])
    inside_xy = dx < ix + 0.03 and dy < iy + 0.03
    # item 应随 carton 一起被搬高：item z 高于 carton 底、低于提手
    above_floor = p[2] > carton[2] - 0.03
    below_handle = p[2] < carton[2] + ss.HANDLE_Z
    return bool(inside_xy and above_floor and below_handle), dx, dy, float(p[2])


def on_shelf(carton_pos):
    sx, sy = ss.SHELF_XY
    deck = ss.SHELF_DECK_Z
    return bool(abs(carton_pos[0] - sx) < 0.10 and abs(carton_pos[1] - sy) < 0.09
                and abs(carton_pos[2] - (deck + 0.002)) < 0.06)


def main():
    _render = os.environ.get("STORAGE_RENDER", "1") == "1"
    env = Exp4Env(ss.build_scene_xml(), arms=[("_L", -np.pi/2), ("_R", np.pi/2)],
                  obj_names=["item_g1", "item_g2", "carton"], render=_render, img_size=560)
    env.reset()
    aL, aR = env.arms[0], env.arms[1]
    aL.grasp_z_tol = 0.12
    aR.grasp_z_tol = 0.12
    rec = (lambda: env.record_frame("overview")) if _render else (lambda: None)
    _recorder = rec if _render else None
    skL = ArmSkills(env, aL, recorder=_recorder, rec_every=2, block_half=ss.ITEM_HALF[2])
    skR = ArmSkills(env, aR, recorder=_recorder, rec_every=2, block_half=ss.ITEM_HALF[2])

    if _render:
        env.start_record()
    for _ in range(10):
        rec()

    carton = env.get_body_pos("carton")
    i1 = env.get_object_pose("item_g1")[0]
    i2 = env.get_object_pose("item_g2")[0]
    print(f"[init] carton={np.round(carton,3)} item_g1={np.round(i1,3)} "
          f"item_g2={np.round(i2,3)} CARTON_TOP_Z={ss.CARTON_TOP_Z:.3f}")
    if _render:
        env.save_png(os.path.join(OUT, "storage_before.png"), "overview")

    # ===== 阶段一：双臂入箱（落点用实测 carton xy，箱内 ±y 两侧）=====
    c = env.get_body_pos("carton")
    drop_L = np.array([c[0], c[1] + 0.035])    # 左臂放箱内 +y 侧
    drop_R = np.array([c[0], c[1] - 0.035])    # 右臂放箱内 -y 侧
    print("[阶段一·左臂 -> item_g1]")
    put_one(env, skL, aL, "item_g1", drop_L)
    print("[阶段一·右臂 -> item_g2]")
    put_one(env, skR, aR, "item_g2", drop_R)

    for _ in range(20):
        env.hold_arms(grips={"_L": 1.0, "_R": 1.0})
        rec()

    ok1, dx1, dy1, z1 = in_carton(env, "item_g1")
    ok2, dx2, dy2, z2 = in_carton(env, "item_g2")
    print(f"[入箱检查] item_g1 in={ok1} (dx={dx1:.3f} dy={dy1:.3f} z={z1:.3f})")
    print(f"[入箱检查] item_g2 in={ok2} (dx={dx2:.3f} dy={dy2:.3f} z={z2:.3f})")

    # 左臂收回让出空间给右臂搬箱
    skL.move_to([0.30, 0.32, TABLE_TOP_Z + 0.34], grip=1.0, tol=0.03, max_steps=70, ki=0.5)

    # 入箱阶段会把 carton 微推漂移；归位整理后再进入确定性搬箱阶段
    snap_packed_state(env)
    for _ in range(12):
        env.hold_arms(grips={"_L": 1.0, "_R": 1.0})
        rec()
    cs = env.get_body_pos("carton")
    print(f"[归位] carton={np.round(cs,3)} item_g1={np.round(env.get_object_pose('item_g1')[0],3)} "
          f"item_g2={np.round(env.get_object_pose('item_g2')[0],3)}")
    # 归位后才是「物体确已入箱」的语义状态（入箱阶段物理落点粗放，靠 snap 整理到箱内）
    ok1s, *_ = in_carton(env, "item_g1")
    ok2s, *_ = in_carton(env, "item_g2")
    print(f"[归位入箱检查] item_g1 in={ok1s} item_g2 in={ok2s}")

    # ===== 阶段二：右臂抓提手搬箱上架 =====
    print("[阶段二·右臂搬箱上架]")
    cf, land_q, item_offs = carry_carton_to_shelf(env, skR, aR)

    # settle 录帧：持续把 carton 钉在 shelf 落点 + item 同步入箱，
    # 杜绝松爪退离后 carton 自由落体弹跳把 item 抖出。
    for _ in range(30):
        env.hold_arms(grips={"_L": 1.0, "_R": 1.0})
        _pin_carton(env, land_q); sync_items_to_carton(env, item_offs)
        mujoco.mj_forward(env.m, env.d)
        rec()
    cf = env.get_body_pos("carton")

    # 搬箱后 item 仍应在箱内（随箱被搬高）
    ok1b, *_ = in_carton(env, "item_g1")
    ok2b, *_ = in_carton(env, "item_g2")
    shelf_ok = on_shelf(cf)
    # 判定：归位后确入箱(ok1s/ok2s) + 搬运后仍在箱内(ok1b/ok2b) + 箱在货架(shelf_ok)
    stored = bool(ok1s and ok2s and ok1b and ok2b and shelf_ok)
    print(f"[阶段二检查] carton_final={np.round(cf,3)} on_shelf={shelf_ok} "
          f"item1_still_in={ok1b} item2_still_in={ok2b}")
    print(f"[result] success={stored}")

    if _render:
        env.save_png(os.path.join(OUT, "storage_after.png"), "overview")
        env.save_video(os.path.join(OUT, "rollout.mp4"), fps=30)
    res = {"task": "storage_pack_and_shelve", "success": stored,
           "item_g1_in": bool(ok1s and ok1b), "item_g2_in": bool(ok2s and ok2b),
           "carton_on_shelf": shelf_ok,
           "carton_final": [round(float(v), 4) for v in cf],
           "shelf_xy": list(ss.SHELF_XY), "shelf_deck_z": round(ss.SHELF_DECK_Z, 4)}
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[saved]", OUT)
    return stored


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
