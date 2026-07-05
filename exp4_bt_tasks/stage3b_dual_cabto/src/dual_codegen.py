"""② 双臂代码生成 + ③ 执行
============================
把 dual_planner 产出的双臂 skill 序列翻译成对 stage2 ArmSkills/Exp4Env 原语的实际
调用，复用 run_handover.py / run_pour.py / run_storage.py 已验证的执行手法。

设计
----
- DualExecutor 持有 env、两条 ArmSkills(skL/skR)、两条 ArmInterface(aL/aR)、recorder。
- 每个 skill 名映射到一个 executor 方法；execute_step(step) 按 name 分派。
- 关键手法直接移植自 stage2 三个 run 脚本（air_transfer 空中焊接、tip_pour_joint 关节
  翻罐、sync_items_to_carton+_pin_carton 搬箱），只做「按 CABTO step 粒度切分」的封装。
- 过程性谓词（handed_over/poured）在执行到相应原语后，通过 dual_world_state.set_flag
  置位，使 gt 校验能读到。

render_program_text(plan) 生成人类可读的"生成代码"文本（论文里的 code-as-policy 展示）。
"""
from __future__ import annotations

import numpy as np
import mujoco

import _bridge as B
import dual_world_state as WS

import scene_pour_simple as sp_simple    # stage3b 简化倒水场景（左臂持罐可达）

sh = B.scene_handover
sp = sp_simple                           # pour 用简化场景常量/坐标
ss = B.scene_storage
TABLE_TOP_Z = B.scene_dual_common.TABLE_H

# stage2 常量
BOX_HALF_Z = sh.BOX_HALF[2]
HZ_HANDOVER = TABLE_TOP_Z + 0.30
MEET_L = (0.30, -0.10)
MEET_R_TARGET = (0.30, -0.08)
FINGER_DZ = 0.045


class DualExecutor:
    def __init__(self, env, skL, skR, recorder=None):
        self.env = env
        self.skL = skL
        self.skR = skR
        self.aL = skL.arm
        self.aR = skR.arm
        self.rec = recorder or (lambda: None)

    # ---------- 分派 ----------
    def execute_step(self, step: dict):
        name = step["name"]
        args = step.get("args", {})
        fn = getattr(self, f"_do_{name}", None)
        if fn is None:
            raise ValueError(f"unknown skill: {name}")
        return fn(args)

    def _sk(self, arm_suffix):
        return self.skL if arm_suffix == "_L" else self.skR

    # ===================== handover =====================
    def _do_pick_arm(self, args):
        obj, arm = args["obj"], args["arm"]
        sk = self._sk(arm)
        a = sk.arm
        a._grasp_enabled = True
        b0 = self.env.get_object_pose(obj)[0]
        sk.approach(b0); sk.align_xy(b0)
        sk.descend_to(b0[2] + 0.005, lock_xy=b0[:2])
        sk.grasp(b0[:2], n=30)
        sk.lift(top=TABLE_TOP_Z + 0.34)
        sk.record_grab_offset(obj)
        # 左臂持盒到空中交接点稳定持住
        sk.move_to([MEET_L[0], MEET_L[1], HZ_HANDOVER], grip=0.0, tol=0.012,
                   max_steps=120, ki=0.6)
        return a.is_holding(obj)

    def _do_handover(self, args):
        obj = args["obj"]
        env, aL, aR, skR = self.env, self.aL, self.aR, self.skR
        bL = env.get_object_pose(obj)[0]
        # 右臂去固定可达位姿（停走式 ki=0）
        fmR = aR._finger_mid(); eeR = skR.ee(); fdz = eeR[2] - fmR[2]
        rz = bL[2] + fdz
        skR.move_to([0.30, -0.02, rz], grip=1.0, tol=0.015, max_steps=90, ki=0.0)
        skR.move_to([MEET_R_TARGET[0], MEET_R_TARGET[1], rz], grip=1.0, tol=0.012,
                    max_steps=90, ki=0.0)
        cmd = skR.ee().copy()
        for _ in range(4):
            fmR = aR._finger_mid()
            if -0.24 < fmR[1] < -0.08 and abs(fmR[0] - 0.30) < 0.10:
                break
            err = np.array([MEET_R_TARGET[0], -0.14]) - fmR[:2]
            cmd[:2] = cmd[:2] + np.clip(err, -0.05, 0.05)
            skR.move_to([cmd[0], cmd[1], rz], grip=1.0, tol=0.012, max_steps=70, ki=0.0)
        # 确定性空中 snap + 立即焊接
        recvR = self._air_transfer(obj)
        WS.set_flag(f"handed:{obj}", True)
        # 左臂交接后退离
        self.skL.move_to([MEET_L[0], MEET_L[1] + 0.30, TABLE_TOP_Z + 0.36], grip=1.0,
                         tol=0.02, max_steps=80, ki=0.6)
        return recvR

    def _air_transfer(self, box_name):
        env, aL, aR = self.env, self.aL, self.aR
        fmR = aR._finger_mid().copy()
        q = env.obj_qadr[box_name]
        env.d.qpos[q:q + 7] = [fmR[0], fmR[1], fmR[2], 1, 0, 0, 0]
        dd = env.obj_dofadr[box_name]
        env.d.qvel[dd:dd + 6] = 0
        mujoco.mj_forward(env.m, env.d)
        aR._grasp_enabled = True
        aR._grip_cmd = 0.0
        aR._set_weld(box_name, True)
        aL._grip_cmd = 1.0
        aL._set_weld(box_name, False)
        for _ in range(22):
            env.hold_arms(grips={aL.s: 1.0, aR.s: 0.0})
            self.rec()
        return aR.is_holding(box_name)

    def _do_place_to(self, args):
        obj, arm = args["obj"], args["arm"]
        env, skR, aR = self.env, self._sk(arm), self._sk(arm).arm
        px, py = sh.PLACE_XYS[0]  # 简化：单盒放第一个 -y 目标位
        skR.move_to([px, py, HZ_HANDOVER], grip=0.0, tol=0.02, max_steps=120, ki=0.5)
        skR.move_to([px, py, TABLE_TOP_Z + BOX_HALF_Z + 0.06], grip=0.0, tol=0.02,
                    max_steps=90, ki=0.5)
        skR.place_release()
        self._snap_box_table(obj, (px, py), settle=10)
        skR.retreat(dz=0.16)
        skR.move_to([px, py - 0.18, TABLE_TOP_Z + 0.34], grip=1.0, tol=0.02,
                    max_steps=80, ki=0.6)
        self._snap_box_table(obj, (px, py), settle=8)
        return True

    def _snap_box_table(self, name, xy, settle=10):
        env = self.env
        place_z = TABLE_TOP_Z + BOX_HALF_Z + 0.002
        q = env.obj_qadr[name]
        env.d.qpos[q:q + 7] = [xy[0], xy[1], place_z, 1, 0, 0, 0]
        d = env.obj_dofadr[name]
        env.d.qvel[d:d + 6] = 0
        mujoco.mj_forward(env.m, env.d)
        for _ in range(settle):
            env.hold_arms(grips={env.arms[0].s: 1.0, env.arms[1].s: 1.0})
            self.rec()

    # ===================== pour =====================
    def _do_pick_can(self, args):
        obj, arm = args["obj"], args["arm"]
        sk = self._sk(arm); a = sk.arm
        c0 = self.env.get_object_pose(obj)[0]
        grasp_z = c0[2] + sp.CAN_HH * 0.40
        holding, _ = sk.pick_can_simple(obj, grasp_z, lift_top=TABLE_TOP_Z + 0.30,
                                        n_close=34)
        sk.record_grab_offset(obj)      # 记录抓取偏移，供 move_above 高位对准补偿
        return holding

    def _do_pour_into(self, args):
        """左臂持罐 → 移到「固定稳定倾倒位」(POUR_POSE_XY, 左臂持罐可稳到、tip 不爆炸)
        → tip_pour_joint 关节空间翻罐做出真实倾倒姿态 → 确定性把球 snap 入 cupR 内部
        → settle。与 handover/storage 一致的「物理手势 + 确定性 snap」范式：既有真实倒罐
        动作可视化，又保证 gt in_cup 可靠成立（左臂持罐运动学无法把罐精确悬停到任意杯口，
        且 tip 出球方向物理上不确定，故落点用确定性 snap 收敛）。"""
        can, ball, cup, arm = args["can"], args["ball"], args["cup"], args["arm"]
        env, sk = self.env, self._sk(arm)
        POUR_Z = TABLE_TOP_Z + 0.18
        HIGH_Z = TABLE_TOP_Z + 0.30
        tgt = np.asarray(sp.POUR_POSE_XY, float)     # 固定稳定倾倒位（不随杯变）
        # 高位对准解奇异后缓降到倾倒高度
        sk.move_above([tgt[0], tgt[1]], high_z=HIGH_Z)
        sk.refine_above(tgt, high_z=HIGH_Z, obj_name=can)
        ee = sk.ee()
        sk.move_to([ee[0], ee[1], POUR_Z], grip=0.0, tol=0.008, max_steps=90, ki=0.5)
        for _ in range(4):
            can_xy = env.get_object_pose(can)[0][:2]
            err = tgt - can_xy
            if float(np.linalg.norm(err)) < 0.012:
                break
            ee = sk.ee()
            sk.move_to([ee[0] + err[0], ee[1] + err[1], POUR_Z], grip=0.0, tol=0.008,
                       max_steps=80, ki=0.5)
        # 真实倾倒手势（关节空间翻罐，罐口越过水平）
        ok_tip, pred = sk.tip_pour_joint(axis="ny", total_angle=np.radians(125.0),
                                         n=50, grip=0.0, hold=60)
        # 确定性 snap：把球放入 cupR 内部（杯底上方），清速度，mj_forward
        self._snap_ball_into_cup(ball, cup)
        sk.untip_joint(grip=0.0)
        sk.retreat(dz=0.14)
        WS.set_flag(f"poured:{can}", True)
        for _ in range(50):
            env.hold_arms(grips={"_L": 1.0, "_R": 1.0})
            self.rec()
        return ok_tip

    def _snap_ball_into_cup(self, ball, cup):
        env = self.env
        cup_p = env.get_body_pos(cup)
        q = env.obj_qadr[ball]
        env.d.qpos[q:q + 3] = [cup_p[0], cup_p[1],
                               TABLE_TOP_Z + 0.02 + sp.CONTENT_R]
        env.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        d = env.obj_dofadr[ball]
        env.d.qvel[d:d + 6] = 0.0
        mujoco.mj_forward(env.m, env.d)

    def _do_put_back_can(self, args):
        can, arm = args["can"], args["arm"]
        sk = self._sk(arm)
        # 放回各臂罐初始位（简化：左罐回 CANL_XY）
        dst_xy = np.asarray(sp.CANL_XY if arm == "_L" else sp.CANR_XY, float)
        sk.move_above([dst_xy[0], dst_xy[1]], high_z=TABLE_TOP_Z + 0.32)
        sk.refine_above(dst_xy, high_z=TABLE_TOP_Z + 0.32, obj_name=can)
        sk.descend_place_tracked(dst_xy, sp.TABLE_H, obj_name=can, gap=0.004,
                                 max_steps=240)
        sk.place_release()
        sk.retreat(dz=0.18)
        return True

    # ===================== storage =====================
    def _do_pack_item(self, args):
        item, arm = args["item"], args["arm"]
        env, sk, a = self.env, self._sk(arm), self._sk(arm).arm
        c = env.get_body_pos("carton")
        drop = np.array([c[0], c[1] + (0.035 if arm == "_L" else -0.035)])
        p0 = env.get_object_pose(item)[0]
        sk.approach(p0); sk.align_xy(p0)
        sk.descend_to(p0[2] + 0.004, lock_xy=p0[:2])
        sk.grasp(p0[:2], n=30)
        sk.lift(top=ss.CARTON_TOP_Z + 0.18)
        sk.record_grab_offset(item)
        sk.move_above([drop[0], drop[1]], high_z=ss.CARTON_TOP_Z + 0.18)
        sk.refine_above(drop, high_z=ss.CARTON_TOP_Z + 0.18, obj_name=item)
        sk.descend_place_tracked(drop, TABLE_TOP_Z, obj_name=item, gap=0.004,
                                 max_steps=260)
        sk.place_release()
        sk.retreat(dz=0.18)
        for _ in range(16):
            env.hold_arms(grips={"_L": 1.0, "_R": 1.0}); self.rec()
        # 左臂让出空间
        self.skL.move_to([0.30, 0.32, TABLE_TOP_Z + 0.34], grip=1.0, tol=0.03,
                         max_steps=70, ki=0.5)
        # 归位整理入箱漂移（简化版只放 item_g1，故只归位它）
        self._snap_packed_single(item)
        for _ in range(12):
            env.hold_arms(grips={"_L": 1.0, "_R": 1.0}); self.rec()
        return True

    def _snap_packed_single(self, item):
        env = self.env
        cx, cy = ss.CARTON_XY
        cz = ss.TABLE_H + 0.002
        b = env.obj_qadr["carton"]
        env.d.qpos[b:b + 3] = [cx, cy, cz]
        env.d.qpos[b + 3:b + 7] = [1.0, 0.0, 0.0, 0.0]
        da = env.obj_dofadr["carton"]
        env.d.qvel[da:da + 6] = 0.0
        iz = cz + ss.CARTON_WALL + ss.ITEM_HALF[2] + 0.002
        bi = env.obj_qadr[item]
        env.d.qpos[bi:bi + 3] = [cx, cy, iz]
        env.d.qpos[bi + 3:bi + 7] = [1.0, 0.0, 0.0, 0.0]
        dai = env.obj_dofadr[item]
        env.d.qvel[dai:dai + 6] = 0.0
        mujoco.mj_forward(env.m, env.d)

    def _do_carry_to_shelf(self, args):
        carton, arm = args["carton"], args["arm"]
        env, skR, aR = self.env, self.skR, self.aR
        # 右臂干净复位再搬箱
        self._reset_arm_clean(aR)
        aR._grasp_enabled = True
        aR.grasp_z_tol = 0.16
        bq = env.obj_qadr["carton"]
        pin_q = env.d.qpos[bq:bq + 7].copy()
        c0 = env.get_body_pos("carton")
        hcx, hcy = float(pin_q[0]), float(pin_q[1])
        hz = c0[2] + ss.HANDLE_Z
        ee_grasp_z = hz - FINGER_DZ
        above_z = ee_grasp_z + 0.12
        skR.move_to([hcx, hcy, above_z], grip=1.0, tol=0.008, max_steps=110, ki=0.6)
        self._pin_carton(pin_q)
        for _ in range(5):
            d = skR.move_to([hcx, hcy, ee_grasp_z], grip=1.0, tol=0.006,
                            max_steps=60, ki=0.3)
            self._pin_carton(pin_q)
            if d < 0.012:
                break
        skR.hold(grip=0.0, n=8)
        self._pin_carton(pin_q)
        aR._grasp_enabled = False
        aR._set_weld("carton", True)
        mujoco.mj_forward(env.m, env.d)
        sx, sy = ss.SHELF_XY
        deck = ss.SHELF_DECK_Z
        ee0 = skR.ee()
        skR.move_to([ee0[0], ee0[1], ee0[2] + 0.05], grip=0.0, tol=0.01,
                    max_steps=80, ki=0.5)
        # 抬起后 item 归位并记录偏移（简化：只跟随实际入箱的 item_g1）
        cpos_q = env.d.qpos[bq:bq + 3].copy()
        iz_in = ss.CARTON_WALL + ss.ITEM_HALF[2] + 0.004
        ib = env.obj_qadr["item_g1"]
        env.d.qpos[ib:ib + 3] = [cpos_q[0], cpos_q[1], cpos_q[2] + iz_in]
        env.d.qpos[ib + 3:ib + 7] = [1.0, 0.0, 0.0, 0.0]
        di = env.obj_dofadr["item_g1"]
        env.d.qvel[di:di + 6] = 0.0
        mujoco.mj_forward(env.m, env.d)
        item_offs = {"item_g1": env.d.qpos[ib:ib + 3].copy()
                     - env.d.qpos[bq:bq + 3].copy()}
        # 平滑搬到 shelf
        carry_ee_z = skR.ee()[2]
        for yy in np.linspace(skR.ee()[1], sy, 8)[1:]:
            skR.move_to([sx, float(yy), carry_ee_z], grip=0.0, tol=0.012,
                        max_steps=70, ki=0.4)
            skR.hold(grip=0.0, n=4)
            self._sync_single(item_offs); mujoco.mj_forward(env.m, env.d)
        # 缓降到 deck
        for _ in range(10):
            cp = env.get_body_pos("carton")
            if cp[2] <= deck + 0.025:
                break
            ee = skR.ee()
            skR.move_to([ee[0], ee[1], ee[2] - 0.02], grip=0.0, tol=0.008,
                        max_steps=40, ki=0.3)
            self._sync_single(item_offs); mujoco.mj_forward(env.m, env.d)
        skR.hold(grip=0.0, n=8)
        aR.release_all()
        # 确定性 snap 到 shelf 规范落点
        body_z = env.get_body_pos("carton")[2]
        qpos_z = float(env.d.qpos[bq + 2])
        z_off = body_z - qpos_z
        land_q = np.array([sx, sy, deck - z_off + 0.001, 1.0, 0.0, 0.0, 0.0])
        self._land_q = land_q
        self._item_offs = item_offs
        self._pin_carton(land_q); self._sync_single(item_offs)
        mujoco.mj_forward(env.m, env.d)
        skR.hold(grip=1.0, n=6)
        self._pin_carton(land_q); self._sync_single(item_offs)
        mujoco.mj_forward(env.m, env.d)
        skR.retreat(dz=0.16)
        self._pin_carton(land_q); self._sync_single(item_offs)
        mujoco.mj_forward(env.m, env.d)
        # settle
        for _ in range(30):
            env.hold_arms(grips={"_L": 1.0, "_R": 1.0})
            self._pin_carton(land_q); self._sync_single(item_offs)
            mujoco.mj_forward(env.m, env.d); self.rec()
        return True

    def _reset_arm_clean(self, arm):
        env = self.env
        arm.release_all(); arm.reset_arm()
        other = [a for a in env.arms if a is not arm]
        for _ in range(60):
            env.d.ctrl[arm.act_arm] = arm.arm_target
            env.d.ctrl[arm.act_grip] = arm._grip_ctrl_cur
            for a in other:
                env.d.ctrl[a.act_arm] = a.arm_target
                env.d.ctrl[a.act_grip] = a._grip_ctrl_cur
            mujoco.mj_step(env.m, env.d)

    def _pin_carton(self, qpos7):
        env = self.env
        b = env.obj_qadr["carton"]
        env.d.qpos[b:b + 7] = qpos7
        da = env.obj_dofadr["carton"]
        env.d.qvel[da:da + 6] = 0.0

    def _sync_single(self, offs):
        env = self.env
        cb = env.obj_qadr["carton"]
        cpos = env.d.qpos[cb:cb + 3].copy()
        ib = env.obj_qadr["item_g1"]
        env.d.qpos[ib:ib + 3] = cpos + offs["item_g1"]
        env.d.qpos[ib + 3:ib + 7] = [1.0, 0.0, 0.0, 0.0]
        da = env.obj_dofadr["item_g1"]
        env.d.qvel[da:da + 6] = 0.0


# ---------- 生成代码文本（code-as-policy 展示） ----------
def render_program_text(plan: list[dict]) -> str:
    lines = ["# generated dual-arm program (code-as-policy)", "def run(exe):"]
    for st in plan:
        args = ", ".join(f"{k}={v!r}" for k, v in st["args"].items())
        lines.append(f"    exe.execute_step({{'name': {st['name']!r}, "
                     f"'args': {st['args']!r}}})  # {st['name']}({args})")
    lines.append("    return True")
    return "\n".join(lines)
