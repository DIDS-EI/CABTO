"""通用单臂原语库（泛化自 stage3 primitives.PrimitiveRunner）
==============================================================
把 stage3 已调通的 pick/place 运动手法（两段式 approach、纯竖直 descend、
闭合跟随 grasp、强积分 converge_xy、收尾 place_release 防弹飞）封装为绑定
到「某一条臂 ArmInterface」的 ArmSkills，使单/双臂任务都能复用同一套手法。

与 PrimitiveRunner 的差异：
- 不调 env.step(7d)，改调 env.step_arms({arm.s: action})，其余臂保持目标不动；
- env.hold_arm/_grasp_enabled/_set_all_welds_inactive 改用 arm/env 上对应方法。
"""

from __future__ import annotations

import numpy as np

from exp4_env import TABLE_TOP_Z, MAX_DPOS

ABOVE_Z = TABLE_TOP_Z + 0.18
HIGH_Z = TABLE_TOP_Z + 0.30


class ArmSkills:
    def __init__(self, env, arm, recorder=None, rec_every=2, block_half=0.025):
        self.env = env
        self.arm = arm
        self.s = arm.s
        self._rec = recorder
        self.rec_every = rec_every
        self._cnt = 0
        self.grab_dz = None
        self.grab_dxy = None
        self.block_half = float(block_half)

    # ---------------- 录帧 ---------------- #
    def _tick(self):
        self._cnt += 1
        if self._rec is not None and self._cnt % self.rec_every == 0:
            self._rec()

    def _step(self, action):
        self.env.step_arms({self.s: np.asarray(action, float)})

    def ee(self):
        return self.arm.get_ee_pose()[0]

    # ============ 底层运动 ============ #
    def move_to(self, target_xyz, grip=1.0, tol=0.01, max_steps=90, ki=0.0,
                i_cap=0.10):
        target = np.asarray(target_xyz, float)
        integ = np.zeros(3)
        for _ in range(max_steps):
            cur = self.ee()
            err = target - cur
            if np.linalg.norm(err) < tol:
                break
            if ki > 0:
                # 限幅积分：消除稳态误差但杜绝 windup 过冲震荡（实测 ki=0.6 无限幅
                # 会在目标附近大幅振荡，加 i_cap 后平滑收敛）。
                integ = np.clip(integ + err, -i_cap, i_cap)
                cmd = err + ki * integ
            else:
                cmd = err
            action = np.zeros(7)
            action[:3] = np.clip(cmd, -MAX_DPOS, MAX_DPOS)
            action[6] = grip
            self._step(action)
            self._tick()
        return float(np.linalg.norm(target - self.ee()))

    def hold(self, grip, n=10):
        for _ in range(n):
            self._step(np.array([0, 0, 0, 0, 0, 0, grip], float))
            self._tick()

    def descend_vertical(self, target_z, grip, max_steps=80, lock_xy=None):
        if lock_xy is None:
            lock_xy = self.ee()[:2].copy()
        lock_xy = np.asarray(lock_xy, float)
        integ = np.zeros(2)
        I_CAP = 0.12
        for _ in range(max_steps):
            cur = self.ee()
            xy_err = float(np.linalg.norm(lock_xy - cur[:2]))
            if abs(cur[2] - target_z) < 0.005 and xy_err < 0.008:
                break
            err_xy = lock_xy - cur[:2]
            integ = np.clip(integ + err_xy, -I_CAP, I_CAP)   # 限幅积分防 windup
            action = np.zeros(7)
            # xy 纠偏增益加强（1.2*P + 限幅I），让边缘 y 位也能把 EE 持续拉回锁定 xy
            action[:2] = np.clip(1.2 * err_xy + 1.0 * integ, -0.03, 0.03)
            # 关键：z 永不完全冻结——xy 越大下降越慢，但始终缓降，避免在边缘 y 位
            # 因 xy_err 跨过门限而彻底卡死、z 永远停在高处。
            if xy_err < 0.025:
                vz = -0.02
            elif xy_err < 0.05:
                vz = -0.012
            else:
                vz = -0.005
            action[2] = np.clip(target_z - cur[2], vz, 0.02)
            action[6] = grip
            self._step(action)
            self._tick()

    def close_following(self, target_xy, n=24):
        target_xy = np.asarray(target_xy, float)
        integ = np.zeros(2)
        for _ in range(n):
            cur = self.ee()
            err_xy = target_xy - cur[:2]
            integ += err_xy
            cmd = err_xy + 0.5 * integ
            action = np.zeros(7)
            action[:2] = np.clip(cmd, -0.02, 0.02)
            action[2] = 0.0
            action[6] = 0.0
            self._step(action)
            self._tick()

    # ============ 对外原语 ============ #
    def stage(self, xy=(0.35, 0.0), z=None, grip=1.0):
        """回到一个固定的「居中高位舞台位」，用强积分消除上一次操作残留的
        关节/EE 偏置，让随后的 approach→align→descend 每次都从同一干净状态开始。
        多对/多物体连续抓取时，在每次抓取前调用，避免从上一次 place/retreat 的
        奇异/边缘位姿斜切，导致 descend 卡门限或蹭倒相邻物体。"""
        z = (ABOVE_Z + 0.08) if z is None else z
        # 先抬高（保持当前 xy），再分两段强积分把 EE 拉到舞台 xy（远距离需多步收敛）
        cur = self.ee()
        self.move_to([cur[0], cur[1], z], grip=grip, tol=0.02, max_steps=60, ki=0.3)
        self.move_to([float(xy[0]), float(xy[1]), z], grip=grip, tol=0.012,
                     max_steps=120, ki=0.6)
        self.move_to([float(xy[0]), float(xy[1]), z], grip=grip, tol=0.006,
                     max_steps=60, ki=0.8)

    def rehome(self, settle=12, grip=1.0):
        """把本臂关节直接复位到 HOME_QPOS（瞬时），再 settle 几步稳定。
        仅在「本臂当前未抓持任何物体」时使用（如每对放置完成、松手之后）：
        这样每次抓取都从与「首对」完全相同的干净关节配置开始，IK 不会被上一次
        的关节分支/积分残留带偏。比纯 move_to 舞台位更彻底可靠。"""
        from exp4_env import HOME_QPOS, FINGER_OPEN, GRIP_OPEN_CTRL
        a = self.arm
        for adr, v in zip(a.arm_qadr, HOME_QPOS):
            self.env.d.qpos[adr] = v
        for adr in a.finger_qadr:
            self.env.d.qpos[adr] = FINGER_OPEN
        a.arm_target = HOME_QPOS.copy()
        a._grip_cmd = float(grip)
        a._grip_ctrl_cur = GRIP_OPEN_CTRL
        a._grasp_enabled = True          # 关键：place_release 关掉了吸附，新一轮抓取前必须重新开启
        import mujoco
        mujoco.mj_forward(self.env.m, self.env.d)
        self.hold(grip=grip, n=settle)

    def approach(self, xyz, above=ABOVE_Z):
        x, y = float(xyz[0]), float(xyz[1])
        self.move_to([x, y, above], grip=1.0, tol=0.012, ki=0.0)

    def align_xy(self, xyz, high_z=ABOVE_Z):
        x, y = float(xyz[0]), float(xyz[1])
        self.move_to([x, y, high_z], grip=1.0, tol=0.005, max_steps=50, ki=0.6)

    def descend_to(self, z, lock_xy, grip=1.0):
        self.hold(grip=grip, n=4)
        # 仅当 xy 尚未对准(>0.02)时才强积分 converge：确保进入 descend_vertical 时
        # xy_err 已 < 其下降门限，否则会卡在高处永不下降（如边缘 y 位）。
        # 若 align_xy 已对准（如右臂某些位形），再调 converge_xy 反而会触发该位形
        # 的 IK 横向漂移把 EE/物体拖偏 ~0.07，故此处跳过。
        lock_xy = np.asarray(lock_xy, float)
        if float(np.linalg.norm(lock_xy - self.ee()[:2])) > 0.02:
            self._converge_xy(lock_xy, grip=grip, max_steps=50)
        self.descend_vertical(z, grip=grip, max_steps=100, lock_xy=lock_xy)

    def grasp(self, xy, n=24):
        self.close_following(np.asarray(xy, float)[:2], n=n)

    def _is_right(self):
        """该臂是否为右臂（base_yaw≈+pi/2）。右臂 IK 对积分敏感、易 windup 发散。"""
        return float(self.arm.base_yaw) > 0.1

    def pick_direct(self, obj_name, grasp_z, lift_top, n_close=32,
                    rehome_first=True, retries=2, mid_z=0.52):
        """双臂稳健抓取（按臂别选不同收敛策略——实测调通）。
        实测（diag_zreach / diag_zreach_R2）：
        - 左臂(yaw=-pi/2)：ki=0.6 分段(高→mid_z→grasp_z)精确达到，lift 用 ki=0.3 分段缓抬 OK；
        - 右臂(yaw=+pi/2)：ki>0 强积分会 windup 发散飞出；用 ki=0「停走式」逐段降稳定，
          残留 ~0.03 xy 偏移由闭合跟随吸收。
        全程限制在安全 z 区(<=mid_z 起步)，再分段缓抬，避开高 z IK 发散区。
        solve_ik 的关节连续性惩罚(JOINT_W=0.2)进一步抑制跳支飞出。
        返回 (holding, grab_dz)。"""
        if rehome_first:
            self.rehome(settle=10)
        right = self._is_right()
        op = self.env.get_object_pose(obj_name)[0]
        if not right:
            # —— 左臂：强积分分段 ——
            self.move_to([op[0], op[1], mid_z], grip=1.0, tol=0.02, max_steps=90, ki=0.4)
            self.move_to([op[0], op[1], mid_z], grip=1.0, tol=0.006, max_steps=80, ki=0.7)
            op = self.env.get_object_pose(obj_name)[0]
            self.move_to([op[0], op[1], grasp_z], grip=1.0, tol=0.006, max_steps=100, ki=0.6)
        else:
            # —— 右臂：ki=0 对准 + 停走式逐段降 + xy 漂移闭环补偿（防 windup 发散）——
            # 实测右臂 ki=0 降 z 时 EE 有持续 +x/+y 漂移：命令 EE→某点，EE 会停在 点+drift。
            # 故用「累积命令补偿」：cmd_xy 初值=罐心，每段后按残余(罐心-EE实测)修正 cmd_xy，
            # 使 EE 真正落到罐心上方（finger-mid 对准罐），而非停在带 drift 的位置。
            self.move_to([op[0], op[1], mid_z], grip=1.0, tol=0.01, max_steps=110, ki=0.0)
            cmd_xy = self.env.get_object_pose(obj_name)[0][:2].copy()
            z = self.ee()[2]
            guard = 0
            while z - grasp_z > 0.006 and guard < 36:
                guard += 1
                op = self.env.get_object_pose(obj_name)[0]
                ee = self.ee()
                cmd_xy = cmd_xy + (op[:2] - ee[:2])   # 累积补偿 EE 相对罐心的漂移
                ztgt = max(grasp_z, z - 0.022)
                self.move_to([cmd_xy[0], cmd_xy[1], ztgt], grip=1.0, tol=0.006,
                             max_steps=45, ki=0.0)
                z = self.ee()[2]
            # 降到位后再做几轮纯 xy 对准（不降 z），把残余偏移压到 grasp 容差内
            for _ in range(5):
                op = self.env.get_object_pose(obj_name)[0]
                ee = self.ee()
                xy_err = op[:2] - ee[:2]
                if float(np.linalg.norm(xy_err)) < 0.012:
                    break
                cmd_xy = cmd_xy + xy_err
                self.move_to([cmd_xy[0], cmd_xy[1], grasp_z], grip=1.0, tol=0.006,
                             max_steps=40, ki=0.0)
        # 闭合跟随
        op = self.env.get_object_pose(obj_name)[0]
        self.grasp(op[:2], n=n_close)
        # 校验 + 重试
        z_try = grasp_z
        ki_retry = 0.0 if right else 0.6
        for _ in range(retries):
            if self.arm.is_holding(obj_name):
                break
            self.hold(grip=1.0, n=4)
            op = self.env.get_object_pose(obj_name)[0]
            z_try -= 0.012
            self.move_to([op[0], op[1], z_try], grip=1.0, tol=0.006, max_steps=55, ki=ki_retry)
            op = self.env.get_object_pose(obj_name)[0]
            self.grasp(op[:2], n=n_close)
        # 分段缓抬（右臂用小 ki=0.15：ki=0 举不起、ki>=0.3 windup 发散，0.15 实测稳定）
        self.lift_safe(top=lift_top, ki=(0.15 if right else 0.3))
        self.record_grab_offset(obj_name)
        return self.arm.is_holding(obj_name), self.grab_dz

    def pick_can_simple(self, obj_name, grasp_z, lift_top, n_close=34,
                        rehome_first=True, retries=2):
        """简洁稳健抓罐（pour 固定杯版专用）。
        实测 diag_reach：左/右臂用 `move_to([can_xy, 0.52])` 再 `move_to([can_xy, grasp_z])`
        (ki=0.6 左 / 0.0 右) 即可把 EE 对到 can xy 误差 ~0.02（< grasp_xy_tol=0.035），
        无需 descend_vertical(其 xy_lock<0.008 门限太严会一路降穿桌面)。
        到位后 close_following 闭合跟随→吸附 weld→lift_safe 缓抬。返回 (holding, grab_dz)。"""
        if rehome_first:
            self.rehome(settle=10)
        right = self._is_right()
        ki = 0.0 if right else 0.6
        op = self.env.get_object_pose(obj_name)[0]
        self.move_to([op[0], op[1], 0.52], grip=1.0, tol=0.006, max_steps=90, ki=ki)
        op = self.env.get_object_pose(obj_name)[0]
        self.move_to([op[0], op[1], grasp_z], grip=1.0, tol=0.006, max_steps=90, ki=ki)
        # 右臂残余漂移：纯 xy 闭环补几轮
        if right:
            for _ in range(5):
                op = self.env.get_object_pose(obj_name)[0]; ee = self.ee()
                xy_err = op[:2] - ee[:2]
                if float(np.linalg.norm(xy_err)) < 0.012:
                    break
                tcp = ee[:2] + xy_err
                self.move_to([tcp[0], tcp[1], grasp_z], grip=1.0, tol=0.006,
                             max_steps=40, ki=0.0)
        op = self.env.get_object_pose(obj_name)[0]
        self.grasp(op[:2], n=n_close)
        z_try = grasp_z
        for _ in range(retries):
            if self.arm.is_holding(obj_name):
                break
            self.hold(grip=1.0, n=4)
            z_try -= 0.012
            op = self.env.get_object_pose(obj_name)[0]
            self.move_to([op[0], op[1], z_try], grip=1.0, tol=0.006, max_steps=55,
                         ki=(0.0 if right else 0.6))
            op = self.env.get_object_pose(obj_name)[0]
            self.grasp(op[:2], n=n_close)
        self.lift_safe(top=lift_top, ki=(0.15 if right else 0.3))
        self.record_grab_offset(obj_name)
        return self.arm.is_holding(obj_name), self.grab_dz

    def lift_safe(self, top, step=0.04, ki=0.3):
        """分段缓慢抬起（每段 <= step），保持 xy，避免一步命令到高 z 触发 IK 发散。
        右臂传 ki=0.15（ki=0 举不起、ki>=0.3 windup 发散；0.15 实测能举到 z=0.62 且不发散）。"""
        cur = self.ee()
        xy = cur[:2].copy()
        z = cur[2]
        while z < top - 0.005:
            z = min(top, z + step)
            self.move_to([xy[0], xy[1], z], grip=0.0, tol=0.01, max_steps=45, ki=ki)

    def pick_secure(self, obj_name, grasp_z, lift_top, n_close=28,
                    rehome_first=True, retries=2):
        """稳健抓取：(可选)rehome → approach → align → 边降边对 xy → 闭合 →
        校验 is_holding；若没抓住，原地再下降一点并重闭合，最多 retries 次。
        返回 (holding, grab_dz)。用于多物体连续抓取的可靠性。"""
        if rehome_first:
            self.rehome(settle=10)
        op = self.env.get_object_pose(obj_name)[0]
        self.approach(op)
        self.align_xy(op)
        op = self.env.get_object_pose(obj_name)[0]
        self.descend_to(grasp_z, lock_xy=op[:2])
        op = self.env.get_object_pose(obj_name)[0]
        self.grasp(op[:2], n=n_close)
        # 校验 + 重试：未咬住则张开、再下降 1.5cm、重闭合
        z_try = grasp_z
        for _ in range(retries):
            if self.arm.is_holding(obj_name):
                break
            self.hold(grip=1.0, n=4)                  # 张开
            op = self.env.get_object_pose(obj_name)[0]
            z_try -= 0.015
            self.descend_vertical(z_try, grip=1.0, max_steps=60, lock_xy=op[:2])
            self.grasp(op[:2], n=n_close)
        self.lift(top=lift_top)
        self.record_grab_offset(obj_name)
        return self.arm.is_holding(obj_name), self.grab_dz

    def lift(self, dz=0.24, xy=None, top=None):
        cur = self.ee()
        if xy is None:
            xy = cur[:2]
        target_z = (ABOVE_Z + 0.06) if top is None else top
        self.move_to([float(xy[0]), float(xy[1]), target_z], grip=0.0,
                     tol=0.02, max_steps=80, ki=0.3)

    def record_grab_offset(self, obj_name):
        ee = self.ee()
        op = self.env.get_object_pose(obj_name)[0]
        self.grab_dz = float(op[2] - ee[2])
        self.grab_dxy = (op[:2] - ee[:2]).copy()
        return self.grab_dz, self.grab_dxy

    def move_above(self, xyz, high_z=HIGH_Z):
        dxy = self.grab_dxy if self.grab_dxy is not None else np.zeros(2)
        tx, ty = float(xyz[0]), float(xyz[1])
        tcp_xy = np.array([tx, ty]) - dxy
        self.move_to([tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.010, max_steps=90, ki=0.0)
        self.move_to([tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.004, max_steps=50, ki=0.6)
        return tcp_xy

    def _converge_xy(self, lock_xy, grip=0.0, max_steps=40):
        lock_xy = np.asarray(lock_xy, float)
        integ = np.zeros(2)
        for _ in range(max_steps):
            cur = self.ee()
            err_xy = lock_xy - cur[:2]
            if float(np.linalg.norm(err_xy)) < 0.004:
                break
            integ += err_xy
            action = np.zeros(7)
            action[:2] = np.clip(err_xy + 0.8 * integ, -0.03, 0.03)
            action[2] = 0.0
            action[6] = grip
            self._step(action)
            self._tick()

    def refine_above(self, dst_xy, high_z=HIGH_Z, obj_name=None):
        if obj_name is None:
            dxy = self.grab_dxy if self.grab_dxy is not None else np.zeros(2)
            return np.array([float(dst_xy[0]), float(dst_xy[1])]) - dxy
        ee = self.ee()
        op = self.env.get_object_pose(obj_name)[0]
        cur_dxy = op[:2] - ee[:2]
        tcp_xy = np.array([float(dst_xy[0]), float(dst_xy[1])]) - cur_dxy
        self.move_to([tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.004, max_steps=40, ki=0.6)
        return tcp_xy

    def descend_place(self, dst_top_z, tcp_xy, gap=0.004, max_steps=220):
        grab_dz = self.grab_dz if self.grab_dz is not None else -0.02
        center_target = dst_top_z + self.block_half + gap
        place_z = center_target - grab_dz
        self._converge_xy(tcp_xy, grip=0.0, max_steps=40)
        self.descend_vertical(place_z, grip=0.0, max_steps=max_steps, lock_xy=tcp_xy)
        return place_z

    def _hold_xy(self, lock_xy, grip, steps=40, gain=1.3, icap=0.15, tol=0.005):
        """原地强积分把 EE 水平钉到 lock_xy（不升降）。"""
        lock_xy = np.asarray(lock_xy, float)
        integ = np.zeros(2)
        for _ in range(steps):
            cur = self.ee()
            err = lock_xy - cur[:2]
            if float(np.linalg.norm(err)) < tol:
                break
            integ = np.clip(integ + err, -icap, icap)
            action = np.zeros(7)
            action[:2] = np.clip(gain * err + 1.0 * integ, -0.03, 0.03)
            action[6] = grip
            self._step(action)
            self._tick()

    def descend_place_tracked(self, dst_xy, dst_top_z, obj_name, gap=0.004,
                              max_steps=260, xy_ok=0.012, z_step=0.025):
        """放置（停走式下降，stop-and-go）：
        1) 用「当前实测抓偏」算出能把物体对到 dst 的 EE 锁定 xy（tcp_lock）；
        2) 高空原地强积分把 EE 收敛到 tcp_lock；
        3) 【停走式下降】每次先把 EE 强收敛回 tcp_lock，再竖直降一小段(~z_step)，循环。
           这样低 z 处 IK 的 +x 漂移不会在一整段长下降里累积——每段只漂一点点就被
           下一轮 converge 拉回，比一次性 descend_vertical 在高塔放置时稳得多
           （实测高塔放置 ee 从漂到 0.62 改善到 落在 tcp_lock±0.01）。
        4) 到底后按「物体实测误差」小幅精对位，确保堆叠/盖盖 xy<0.01。"""
        import os as _os
        _dbg = _os.environ.get("DPT_DEBUG") == "1"
        dst_xy = np.asarray(dst_xy, float)
        grab_dz = self.grab_dz if self.grab_dz is not None else -0.02
        center_target = dst_top_z + self.block_half + gap
        place_z = center_target - grab_dz
        ee = self.ee()
        op = self.env.get_object_pose(obj_name)[0]
        cur_dxy = op[:2] - ee[:2]
        tcp_lock = dst_xy - cur_dxy
        if _dbg:
            print(f"      [dpt] ee={np.round(ee,4)} obj={np.round(op[:2],4)} "
                  f"cur_dxy={np.round(cur_dxy,4)} tcp_lock={np.round(tcp_lock,4)} place_z={place_z:.4f}")
        self._converge_xy(tcp_lock, grip=0.0, max_steps=50)
        # 停走式下降：converge xy → 降一段 → 再 converge → 再降……漂移不累积
        z = self.ee()[2]
        guard = 0
        while z - place_z > 0.004 and guard < 24:
            guard += 1
            self._hold_xy(tcp_lock, grip=0.0, steps=24, tol=0.005)
            ztgt = max(place_z, z - z_step)
            for _ in range(30):
                cur = self.ee()
                action = np.zeros(7)
                action[:2] = np.clip(tcp_lock - cur[:2], -0.02, 0.02)
                action[2] = float(np.clip(ztgt - cur[2], -0.02, 0.0))
                action[6] = 0.0
                self._step(action)
                self._tick()
                if abs(self.ee()[2] - ztgt) < 0.004:
                    break
            z = self.ee()[2]
        # 到位后纯物体闭环小幅精对位（仍闭爪、不升降）
        integ = np.zeros(2)
        for _ in range(40):
            op = self.env.get_object_pose(obj_name)[0]
            obj_err = dst_xy - op[:2]
            if float(np.linalg.norm(obj_err)) < 0.008:
                break
            integ = np.clip(integ + obj_err, -0.05, 0.05)
            action = np.zeros(7)
            action[:2] = np.clip(1.0 * obj_err + 0.8 * integ, -0.008, 0.008)
            action[6] = 0.0
            self._step(action)
            self._tick()
        if _dbg:
            op = self.env.get_object_pose(obj_name)[0]
            print(f"      [dpt done] ee={np.round(self.ee(),4)} obj={np.round(op[:2],4)} "
                  f"obj_e={np.linalg.norm(dst_xy-op[:2]):.4f}")
        return place_z

    def descend_stack(self, dst_xy, dst_top_z, obj_name, gap=0.012,
                      z_step=0.02, max_rounds=30):
        """堆叠专用放置（物体闭环 + 高 z 抗漂移）。
        与 descend_place_tracked 的区别：闭环量取「物体实测 xy 到 dst 的误差」（而非
        EE 到锁点的误差），用强增益把**物体**推回塔心；并把 xy 修正的限幅放宽到 ±0.05，
        足以抵抗高 z 放置时 IK 的 +x 系统漂移（实测高塔放置 EE 自然漂 +0.07~0.09，
        旧 ±0.03 限幅压不住）。停走式：每轮先把物体 xy 收敛到塔心，再竖直降一小段。"""
        import os as _os
        _dbg = _os.environ.get("DPT_DEBUG") == "1"
        dst_xy = np.asarray(dst_xy, float)
        grab_dz = self.grab_dz if self.grab_dz is not None else -0.02
        center_target = dst_top_z + self.block_half + gap
        place_z = center_target - grab_dz

        def converge_block(steps, tol=0.006, gain=1.6, vz=0.0):
            integ = np.zeros(2)
            for _ in range(steps):
                op = self.env.get_object_pose(obj_name)[0]
                err = dst_xy - op[:2]
                if float(np.linalg.norm(err)) < tol and vz == 0.0:
                    break
                integ = np.clip(integ + err, -0.20, 0.20)
                action = np.zeros(7)
                # 闭环目标是「物体」对准塔心；强增益+宽限幅抵抗高 z IK 漂移
                action[:2] = np.clip(gain * err + 1.0 * integ, -0.05, 0.05)
                action[2] = vz
                action[6] = 0.0
                self._step(action)
                self._tick()

        # 1) 先在高空把物体 xy 收敛到塔心正上方
        converge_block(60, tol=0.005)
        if _dbg:
            op = self.env.get_object_pose(obj_name)[0]
            print(f"      [stk] after high converge obj={np.round(op[:2],4)} "
                  f"ee={np.round(self.ee(),4)} place_z={place_z:.4f}", flush=True)
        # 2) 停走式下降：每轮降一小段(z_step)，同时持续把物体 xy 拉回塔心
        for r in range(max_rounds):
            z = self.ee()[2]
            if z - place_z <= 0.004:
                break
            ztgt = max(place_z, z - z_step)
            for _ in range(26):
                op = self.env.get_object_pose(obj_name)[0]
                cur = self.ee()
                err = dst_xy - op[:2]
                action = np.zeros(7)
                action[:2] = np.clip(1.6 * err, -0.05, 0.05)   # 边降边强拉物体回塔心
                action[2] = float(np.clip(ztgt - cur[2], -0.018, 0.0))
                action[6] = 0.0
                self._step(action)
                self._tick()
                if abs(self.ee()[2] - ztgt) < 0.004:
                    break
            # 每段降完，原地把物体 xy 再精收一下（不升降）
            converge_block(20, tol=0.005)
        # 3) 到底后纯物体闭环精对位
        converge_block(40, tol=0.006)
        if _dbg:
            op = self.env.get_object_pose(obj_name)[0]
            print(f"      [stk done] ee={np.round(self.ee(),4)} obj={np.round(op[:2],4)} "
                  f"obj_e={np.linalg.norm(dst_xy-op[:2]):.4f} ee_z={self.ee()[2]:.4f}", flush=True)
        return place_z

    def place_release(self):
        env, arm = self.env, self.arm
        arm._grasp_enabled = False
        for _ in range(12):
            env.hold_arms(grips={self.s: 1.0})
            self._tick()
        arm.release_all()
        for _ in range(24):
            env.hold_arms(grips={self.s: 1.0})
            self._tick()

    def tip_pour(self, axis="x", total_angle=2.0, n=40, grip=0.0, hold=30):
        """倾倒：保持位置，绕水平轴(世界系)把末端从「朝下」逐步翻到目标角度，倒出内容物。
        关键修复（实测）：
        - 用「绝对目标姿态」abs_quat（在初始朝下姿态上累积旋转）而非相对增量——否则 IK
          每步只追一点、下一步又从几乎没转的当前姿态加增量，腕部永远不转（mouth_z 全程 +1）；
        - 开启 pour_local（IK 仅局部求解、不随机重启）+ 高 w_rot，保证关节连续不甩飞。
        axis: 'x' 或 'y'（世界水平轴）。total_angle 建议 ~2.0(115°) 足以倒出。"""
        from scipy.spatial.transform import Rotation as _R
        self.arm.lock_down = False
        self.arm.pour_local = True
        # 该臂「朝下」基准姿态（世界四元数 wxyz）
        base_q = np.asarray(self.arm.grasp_quat, float)
        base_r = _R.from_quat([base_q[1], base_q[2], base_q[3], base_q[0]])
        ax = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
              "nx": np.array([-1.0, 0, 0]), "ny": np.array([0, -1.0, 0])}[axis]
        for k in range(1, n + 1):
            ang = total_angle * k / n
            # 世界系左乘：先做朝下基准，再绕世界 ax 转 ang
            r = _R.from_rotvec(ax * ang) * base_r
            q_xyzw = r.as_quat()
            self.arm.abs_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            action = np.zeros(7); action[6] = grip
            self._step(action); self._tick()
        # 倒完停留让内容物滚出（保持倾倒姿态）
        for _ in range(hold):
            action = np.zeros(7); action[6] = grip
            self._step(action); self._tick()
        self.arm.pour_local = False

    def untip(self, axis="x", total_angle=2.0, n=30, grip=0.0):
        """倒完转回竖直（绝对姿态从当前角回到朝下），并恢复 lock_down。"""
        from scipy.spatial.transform import Rotation as _R
        base_q = np.asarray(self.arm.grasp_quat, float)
        base_r = _R.from_quat([base_q[1], base_q[2], base_q[3], base_q[0]])
        ax = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
              "nx": np.array([-1.0, 0, 0]), "ny": np.array([0, -1.0, 0])}[axis]
        for k in range(n - 1, -1, -1):
            ang = total_angle * k / n
            r = _R.from_rotvec(ax * ang) * base_r
            q_xyzw = r.as_quat()
            self.arm.abs_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            action = np.zeros(7); action[6] = grip
            self._step(action); self._tick()
        self.arm.abs_quat = None        # 解除绝对姿态接管，恢复正常 lock_down
        self.arm.pour_local = False
        self.arm.lock_down = True

    def tip_pour_joint(self, axis="x", total_angle=1.2, n=40, grip=0.0, hold=80,
                       n_solve=64, can_name=None):
        """关节空间倾倒（绕过 IK 跳支与 local-IK 不转的双重困境）。

        关键修正（diag_weld_rot 实测）：weld 刚性传递旋转 OK，且抓取时
        hand_mz≈-1（朝下）而 can_mz≈+1（罐口朝上）—— **两者相反**。
        要把罐倒出（can_mz 变负），必须让 hand 绕水平轴转 ~120°（让 hand 由朝下转向斜上），
        罐随之同样旋转，罐口 z 分量 = cos(θ) → θ=120° 时 can_mz=-0.5（过水平，球可滚出）。
        因此校验对象是「罐」（绕水平轴旋转当前竖直 +z 得到的 z 分量），而非 hand。

        本方法在「目标倾角」用 full-restart IK 解一次（多次取既翻转又关节最近的解），
        然后从当前关节角线性插值到该解，逐段用 step_joint 驱动（不再每步 IK），
        关节空间平滑、无跳支、不爆炸。返回 (罐是否倒过水平, 预测 can_mz)。"""
        from scipy.spatial.transform import Rotation as _R
        env, arm = self.env, self.arm
        cur_pos = self.ee()
        base_q = np.asarray(arm.grasp_quat, float)
        base_r = _R.from_quat([base_q[1], base_q[2], base_q[3], base_q[0]])
        ax = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
              "nx": np.array([-1.0, 0, 0]), "ny": np.array([0, -1.0, 0])}[axis]
        tilt = _R.from_rotvec(ax * total_angle)
        # hand 目标姿态 = 把抓取姿态绕世界水平轴旋转 total_angle
        r = tilt * base_r
        qx = r.as_quat()
        tgt_quat = np.array([qx[3], qx[0], qx[1], qx[2]])
        # 预测罐口朝向：罐当前 +z(向上) 同样被该世界旋转作用
        pred_can_mz = float(tilt.apply([0, 0, 1])[2])
        ik_t = cur_pos + arm.bias_world
        q_goal = arm.solve_ik(ik_t, tgt_quat, restarts=n_solve, w_rot=3.0)
        # 2) 关节空间从当前角线性插值到 q_goal，逐段驱动
        q_start = np.array(arm.arm_target, float)
        arm.lock_down = False
        for k in range(1, n + 1):
            f = k / n
            q_i = (1 - f) * q_start + f * q_goal
            env.step_joint(arm, q_i, grip=grip)
            self._tick()
        # 倒完停留让球滚出
        for _ in range(hold):
            env.step_joint(arm, q_goal, grip=grip)
            self._tick()
        return pred_can_mz < 0.2, pred_can_mz

    def untip_joint(self, n=30, grip=0.0):
        """从当前(倾倒)关节角线性插回「朝下」HOME 附近——这里用插回 untip 前记录的角。
        简化：直接用 lock_down 接管，move_to 抬一点即可恢复朝下。"""
        # 直接重新启用朝下锁定并轻微移动，让 _resolve_target 的 lock_down 把腕拉回朝下
        self.arm.lock_down = True
        cur = self.ee()
        self.move_to([cur[0], cur[1], cur[2] + 0.02], grip=grip, tol=0.02,
                     max_steps=40, ki=0.0)

    def retreat(self, dz=0.10):
        rel_xy = self.ee()[:2].copy()
        ki = 0.15 if self._is_right() else 0.9   # 右臂强积分 windup → 用小 ki
        self.move_to([rel_xy[0], rel_xy[1], ABOVE_Z + dz], grip=1.0,
                     tol=0.02, max_steps=80, ki=ki)

    def settle(self, n=40):
        self.hold(grip=1.0, n=n)
