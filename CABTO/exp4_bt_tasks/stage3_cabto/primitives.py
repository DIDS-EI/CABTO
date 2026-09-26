"""B. 原语层（把 test_blocks.py 已调通的运动手法封装成可复用方法）
====================================================================
PrimitiveRunner(env, recorder) 把 test_blocks 的：
  move_to / descend_vertical / close_following / hold / open_in_place /
  hold_arm 收尾抬手
封装成原语：approach / align_xy / descend_to / grasp / lift /
  move_above / place_release / retreat / settle。

要点（与 test_blocks 完全一致的运动学手法）：
- 两段式 approach：先粗移到正上方，再高处积分精对准 xy。
- descend_to 纯竖直下降（锁 xy，绕过 EE_SERVO_BIAS 横移）。
- grasp 闭合跟随 close_following（tcp 持续拉回 xy + grasp_assist 激活 weld）。
- place_release 收尾：_grasp_enabled=False → hold_arm 张爪(冻结手臂) →
  _set_all_welds_inactive 解 weld → hold_arm 稳定 → 竖直抬手。

原语接收 3D 目标坐标（来自 Pointer，不是真值），高度换算逻辑照搬 test_blocks：
- grasp_z = top - 0.30*BLOCK_FULL（抓物体上部）
- 放置 center_target = dst_top + BLOCK_HALF + 0.003，用抓后实测 grab_dz/grab_dxy 补偿。
"""

from __future__ import annotations

import numpy as np

from blocks_env import TABLE_TOP_Z, BLOCK_HALF, MAX_DPOS

BLOCK_FULL = 2 * BLOCK_HALF
ABOVE_Z = TABLE_TOP_Z + 0.18
HIGH_Z = TABLE_TOP_Z + 0.30


class PrimitiveRunner:
    def __init__(self, env, recorder=None, rec_every=2):
        """recorder: 无参回调，被调用时录一帧（None 则不录）。"""
        self.env = env
        self._rec = recorder
        self.rec_every = rec_every
        self._cnt = 0
        # 抓取后实测的补偿量（grasp 时填）
        self.grab_dz = None
        self.grab_dxy = None

    # ---------------- 录帧 ---------------- #
    def _tick(self):
        self._cnt += 1
        if self._rec is not None and self._cnt % self.rec_every == 0:
            self._rec()

    # ============ 底层运动（照搬 test_blocks） ============ #
    def move_to(self, target_xyz, grip=1.0, tol=0.01, max_steps=90, ki=0.0):
        env = self.env
        target = np.asarray(target_xyz, float)
        integ = np.zeros(3)
        for _ in range(max_steps):
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
            self._tick()
        return float(np.linalg.norm(target - env.get_ee_pose()[0]))

    def hold(self, grip, n=10):
        env = self.env
        for _ in range(n):
            env.step(np.array([0, 0, 0, 0, 0, 0, grip], float))
            self._tick()

    def descend_vertical(self, target_z, grip, max_steps=80, lock_xy=None):
        env = self.env
        if lock_xy is None:
            lock_xy = env.get_ee_pose()[0][:2].copy()
        lock_xy = np.asarray(lock_xy, float)
        integ = np.zeros(2)
        for _ in range(max_steps):
            cur = env.get_ee_pose()[0]
            xy_err = float(np.linalg.norm(lock_xy - cur[:2]))
            if abs(cur[2] - target_z) < 0.005 and xy_err < 0.006:
                break
            err_xy = lock_xy - cur[:2]
            integ += err_xy
            action = np.zeros(7)
            # xy 强积分(0.8)压住 EE_SERVO_BIAS 前馈横推；z 仅在 xy 已收敛时才下降，
            # 否则先停下来收 xy（防止大行程下降途中横扫撞翻旁物）。
            action[:2] = np.clip(err_xy + 0.8 * integ, -0.03, 0.03)
            if xy_err < 0.02:
                action[2] = np.clip(target_z - cur[2], -0.02, 0.02)
            else:
                action[2] = 0.0          # xy 未收敛先不降，避免斜向横扫
            action[6] = grip
            env.step(action)
            self._tick()

    def close_following(self, target_xy, n=24):
        env = self.env
        target_xy = np.asarray(target_xy, float)
        integ = np.zeros(2)
        for _ in range(n):
            cur = env.get_ee_pose()[0]
            err_xy = target_xy - cur[:2]
            integ += err_xy
            cmd = err_xy + 0.5 * integ
            action = np.zeros(7)
            action[:2] = np.clip(cmd, -0.02, 0.02)
            action[2] = 0.0
            action[6] = 0.0
            env.step(action)
            self._tick()

    # ============ 对外原语 ============ #
    def approach(self, xyz):
        """粗移到目标正上方 ABOVE_Z 高处（张爪）。"""
        x, y = float(xyz[0]), float(xyz[1])
        self.move_to([x, y, ABOVE_Z], grip=1.0, tol=0.012, ki=0.0)

    def align_xy(self, xyz, high_z=ABOVE_Z):
        """高处积分精对准 xy（消稳态偏差，避免斜向下探碰倒块）。"""
        x, y = float(xyz[0]), float(xyz[1])
        self.move_to([x, y, high_z], grip=1.0, tol=0.005, max_steps=50, ki=0.6)

    def descend_to(self, z, lock_xy, grip=1.0):
        """纯竖直下降到 z（锁 xy）。"""
        self.hold(grip=grip, n=4)
        self.descend_vertical(z, grip=grip, max_steps=80, lock_xy=lock_xy)

    def grasp(self, xy):
        """闭合跟随抓取（grasp_assist 激活 weld），并记录抓后补偿量 grab_dz/grab_dxy。"""
        self.close_following(np.asarray(xy, float)[:2], n=24)

    def lift(self, dz=0.24, xy=None):
        """抬起 dz。抓取后调用；抬起后记录 grab_dz/grab_dxy。"""
        env = self.env
        cur = env.get_ee_pose()[0]
        if xy is None:
            xy = cur[:2]
        target_z = ABOVE_Z + 0.06
        self.move_to([float(xy[0]), float(xy[1]), target_z], grip=0.0,
                     tol=0.02, max_steps=70, ki=0.3)

    def record_grab_offset(self, obj_name):
        """抓稳后实测物体中心相对 tcp 的偏移（weld 锁定后恒定），供放置精确换算。"""
        env = self.env
        ee = env.get_ee_pose()[0]
        op = env.get_object_pose(obj_name)[0]
        self.grab_dz = float(op[2] - ee[2])
        self.grab_dxy = (op[:2] - ee[:2]).copy()
        return self.grab_dz, self.grab_dxy

    def move_above(self, xyz, high_z=HIGH_Z):
        """高空把「被抓物体」对准目标 xyz 正上方（补偿抓偏量 grab_dxy）。
        xyz 为放置目标顶面 3D 点。返回最终 tcp_xy（供 place_release 锁 xy）。"""
        env = self.env
        dxy = self.grab_dxy if self.grab_dxy is not None else np.zeros(2)
        tx, ty = float(xyz[0]), float(xyz[1])
        tcp_xy = np.array([tx, ty]) - dxy
        self.move_to([tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.010,
                     max_steps=90, ki=0.0)
        self.move_to([tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.004,
                     max_steps=50, ki=0.6)
        return tcp_xy

    def descend_place(self, dst_top_z, tcp_xy, obj_name=None):
        """纯竖直下放：让被抓物体底面落到 dst 顶面上方 ~3mm。
        center_target = dst_top + BLOCK_HALF + 0.003；place_z = center_target - grab_dz。
        若提供 obj_name 则先按当前实际抓偏精修 tcp_xy 使物体正对目标中心。"""
        env = self.env
        grab_dz = self.grab_dz if self.grab_dz is not None else -0.02
        center_target = dst_top_z + BLOCK_HALF + 0.003
        place_z = center_target - grab_dz
        # 先在高空把 xy 牢牢收敛到 lock_xy（强积分，不降 z），再竖直下放。
        self._converge_xy(tcp_xy, grip=0.0, max_steps=40)
        self.descend_vertical(place_z, grip=0.0, max_steps=220, lock_xy=tcp_xy)
        return place_z

    def _converge_xy(self, lock_xy, grip=0.0, max_steps=40):
        """原地(z不动)用强积分把 tcp 的 xy 收敛到 lock_xy。"""
        env = self.env
        lock_xy = np.asarray(lock_xy, float)
        integ = np.zeros(2)
        for _ in range(max_steps):
            cur = env.get_ee_pose()[0]
            err_xy = lock_xy - cur[:2]
            if float(np.linalg.norm(err_xy)) < 0.004:
                break
            integ += err_xy
            action = np.zeros(7)
            action[:2] = np.clip(err_xy + 0.8 * integ, -0.03, 0.03)
            action[2] = 0.0
            action[6] = grip
            env.step(action)
            self._tick()

    def refine_above(self, dst_xy, high_z=HIGH_Z, obj_name=None):
        """在高空按当前实际抓偏再精修一次 tcp_xy（需 obj_name 真值；oracle/调试用）。
        VLM 路无真值时跳过（用 move_above 的 grab_dxy 补偿即可）。返回 tcp_xy。"""
        env = self.env
        if obj_name is None:
            dxy = self.grab_dxy if self.grab_dxy is not None else np.zeros(2)
            return np.array([float(dst_xy[0]), float(dst_xy[1])]) - dxy
        ee = env.get_ee_pose()[0]
        op = env.get_object_pose(obj_name)[0]
        cur_dxy = op[:2] - ee[:2]
        tcp_xy = np.array([float(dst_xy[0]), float(dst_xy[1])]) - cur_dxy
        self.move_to([tcp_xy[0], tcp_xy[1], high_z], grip=0.0, tol=0.004,
                     max_steps=40, ki=0.6)
        return tcp_xy

    def place_release(self, place_z=None, lock_xy=None):
        """放置收尾（照搬 test_blocks 关键顺序，避免松手弹飞）：
        a) _grasp_enabled=False b) hold_arm 张爪(冻结手臂,weld保持)
        c) _set_all_welds_inactive 解 weld d) hold_arm 原地稳定 e) 竖直抬手。"""
        env = self.env
        env._grasp_enabled = False
        for _ in range(12):
            env.hold_arm(grip=1.0)
            self._tick()
        env._set_all_welds_inactive()
        for _ in range(24):
            env.hold_arm(grip=1.0)
            self._tick()

    def retreat(self, dz=0.10):
        """竖直抬手离开（强 xy 积分 ki=0.9 压住 EE_SERVO_BIAS 横移）。"""
        env = self.env
        rel_xy = env.get_ee_pose()[0][:2].copy()
        self.move_to([rel_xy[0], rel_xy[1], ABOVE_Z + dz], grip=1.0,
                     tol=0.02, max_steps=80, ki=0.9)

    def settle(self, n=40):
        """张爪原地保持让物体落稳。"""
        self.hold(grip=1.0, n=n)


def grasp_z_for_top(top_z):
    """由物体顶面 z 算抓取高度（抓上部，照搬 test_blocks）。"""
    return top_z - 0.30 * BLOCK_FULL
