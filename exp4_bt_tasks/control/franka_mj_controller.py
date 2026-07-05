"""
MuJoCo Franka 移动臂控制库（与 Genesis 版 franka_controller 对应）
====================================================================
提供两个通用接口：
  1) drive_to((x, y, yaw))   —— 把整机底座(freejoint 基座焊接到 base_mocap)
                                运动学移动到任意 (x, y) 并转到任意朝向 yaw。
  2) move_ee_pose(pos, rpy)  —— 给定末端 6 维位姿 (x,y,z + roll,pitch,yaw)，
                                用阻尼最小二乘(DLS)雅可比 IK 求解 7 轴角度，
                                再让位置伺服把指尖中心移动到该位姿。

控制约定：
  - 底座：每步写 base_mocap 的 mocap_pos/quat，靠 weld 约束让整机跟随（运动学移动）；
  - 手臂：position 伺服(actuator1..7) 力控保持/跟随目标关节角；
  - 夹爪：actuator8 (0..255, 255≈张开 0.04m)；
  - IK 在一份 scratch MjData 上用 mj_kinematics + mj_jacSite 迭代，零扰动求解。
"""

import math

import numpy as np
import mujoco


ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"actuator{i}" for i in range(1, 8)]
GRIP_ACT = "actuator8"

# Panda hand 坐标系：接近轴 +z，两指沿 y。正面水平抓取 = hand +z 转到世界 +x。
GRASP_QUAT_FWD = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
GRASP_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

GRIP_OPEN = 255.0       # actuator8 ctrl: 255 ≈ 张开 0.04m
GRIP_CLOSE = 0.0


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def euler_to_quat(roll, pitch, yaw):
    """欧拉角 (roll绕x, pitch绕y, yaw绕z) -> 四元数 (w,x,y,z)，R=Rz·Ry·Rx。"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


class MjFrankaController:
    def __init__(self, model, data, base_start=(0.0, 0.0, 0.0), base_z=0.30,
                 renderer=None, camera="overview", fps=60):
        self.m, self.d = model, data
        self.renderer, self.camera = renderer, camera
        self.frames = []
        self._frame_i = 0
        self.render_every = max(1, round(1.0 / (fps * model.opt.timestep)))

        self.base_z = base_z
        bs = tuple(base_start)
        self.base_x, self.base_y = bs[0], bs[1]
        self.base_yaw = bs[2] if len(bs) > 2 else 0.0

        # 索引
        self.tcp_sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        bmid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "base_mocap")
        self.base_mocap = model.body_mocapid[bmid]
        self.green_mocap = [model.body_mocapid[_id(model, mujoco.mjtObj.mjOBJ_BODY, f"green{i}")]
                            for i in range(5)]
        self.red_mocap = model.body_mocapid[_id(model, mujoco.mjtObj.mjOBJ_BODY, "red_target")]

        self.arm_qadr = [model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in ARM_JOINTS]
        self.arm_dofadr = [model.jnt_dofadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                           for n in ARM_JOINTS]
        self.finger_qadr = [model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                            for n in ("finger_joint1", "finger_joint2")]
        self.act_arm = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTS]
        self.act_grip = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACT)
        self.base_free_qadr = model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]

        self.arm_target = HOME_QPOS.copy()
        self.grip_cmd = GRIP_OPEN
        self.ik_data = mujoco.MjData(model)   # IK 求解用 scratch

    # ----------------------------------------------------------------- 底座/标记
    def _set_base_mocap(self):
        self.d.mocap_pos[self.base_mocap] = (self.base_x, self.base_y, self.base_z)
        self.d.mocap_quat[self.base_mocap] = yaw_quat(self.base_yaw)

    def set_green(self, i, x, y, yaw):
        self.d.mocap_pos[self.green_mocap[i]] = (x, y, 0.006)
        self.d.mocap_quat[self.green_mocap[i]] = yaw_quat(yaw)

    def set_red(self, pos):
        self.d.mocap_pos[self.red_mocap] = pos

    # ----------------------------------------------------------------- 基础循环
    def tcp_pos(self):
        return self.d.site_xpos[self.tcp_sid].copy()

    def _apply_state(self):
        """运动学驱动：直接把底座/手臂/夹爪写入 qpos（精确无伺服滞后）。"""
        q = self.d.qpos
        b = self.base_free_qadr
        q[b + 0], q[b + 1], q[b + 2] = self.base_x, self.base_y, self.base_z
        q[b + 3:b + 7] = yaw_quat(self.base_yaw)
        for adr, v in zip(self.arm_qadr, self.arm_target):
            q[adr] = v
        fg = float(self.grip_cmd) / 255.0 * 0.04
        for adr in self.finger_qadr:
            q[adr] = fg

    def _maybe_render(self):
        if self.renderer is None:
            return
        if self._frame_i % self.render_every == 0:
            self.renderer.update_scene(self.d, camera=self.camera)
            self.frames.append(self.renderer.render())
        self._frame_i += 1

    def step(self):
        self._apply_state()
        self._set_base_mocap()
        mujoco.mj_forward(self.m, self.d)
        self._maybe_render()

    def settle(self, n):
        for _ in range(n):
            self.step()

    # ----------------------------------------------------------------- 复位/初始化
    def reset(self):
        self.arm_target = HOME_QPOS.copy()
        self.grip_cmd = GRIP_OPEN
        self._apply_state()
        self._set_base_mocap()
        mujoco.mj_forward(self.m, self.d)
        self.settle(30)

    # ----------------------------------------------------------------- 接口 1：底座
    def drive_to(self, target, steps=160):
        """底座移动到 (x, y) 或 (x, y, yaw)。"""
        t = tuple(target)
        tx, ty = t[0], t[1]
        tyaw = t[2] if len(t) > 2 else self.base_yaw
        sx, sy, syaw = self.base_x, self.base_y, self.base_yaw
        for i in range(steps):
            a = (i + 1) / steps
            self.base_x = sx + (tx - sx) * a
            self.base_y = sy + (ty - sy) * a
            self.base_yaw = syaw + (tyaw - syaw) * a
            self.step()
        self.base_x, self.base_y, self.base_yaw = tx, ty, tyaw
        self.settle(20)

    # ----------------------------------------------------------------- IK
    def _err_and_jac(self, s, target_pos, target_quat, w_rot):
        """在 scratch s 上算加权 6D 误差与雅可比(只取 7 个手臂 dof 列)。"""
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        cur_pos = s.site_xpos[self.tcp_sid]
        cur_quat = np.zeros(4); err_rot = np.zeros(3)
        mujoco.mju_mat2Quat(cur_quat, s.site_xmat[self.tcp_sid])
        mujoco.mju_subQuat(err_rot, target_quat, cur_quat)
        err = np.concatenate([target_pos - cur_pos, err_rot * w_rot])
        nv = self.m.nv
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.m, s, jp, jr, self.tcp_sid)
        J = np.vstack([jp[:, self.arm_dofadr], jr[:, self.arm_dofadr] * w_rot])
        pos_err = float(np.linalg.norm(target_pos - cur_pos))
        return err, J, pos_err

    def _lm_solve(self, s, q0, target_pos, target_quat, iters, w_rot):
        """Levenberg–Marquardt + 步长接受(只接受使误差下降的步)。返回 (q, pos_err)。"""
        lo = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
                       for n in ARM_JOINTS])
        hi = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
                       for n in ARM_JOINTS])

        def setq(qq):
            for k, adr in enumerate(self.arm_qadr):
                s.qpos[adr] = qq[k]

        q = np.clip(q0, lo, hi)
        setq(q)
        err, J, pe = self._err_and_jac(s, target_pos, target_quat, w_rot)
        e = np.linalg.norm(err)
        lam = 1e-2
        for _ in range(iters):
            if pe < 5e-4:
                break
            improved = False
            for _ls in range(40):
                H = J.T @ J + lam * np.eye(7)
                dq = np.linalg.solve(H, J.T @ err)
                qn = np.clip(q + dq, lo, hi)
                setq(qn)
                err_n, J_n, pe_n = self._err_and_jac(s, target_pos, target_quat, w_rot)
                e_n = np.linalg.norm(err_n)
                if e_n < e:
                    q, err, J, e, pe = qn, err_n, J_n, e_n, pe_n
                    lam = max(lam * 0.5, 1e-6)
                    improved = True
                    break
                lam *= 3.0
                if lam > 1e9:
                    break
            if not improved:
                break
        return q, pe

    def solve_ik(self, target_pos, target_quat, restarts=14, iters=80, w_rot=0.6, tol=1.5e-3):
        """带多次随机重启的 LM 雅可比 IK（类比 Genesis 的 max_samples），
        以指尖位置误差最小为准挑选最优解，鲁棒避开局部极小。返回 7 轴关节角。"""
        target_pos = np.asarray(target_pos, float)
        target_quat = np.asarray(target_quat, float)
        s = self.ik_data
        lo = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
                       for n in ARM_JOINTS])
        hi = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
                       for n in ARM_JOINTS])
        rng = np.random.default_rng(0)
        best_q, best_pe = None, np.inf
        for r in range(restarts):
            s.qpos[:] = self.d.qpos
            s.qvel[:] = 0
            if r == 0:
                seed = np.array(self.arm_target, float)
            elif r == 1:
                seed = HOME_QPOS.copy()
            else:
                seed = lo + rng.random(7) * (hi - lo)
            q, pe = self._lm_solve(s, seed, target_pos, target_quat, iters, w_rot)
            if pe < best_pe:
                best_pe, best_q = pe, q
            if best_pe < tol:
                break
        return best_q

    # ----------------------------------------------------------------- 接口 2：末端 6 维位姿
    def move_arm(self, q_goal, steps=100, settle=20, grip=None):
        q0 = np.array(self.arm_target, float)
        q_goal = np.asarray(q_goal, float)
        for i in range(steps):
            a = (i + 1) / steps
            self.arm_target = q0 + (q_goal - q0) * a
            if grip is not None:
                self.grip_cmd = grip
            self.step()
        self.arm_target = q_goal
        self.settle(settle)

    def move_tcp(self, pos, quat=GRASP_QUAT_FWD, steps=100, settle=20, grip=None):
        q = self.solve_ik(pos, quat)
        self.move_arm(q, steps=steps, settle=settle, grip=grip)
        return q

    def move_ee_pose(self, pos, rpy=(0.0, 0.0, 0.0), steps=100, settle=25, grip=None):
        """6 维末端位姿接口：xyz + 欧拉角 rpy。返回到达后指尖位置误差(m)。"""
        quat = euler_to_quat(rpy[0], rpy[1], rpy[2])
        self.move_tcp(pos, quat=quat, steps=steps, settle=settle, grip=grip)
        return float(np.linalg.norm(self.tcp_pos() - np.asarray(pos, float)))

    def set_gripper(self, cmd, steps=50):
        self.grip_cmd = cmd
        self.settle(steps)
