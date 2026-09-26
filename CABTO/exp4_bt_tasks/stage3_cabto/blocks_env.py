"""BlocksEnv —— stage3 多物体抓取环境（MuJoCo）
================================================
仿照 exp2 的 FrankaDPEnv，但突破其「单 freejoint 物体 / 单 grasp_weld」限制，
支持 **多个具名 freejoint 物体 + 多 weld + place 语义**，作为 CABTO stage3
（LLM 规划 + 代码生成 + VLM 校验）闭环的物理底座。

已调通机制原样照搬 FrankaDPEnv：
- IK：LM 雅可比多重启 DLS（_err_and_jac / _lm_solve / solve_ik）。
- step：末端 6D delta + EE_SERVO_BIAS 前馈补偿 + 渐进闭合斜坡 + 每子步 grasp_assist。
- weld 吸附：按当前 hand↔object 相对位姿写 eq_data，激活对应物体的 weld 锁死。

多物体扩展：
- grasp_assist 选取「当前夹爪间最近、且在指间」的物体，闭合时激活该物体的 weld，
  松开时解除所有 weld（一次只 weld 一个）。
- get_object_pose(name) / is_holding(name) 支持按名查询；place 由上层把物体移到
  目标顶面上方后 open gripper 释放实现（解除 weld，物体自由落稳）。

对外核心方法（供 codegen/executor 上层调用）
- reset() / step(action, sub_steps) / solve_ik(pos, quat)
- get_ee_pose() / get_object_pose(name) / get_gripper_width()
- is_holding(name=None) / grasp_assist()
- render_cam(name) / render_depth(name)
- start_record() / record_frame(cam) / save_video(path, fps)
"""

from __future__ import annotations

import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

import blocks_scene


# ----------------------- 常量（与 FrankaDPEnv 对齐） ----------------------- #
TABLE_TOP_Z = 0.40
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])     # hand +z 朝下

FINGER_OPEN = 0.04
GRIP_OPEN_CTRL = 255.0
GRIP_CLOSE_CTRL = 0.0
GRIP_CLOSE_RAMP = 6.0

HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

EE_POS_LOW = np.array([0.30, -0.30, TABLE_TOP_Z + 0.01])
EE_POS_HIGH = np.array([0.75, 0.30, TABLE_TOP_Z + 0.45])

MAX_DPOS = 0.05
MAX_DROT = 0.30
EE_SERVO_BIAS = np.array([0.024, 0.0, -0.007])

ACTION_DIM = 7
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"actuator{i}" for i in range(1, 8)]
GRIP_ACT = "actuator8"

BLOCK_HALF = blocks_scene.BLOCK_HALF            # 0.025
BLOCK_NAMES = [n for n, _, _ in blocks_scene.BLOCKS]


# ----------------------- 旋转工具（内联，stage3 自含） ----------------------- #
def _wxyz_to_xyzw(q):
    q = np.asarray(q, float).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q):
    q = np.asarray(q, float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]])


def apply_delta_rotvec(q_wxyz, delta_rotvec):
    r_old = R.from_quat(_wxyz_to_xyzw(q_wxyz))
    r_delta = R.from_rotvec(np.asarray(delta_rotvec, float).reshape(3))
    return _xyzw_to_wxyz((r_delta * r_old).as_quat())


def clip_rotvec(rotvec, max_angle):
    rotvec = np.asarray(rotvec, float).reshape(3)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-9 or angle <= max_angle:
        return rotvec
    return rotvec * (max_angle / angle)


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


class BlocksEnv:
    def __init__(self, render=True, img_size=512):
        self.img_size = int(img_size)
        self.render_enabled = bool(render)
        self.renderer = None
        self.depth_renderer = None
        self._rec_frames = []

        # 编译场景
        xml = blocks_scene.build_blocks_scene_xml()
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        # ---- 索引 ----
        self.tcp_sid = _id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.arm_qadr = [self.m.jnt_qposadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in ARM_JOINTS]
        self.arm_dofadr = [self.m.jnt_dofadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                           for n in ARM_JOINTS]
        self.finger_qadr = [self.m.jnt_qposadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                            for n in ("finger_joint1", "finger_joint2")]
        self.act_arm = [_id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTS]
        self.act_grip = _id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACT)

        self.hand_bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.lfinger_bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.rfinger_bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        # 每个物体：freejoint qadr/dofadr、body id、对应 weld id
        self.obj_qadr = {}
        self.obj_dofadr = {}
        self.obj_bid = {}
        self.obj_weld = {}
        for name in BLOCK_NAMES:
            jid = _id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            self.obj_qadr[name] = self.m.jnt_qposadr[jid]
            self.obj_dofadr[name] = self.m.jnt_dofadr[jid]
            self.obj_bid[name] = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            suffix = name.split("_")[0]
            self.obj_weld[name] = _id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                      f"grasp_weld_{suffix}")

        # 初始位姿快照（reset 用）
        mujoco.mj_forward(self.m, self.d)
        self._init_obj_qpos = {}
        for name in BLOCK_NAMES:
            b = self.obj_qadr[name]
            self._init_obj_qpos[name] = self.d.qpos[b:b + 7].copy()

        # IK 关节限位
        self._jnt_lo = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
                                 for n in ARM_JOINTS])
        self._jnt_hi = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
                                 for n in ARM_JOINTS])

        self.arm_target = HOME_QPOS.copy()
        self.ik_data = mujoco.MjData(self.m)

        self._grip_cmd = 1.0
        self._grip_ctrl_cur = GRIP_OPEN_CTRL
        self._grasp_enabled = True   # 关掉后 grasp_assist 不再激活新 weld（用于 place 收尾）
        self.lock_down = True        # 末端姿态持续朝垂直向下纠正，避免移动中歪斜
        self.DOWN_CORR_STEP = 0.08   # 每步朝垂直纠正的最大角(rad≈4.6°)，平滑回正不毁 position

        if self.render_enabled:
            try:
                self.renderer = mujoco.Renderer(self.m, self.img_size, self.img_size)
                self.depth_renderer = mujoco.Renderer(self.m, self.img_size, self.img_size)
                self.depth_renderer.enable_depth_rendering()
            except Exception as e:
                print(f"[警告] 渲染器初始化失败，降级零图（物理不受影响）: {e}")
                self.render_enabled = False
                self.renderer = None
                self.depth_renderer = None

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self._set_all_welds_inactive()

        for adr, v in zip(self.arm_qadr, HOME_QPOS):
            self.d.qpos[adr] = v
        for adr in self.finger_qadr:
            self.d.qpos[adr] = FINGER_OPEN
        for name in BLOCK_NAMES:
            b = self.obj_qadr[name]
            self.d.qpos[b:b + 7] = self._init_obj_qpos[name]
            self.d.qvel[self.obj_dofadr[name]:self.obj_dofadr[name] + 6] = 0.0

        self.arm_target = HOME_QPOS.copy()
        self._grip_cmd = 1.0
        self._grip_ctrl_cur = GRIP_OPEN_CTRL
        self._grasp_enabled = True

        self.d.ctrl[self.act_arm] = HOME_QPOS
        self.d.ctrl[self.act_grip] = GRIP_OPEN_CTRL
        mujoco.mj_forward(self.m, self.d)
        for _ in range(80):
            self.d.ctrl[self.act_arm] = HOME_QPOS
            self.d.ctrl[self.act_grip] = GRIP_OPEN_CTRL
            mujoco.mj_step(self.m, self.d)
        return self.get_ee_pose()

    # ------------------------------------------------------------------ #
    # IK（照搬 FrankaDPEnv）
    # ------------------------------------------------------------------ #
    def _err_and_jac(self, s, target_pos, target_quat, w_rot):
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        cur_pos = s.site_xpos[self.tcp_sid]
        cur_quat = np.zeros(4)
        err_rot = np.zeros(3)
        mujoco.mju_mat2Quat(cur_quat, s.site_xmat[self.tcp_sid])
        mujoco.mju_subQuat(err_rot, target_quat, cur_quat)
        # subQuat 是当前局部轴误差，旋转 Jacobian 使用世界轴。
        err_rot = s.site_xmat[self.tcp_sid].reshape(3, 3) @ err_rot
        err = np.concatenate([target_pos - cur_pos, err_rot * w_rot])
        nv = self.m.nv
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.m, s, jp, jr, self.tcp_sid)
        J = np.vstack([jp[:, self.arm_dofadr], jr[:, self.arm_dofadr] * w_rot])
        pos_err = float(np.linalg.norm(target_pos - cur_pos))
        rot_err = float(np.linalg.norm(err_rot))   # 姿态误差(弧度,未乘权重)
        return err, J, pos_err, rot_err

    def _lm_solve(self, s, q0, target_pos, target_quat, iters, w_rot):
        lo, hi = self._jnt_lo, self._jnt_hi

        def setq(qq):
            for k, adr in enumerate(self.arm_qadr):
                s.qpos[adr] = qq[k]

        q = np.clip(q0, lo, hi)
        setq(q)
        err, J, pe, re = self._err_and_jac(s, target_pos, target_quat, w_rot)
        e = np.linalg.norm(err)
        # 全程跟踪综合代价最优的解（位置为主 + 姿态弧度小权重折算）
        best = (q.copy(), pe, re, pe + 0.05 * re)
        lam = 1e-2
        stall = 0
        for _ in range(iters):
            if pe < 5e-4 and re < 1e-3:
                break
            improved = False
            for _ls in range(40):
                H = J.T @ J + lam * np.eye(7)
                dq = np.linalg.solve(H, J.T @ err)
                qn = np.clip(q + dq, lo, hi)
                setq(qn)
                err_n, J_n, pe_n, re_n = self._err_and_jac(s, target_pos, target_quat, w_rot)
                e_n = np.linalg.norm(err_n)
                if e_n < e:
                    q, err, J, e, pe, re = qn, err_n, J_n, e_n, pe_n, re_n
                    lam = max(lam * 0.5, 1e-6)
                    improved = True
                    cost = pe + 0.05 * re
                    if cost < best[3]:
                        best = (q.copy(), pe, re, cost)
                    break
                lam *= 3.0
                if lam > 1e9:
                    break
            if not improved:
                # 脊点卡死：不直接退出，重置 lambda + 小随机扰动跳出，最多容忍几次。
                stall += 1
                if stall > 3:
                    break
                lam = 1e-2
                q = np.clip(q + 1e-2 * (np.random.default_rng(stall).random(7) - 0.5), lo, hi)
                setq(q)
                err, J, pe, re = self._err_and_jac(s, target_pos, target_quat, w_rot)
                e = np.linalg.norm(err)
            else:
                stall = 0
        # 返回全程综合最优解（而非最后一步解）
        bq, bpe, bre, _ = best
        return bq, bpe, bre

    def solve_ik(self, target_pos, target_quat, restarts=24, iters=120, w_rot=0.5, tol=1.0e-3):
        target_pos = np.asarray(target_pos, float)
        target_quat = np.asarray(target_quat, float)
        s = self.ik_data
        rng = np.random.default_rng(0)
        # 综合代价选解：position 误差为主，姿态误差(弧度)按 POS_TOL 等价折算加权，
        # 避免出现「position 达标但姿态歪」或「姿态正但 position 差」的劣质解。
        POS_TOL = 2.0e-3        # 2mm 内视为 position 达标
        ROT_W = 0.05            # 姿态误差权重（米/弧度量级折算）
        best_q, best_cost = None, np.inf
        best_pe = best_re = np.inf
        for r in range(restarts):
            s.qpos[:] = self.d.qpos
            s.qvel[:] = 0
            if r == 0:
                seed = np.array(self.arm_target, float)
            elif r == 1:
                seed = HOME_QPOS.copy()
            else:
                seed = self._jnt_lo + rng.random(7) * (self._jnt_hi - self._jnt_lo)
            q, pe, re = self._lm_solve(s, seed, target_pos, target_quat, iters, w_rot)
            cost = pe + ROT_W * re
            if cost < best_cost:
                best_cost, best_pe, best_re, best_q = cost, pe, re, q
            if best_pe < tol and best_re < 0.03:   # 同一个 best 的 position 和姿态均达标
                break
        return best_q

    # ------------------------------------------------------------------ #
    # step（照搬 FrankaDPEnv：6D delta + bias + ramp + 每子步 grasp_assist）
    # ------------------------------------------------------------------ #
    def step(self, action, sub_steps: int = 16):
        action = np.asarray(action, dtype=float).reshape(ACTION_DIM)
        dpos = np.clip(action[:3], -MAX_DPOS, MAX_DPOS)
        drot = clip_rotvec(action[3:6], MAX_DROT)
        grip = float(np.clip(action[6], 0.0, 1.0))
        self._grip_cmd = grip

        cur_pos, cur_quat = self.get_ee_pose()
        tgt_pos = np.clip(cur_pos + dpos, EE_POS_LOW, EE_POS_HIGH)
        # 姿态策略：position 用低姿态权重 IK（精度高，已验证），姿态靠「朝垂直的有界
        # 小角度增量纠正」逐步回正——既不破坏 position 闭环，又把末端缓慢拉回垂直向下。
        if np.linalg.norm(drot) > 1e-9:
            # 显式转腕：在当前姿态上叠加增量
            tgt_quat = apply_delta_rotvec(cur_quat, drot)
        elif self.lock_down:
            # 计算当前姿态→GRASP_QUAT 的旋转误差，取有界小步朝垂直纠正
            err_rot = np.zeros(3)
            mujoco.mju_subQuat(err_rot, GRASP_QUAT, cur_quat)
            err_rot = R.from_quat(_wxyz_to_xyzw(cur_quat)).apply(err_rot)
            corr = clip_rotvec(err_rot, self.DOWN_CORR_STEP)
            tgt_quat = apply_delta_rotvec(cur_quat, corr)
        else:
            tgt_quat = cur_quat

        ik_target = tgt_pos + EE_SERVO_BIAS
        # w_rot=0.5：position 优先（小角度姿态增量下 IK 收敛好，position 不被牺牲）。
        arm_q = self.solve_ik(ik_target, tgt_quat, w_rot=0.5)
        if arm_q is None:
            arm_q = self.arm_target
        self.arm_target = arm_q

        grip_ctrl_goal = GRIP_CLOSE_CTRL + (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL) * grip
        ramp = GRIP_CLOSE_RAMP
        for _ in range(sub_steps):
            if self._grip_ctrl_cur < grip_ctrl_goal:
                self._grip_ctrl_cur = min(grip_ctrl_goal, self._grip_ctrl_cur + GRIP_OPEN_CTRL)
            elif self._grip_ctrl_cur > grip_ctrl_goal:
                self._grip_ctrl_cur = max(grip_ctrl_goal, self._grip_ctrl_cur - ramp)
            self.d.ctrl[self.act_arm] = arm_q
            self.d.ctrl[self.act_grip] = self._grip_ctrl_cur
            mujoco.mj_step(self.m, self.d)
            self.grasp_assist()
        return self.get_ee_pose()

    # ------------------------------------------------------------------ #
    # 位姿 / 状态查询
    # ------------------------------------------------------------------ #
    def get_ee_pose(self):
        pos = self.d.site_xpos[self.tcp_sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.tcp_sid])
        return pos, quat

    def get_object_pose(self, name):
        b = self.obj_qadr[name]
        pos = self.d.qpos[b:b + 3].copy()
        quat = self.d.qpos[b + 3:b + 7].copy()
        return pos, quat

    def get_gripper_width(self):
        q = np.array([self.d.qpos[a] for a in self.finger_qadr])
        return float(q[0] + q[1])

    def _active_weld_name(self):
        for name in BLOCK_NAMES:
            wid = self.obj_weld[name]
            if wid >= 0 and self.d.eq_active[wid] == 1:
                return name
        return None

    def is_holding(self, name=None) -> bool:
        """name=None：是否抓着任意物体；否则是否抓着指定物体。
        判定：夹爪闭合 且 (该物体 weld 激活 或 物体已离桌)。"""
        if self._grip_cmd >= 0.5:
            return False
        if name is None:
            return self._active_weld_name() is not None
        wid = self.obj_weld[name]
        welded = (wid >= 0 and self.d.eq_active[wid] == 1)
        z = float(self.d.qpos[self.obj_qadr[name] + 2])
        lifted = z > (TABLE_TOP_Z + BLOCK_HALF + 0.02)
        return bool(welded or (lifted and self._obj_between_fingers(name)))

    # ------------------------------------------------------------------ #
    # grasp_assist（多物体：选最近且在指间的物体激活其 weld，一次只一个）
    # ------------------------------------------------------------------ #
    def _finger_mid(self):
        lfp = self.d.xpos[self.lfinger_bid]
        rfp = self.d.xpos[self.rfinger_bid]
        return 0.5 * (lfp + rfp)

    def _obj_between_fingers(self, name, xy_tol=0.03, z_tol=0.05) -> bool:
        mid = self._finger_mid()
        obj = self.d.qpos[self.obj_qadr[name]:self.obj_qadr[name] + 3]
        horiz = float(np.linalg.norm(mid[:2] - obj[:2]))
        vert = abs(float(mid[2] - obj[2]))
        return horiz < xy_tol and vert < z_tol

    def _nearest_graspable(self):
        """返回当前指间中点最近、且在指间范围内的物体名（否则 None）。"""
        mid = self._finger_mid()
        best, best_d = None, np.inf
        for name in BLOCK_NAMES:
            obj = self.d.qpos[self.obj_qadr[name]:self.obj_qadr[name] + 3]
            horiz = float(np.linalg.norm(mid[:2] - obj[:2]))
            vert = abs(float(mid[2] - obj[2]))
            if horiz < 0.03 and vert < 0.05 and horiz < best_d:
                best, best_d = name, horiz
        return best

    def _set_weld(self, name, active: bool):
        wid = self.obj_weld[name]
        if wid < 0:
            return
        if active:
            hp = self.d.xpos[self.hand_bid].copy()
            hq = np.zeros(4); mujoco.mju_mat2Quat(hq, self.d.xmat[self.hand_bid])
            ob = self.obj_bid[name]
            op = self.d.xpos[ob].copy()
            oq = np.zeros(4); mujoco.mju_mat2Quat(oq, self.d.xmat[ob])
            neg_hq = np.zeros(4); mujoco.mju_negQuat(neg_hq, hq)
            rel_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_pos, op - hp, neg_hq)
            rel_quat = np.zeros(4); mujoco.mju_mulQuat(rel_quat, neg_hq, oq)
            data = self.m.eq_data[wid]
            data[0:3] = 0.0
            data[3:6] = rel_pos
            data[6:10] = rel_quat
            data[10] = 1.0
            self.m.eq_active0[wid] = 1
            self.d.eq_active[wid] = 1
        else:
            self.m.eq_active0[wid] = 0
            self.d.eq_active[wid] = 0

    def _set_all_welds_inactive(self):
        for name in BLOCK_NAMES:
            self._set_weld(name, False)

    def grasp_assist(self):
        """夹爪闭合时：若指间有物体且尚无 weld 激活，则激活最近物体的 weld。
        松开时：解除所有 weld。一次只锁一个物体。"""
        if not self._grasp_enabled:
            # 关闭自动 grasp_assist：weld 状态完全由上层手动控制（place 收尾用）
            return
        closed = self._grip_cmd < 0.5
        if closed:
            if self._active_weld_name() is None:
                cand = self._nearest_graspable()
                if cand is not None:
                    self._set_weld(cand, True)
        else:
            if self._active_weld_name() is not None:
                self._set_all_welds_inactive()

    # ------------------------------------------------------------------ #
    # 定点保持（绕过 IK/BIAS，冻结当前关节角，只动夹爪）—— 供 place 收尾用，
    # 避免 step 内 EE_SERVO_BIAS 前馈每步把 hand 往偏置方向推、拖走 weld 锁定的物体。
    # ------------------------------------------------------------------ #
    def hold_arm(self, grip, sub_steps=16):
        """保持当前 arm_target 关节角不变，仅把夹爪 ctrl 朝 grip 目标渐进，手动推进。
        不解 IK、不加 BIAS、不动手臂关节目标，故 hand 不漂移。"""
        grip = float(np.clip(grip, 0.0, 1.0))
        self._grip_cmd = grip
        grip_ctrl_goal = GRIP_CLOSE_CTRL + (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL) * grip
        for _ in range(sub_steps):
            if self._grip_ctrl_cur < grip_ctrl_goal:
                self._grip_ctrl_cur = min(grip_ctrl_goal, self._grip_ctrl_cur + GRIP_OPEN_CTRL)
            elif self._grip_ctrl_cur > grip_ctrl_goal:
                self._grip_ctrl_cur = max(grip_ctrl_goal, self._grip_ctrl_cur - GRIP_CLOSE_RAMP)
            self.d.ctrl[self.act_arm] = self.arm_target
            self.d.ctrl[self.act_grip] = self._grip_ctrl_cur
            mujoco.mj_step(self.m, self.d)
            self.grasp_assist()
        return self.get_ee_pose()

    def lift_straight(self, dz, sub_steps=16, lock_xy=None):
        """竖直抬升 dz（xy 锁定到 lock_xy，默认当前 tcp xy；保持朝下姿态），
        解 IK 后用冻结推进，过程中 hand xy 不漂、不刮碰下方已放置物体。"""
        cur_pos, cur_quat = self.get_ee_pose()
        tgt = cur_pos.copy()
        if lock_xy is not None:
            tgt[0], tgt[1] = float(lock_xy[0]), float(lock_xy[1])
        tgt[2] = min(cur_pos[2] + dz, EE_POS_HIGH[2])
        ik_target = tgt + EE_SERVO_BIAS
        # 仅从当前关节角局部求解（不全局多重启），避免 IK 跳到远端解使 hand 瞬移撞飞物体
        s = self.ik_data
        s.qpos[:] = self.d.qpos
        s.qvel[:] = 0
        q, pe = self._lm_solve(s, np.array(self.arm_target, float),
                               ik_target, GRASP_QUAT, iters=120, w_rot=0.5)
        if q is not None:
            self.arm_target = q
        return self.hold_arm(self._grip_cmd, sub_steps=sub_steps)

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def render_cam(self, cam_name="overview"):
        if not self.render_enabled or self.renderer is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.renderer.update_scene(self.d, camera=cam_name)
        return np.asarray(self.renderer.render(), dtype=np.uint8)[..., :3]

    def render_depth(self, cam_name="overview"):
        if not self.render_enabled or self.depth_renderer is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)
        self.depth_renderer.update_scene(self.d, camera=cam_name)
        return np.asarray(self.depth_renderer.render(), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 录制
    # ------------------------------------------------------------------ #
    def start_record(self):
        self._rec_frames = []

    def record_frame(self, cam="overview"):
        self._rec_frames.append(self.render_cam(cam))

    def save_video(self, path, fps=30):
        if not self._rec_frames:
            print("[警告] 无录制帧，跳过保存")
            return
        try:
            import imageio
            imageio.mimwrite(path, self._rec_frames, fps=fps, quality=8)
            print(f"[ok] 视频已保存 {path}（{len(self._rec_frames)} 帧）")
        except Exception as e:
            print(f"[警告] 保存视频失败: {e}")


if __name__ == "__main__":
    env = BlocksEnv(render=False)
    env.reset()
    print("[ok] BlocksEnv 构建+reset 成功")
    print("  ee:", env.get_ee_pose()[0])
    for n in BLOCK_NAMES:
        print(f"  {n}:", env.get_object_pose(n)[0])
    print("  holding(any):", env.is_holding())
