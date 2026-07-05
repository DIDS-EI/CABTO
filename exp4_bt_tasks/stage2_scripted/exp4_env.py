"""exp4 通用任务环境（MuJoCo）—— 泛化 stage3 的 BlocksEnv
=========================================================
把已在 stage3 调通的「IK + 6D delta step + EE_SERVO_BIAS 前馈 + 渐进闭合 +
grasp_weld 吸附 + 朝下姿态增量纠正」一套机制，从写死单臂 blocks 场景，
泛化为：
  - 任意场景 XML（由各任务的 scene 生成器产出，含 panda + 物体 + grasp_weld）；
  - 任意一组可抓 freejoint 物体（按名查询/吸附）；
  - 单臂（suffix=""）或双臂（suffix="_L"/"_R" 各一条 ArmInterface）。

设计：一条臂的全部状态/控制封装进 ArmInterface（参数化 suffix），
Exp4Env 持有 1~2 条臂 + 物体索引 + 渲染录制，供各任务 scripted expert 调用。

与 blocks_env 的关系：单臂行为逐行对齐（同样的 IK/step/grasp_assist），
故 stage3 的垂直抓取修复成果在这里原样继承。
"""

from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


# ----------------------- 常量（与 blocks_env 对齐） ----------------------- #
TABLE_TOP_Z = 0.40
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])     # hand +z 朝下（垂直向下抓）

FINGER_OPEN = 0.04
GRIP_OPEN_CTRL = 255.0
GRIP_CLOSE_CTRL = 0.0
GRIP_CLOSE_RAMP = 6.0

HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

MAX_DPOS = 0.05
MAX_DROT = 0.30
EE_SERVO_BIAS = np.array([0.024, 0.0, -0.007])   # 单臂标定的前馈（双臂经基座旋转换算）

ACTION_DIM = 7


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


class ArmInterface:
    """单条 Franka 臂的 IK / step / grasp 封装（参数化 suffix）。

    多条臂共享同一 (model, data)，各自只读写自己后缀的 joint/actuator/site/finger。
    grasp_weld 是「物体级」的（grasp_weld_{arm}_{obj}），运行时按该臂抓哪个物体激活。
    """

    def __init__(self, env, suffix, base_yaw, obj_names):
        self.env = env
        self.m, self.d = env.m, env.d
        self.s = suffix                       # "" / "_L" / "_R"
        self.base_yaw = float(base_yaw)        # 该臂基座 yaw（用于把 EE_SERVO_BIAS 转到世界系）
        self.obj_names = list(obj_names)

        m = self.m
        self.tcp_sid = _id(m, mujoco.mjtObj.mjOBJ_SITE, f"tcp{suffix}")
        arm_joints = [f"joint{i}{suffix}" for i in range(1, 8)]
        arm_acts = [f"actuator{i}{suffix}" for i in range(1, 8)]
        self.arm_qadr = [m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in arm_joints]
        self.arm_dofadr = [m.jnt_dofadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in arm_joints]
        self.finger_qadr = [m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                            for n in (f"finger_joint1{suffix}", f"finger_joint2{suffix}")]
        self.act_arm = [_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in arm_acts]
        self.act_grip = _id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator8{suffix}")
        self.hand_bid = _id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand{suffix}")
        self.lfinger_bid = _id(m, mujoco.mjtObj.mjOBJ_BODY, f"left_finger{suffix}")
        self.rfinger_bid = _id(m, mujoco.mjtObj.mjOBJ_BODY, f"right_finger{suffix}")

        self._arm_name = arm_joints
        self._jnt_lo = np.array([m.jnt_range[_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
                                 for n in arm_joints])
        self._jnt_hi = np.array([m.jnt_range[_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
                                 for n in arm_joints])

        # 该臂对每个物体的 weld id（场景里命名 grasp_weld{suffix}_{obj}）。可能不存在(-1)。
        self.obj_weld = {}
        for name in self.obj_names:
            wid = _id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_weld{suffix}_{name}")
            self.obj_weld[name] = wid

        self.arm_target = HOME_QPOS.copy()
        self._grip_cmd = 1.0
        self._grip_ctrl_cur = GRIP_OPEN_CTRL
        self._grasp_enabled = True
        self.lock_down = True
        self.pour_local = False        # True 时 IK 仅局部求解(不随机重启)，用于倾倒严格连续
        self.abs_quat = None           # 非 None 时 _resolve_target 用此绝对姿态(倾倒用)
        self.DOWN_CORR_STEP = 0.08
        # 吸附判定容差（finger_mid 与物体中心）。抓「带高把手的薄盖」时，
        # 抓取点比物体中心高，需放宽垂直容差；任务可在 reset 后改。
        self.grasp_xy_tol = 0.035
        self.grasp_z_tol = 0.06
        # EE_SERVO_BIAS 在「基座坐标系」标定；该臂基座绕 z 转 base_yaw，把 bias 转到世界系
        c, sgn = np.cos(self.base_yaw), np.sin(self.base_yaw)
        bx, by, bz = EE_SERVO_BIAS
        self.bias_world = np.array([c * bx - sgn * by, sgn * bx + c * by, bz])
        # 朝下抓取姿态：单臂 GRASP_QUAT 是「base_yaw=0」时 hand +z 朝下的世界四元数。
        # 该臂基座绕 z 转了 base_yaw，要让 hand 同样朝下，目标世界四元数需左乘 R_z(base_yaw)。
        rz = R.from_euler("z", self.base_yaw).as_quat()       # xyzw
        gq = R.from_quat(_wxyz_to_xyzw(GRASP_QUAT))
        self.grasp_quat = _xyzw_to_wxyz((R.from_quat(rz) * gq).as_quat())

    # --------------------------------------------------------------- 复位
    def reset_arm(self):
        for adr, v in zip(self.arm_qadr, HOME_QPOS):
            self.d.qpos[adr] = v
        for adr in self.finger_qadr:
            self.d.qpos[adr] = FINGER_OPEN
        self.arm_target = HOME_QPOS.copy()
        self._grip_cmd = 1.0
        self._grip_ctrl_cur = GRIP_OPEN_CTRL
        self._grasp_enabled = True
        self.d.ctrl[self.act_arm] = HOME_QPOS
        self.d.ctrl[self.act_grip] = GRIP_OPEN_CTRL

    # --------------------------------------------------------------- IK
    def _err_and_jac(self, s, target_pos, target_quat, w_rot):
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        cur_pos = s.site_xpos[self.tcp_sid]
        cur_quat = np.zeros(4)
        err_rot = np.zeros(3)
        mujoco.mju_mat2Quat(cur_quat, s.site_xmat[self.tcp_sid])
        mujoco.mju_subQuat(err_rot, target_quat, cur_quat)
        err = np.concatenate([target_pos - cur_pos, err_rot * w_rot])
        nv = self.m.nv
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.m, s, jp, jr, self.tcp_sid)
        J = np.vstack([jp[:, self.arm_dofadr], jr[:, self.arm_dofadr] * w_rot])
        pos_err = float(np.linalg.norm(target_pos - cur_pos))
        rot_err = float(np.linalg.norm(err_rot))
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
                stall += 1
                if stall > 3:
                    break
                lam = 1e-2
                q = np.clip(q + 1e-2 * (np.random.default_rng(stall).random(7) - 0.5), lo, hi)
        bq, bpe, bre, _ = best
        return bq, bpe, bre

    def solve_ik(self, target_pos, target_quat, restarts=24, iters=120, w_rot=0.5,
                 tol=1.0e-3, local_only=False):
        target_pos = np.asarray(target_pos, float)
        target_quat = np.asarray(target_quat, float)
        s = self.env.ik_data
        rng = np.random.default_rng(0)
        ROT_W = 0.05
        # local_only：仅从当前 arm_target 单点起解、不做随机重启——用于「持物倾倒」等
        # 需要严格关节连续的小幅增量旋转，避免 IK 跳支把所持容器/内容物甩飞（致仿真爆炸）。
        if local_only:
            s.qpos[:] = self.d.qpos
            s.qvel[:] = 0
            q, pe, re = self._lm_solve(s, np.array(self.arm_target, float),
                                       target_pos, target_quat, iters, w_rot)
            return q
        # 关节连续性惩罚：在「位置/姿态都达标」的候选里，优先选离当前 arm_target 最近
        # 的关节解，避免 IK 随机重启跳到「位置误差略小但关节大幅翻转」的远支——
        # 那会让臂在 16 substep 内剧烈甩动、把所持物/桌面物飞出（右臂尤甚）。
        q_ref = np.array(self.arm_target, float)
        # 关节连续性惩罚【仅右臂启用】：实测左臂(base_yaw<0)在无惩罚时 IK 已稳定
        # 且能精确达点，加惩罚反而把它推离安全支造成发散（can 飞出）；右臂(base_yaw>0)
        # 则必须靠惩罚抑制随机重启跳支甩飞。故按臂别启用，避免 whack-a-mole。
        is_right = float(self.base_yaw) > 0.1
        JOINT_W = 0.20 if is_right else 0.0
        best_q, best_cost, best_pe = None, np.inf, np.inf
        for r in range(restarts):
            s.qpos[:] = self.d.qpos
            s.qvel[:] = 0
            if r == 0:
                seed = q_ref.copy()
            elif r == 1:
                seed = HOME_QPOS.copy()
            else:
                seed = self._jnt_lo + rng.random(7) * (self._jnt_hi - self._jnt_lo)
            q, pe, re = self._lm_solve(s, seed, target_pos, target_quat, iters, w_rot)
            jdist = float(np.linalg.norm(q - q_ref))
            # 仅当位姿基本达标时才计入关节惩罚（否则优先把位置解出来）
            reach_ok = (pe < 0.02 and re < 0.05)
            cost = pe + ROT_W * re + (JOINT_W * jdist if reach_ok else 0.0)
            if cost < best_cost:
                best_cost, best_pe, best_q = cost, pe, q
            # 早停：找到既达标又关节连续(jdist 小)的解就够了
            if best_pe < tol and re < 0.03 and (not is_right or jdist < 0.6):
                break
        return best_q

    # --------------------------------------------------------------- 位姿查询
    def get_ee_pose(self):
        pos = self.d.site_xpos[self.tcp_sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.tcp_sid])
        return pos, quat

    def get_gripper_width(self):
        q = np.array([self.d.qpos[a] for a in self.finger_qadr])
        return float(q[0] + q[1])

    def _finger_mid(self):
        return 0.5 * (self.d.xpos[self.lfinger_bid] + self.d.xpos[self.rfinger_bid])

    # --------------------------------------------------------------- grasp weld
    def _active_weld_name(self):
        for name in self.obj_names:
            wid = self.obj_weld[name]
            if wid >= 0 and self.d.eq_active[wid] == 1:
                return name
        return None

    def _obj_between_fingers(self, name, xy_tol=None, z_tol=None):
        xy_tol = self.grasp_xy_tol if xy_tol is None else xy_tol
        z_tol = self.grasp_z_tol if z_tol is None else z_tol
        mid = self._finger_mid()
        b = self.env.obj_qadr[name]
        obj = self.d.qpos[b:b + 3]
        horiz = float(np.linalg.norm(mid[:2] - obj[:2]))
        vert = abs(float(mid[2] - obj[2]))
        return horiz < xy_tol and vert < z_tol

    def _nearest_graspable(self, xy_tol=None, z_tol=None):
        xy_tol = self.grasp_xy_tol if xy_tol is None else xy_tol
        z_tol = self.grasp_z_tol if z_tol is None else z_tol
        mid = self._finger_mid()
        best, best_d = None, np.inf
        for name in self.obj_names:
            if self.obj_weld[name] < 0:
                continue
            b = self.env.obj_qadr[name]
            obj = self.d.qpos[b:b + 3]
            horiz = float(np.linalg.norm(mid[:2] - obj[:2]))
            vert = abs(float(mid[2] - obj[2]))
            if horiz < xy_tol and vert < z_tol and horiz < best_d:
                best, best_d = name, horiz
        return best

    def _set_weld(self, name, active):
        wid = self.obj_weld[name]
        if wid < 0:
            return
        if active:
            hp = self.d.xpos[self.hand_bid].copy()
            hq = np.zeros(4); mujoco.mju_mat2Quat(hq, self.d.xmat[self.hand_bid])
            ob = self.env.obj_bid[name]
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

    def release_all(self):
        for name in self.obj_names:
            self._set_weld(name, False)

    def grasp_assist(self):
        if not self._grasp_enabled:
            return
        closed = self._grip_cmd < 0.5
        if closed:
            if self._active_weld_name() is None:
                cand = self._nearest_graspable()
                if cand is not None:
                    self._set_weld(cand, True)
        else:
            if self._active_weld_name() is not None:
                self.release_all()

    def is_holding(self, name=None):
        if self._grip_cmd >= 0.5:
            return False
        if name is None:
            return self._active_weld_name() is not None
        wid = self.obj_weld[name]
        return bool(wid >= 0 and self.d.eq_active[wid] == 1)

    # --------------------------------------------------------------- 写关节目标（不步进）
    def set_arm_ctrl(self, arm_q, grip):
        self.arm_target = arm_q
        self._grip_cmd = float(np.clip(grip, 0.0, 1.0))


class Exp4Env:
    """exp4 通用任务环境：加载场景 XML，持 1~2 条臂 + 可抓物体索引 + 渲染录制。

    参数
    ----
    xml : 场景 XML 字符串。
    arms : [(suffix, base_yaw), ...]。单臂传 [("", 0.0)]；
           双臂传 [("_L", -pi/2), ("_R", +pi/2)]（与 scene 基座朝向一致）。
    obj_names : 可抓 freejoint 物体名列表（每个在场景里有 {name}_free 关节）。
    """

    def __init__(self, xml, arms, obj_names, render=True, img_size=512):
        self.img_size = int(img_size)
        self.render_enabled = bool(render)
        self.renderer = None
        self._rec_frames = []

        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.ik_data = mujoco.MjData(self.m)
        self.obj_names = list(obj_names)

        # 物体索引
        self.obj_qadr, self.obj_dofadr, self.obj_bid = {}, {}, {}
        for name in self.obj_names:
            jid = _id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            self.obj_qadr[name] = self.m.jnt_qposadr[jid]
            self.obj_dofadr[name] = self.m.jnt_dofadr[jid]
            self.obj_bid[name] = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

        # 臂
        self.arms = [ArmInterface(self, sfx, yaw, self.obj_names) for sfx, yaw in arms]
        self.arm = {a.s: a for a in self.arms}

        mujoco.mj_forward(self.m, self.d)
        self._init_obj_qpos = {}
        for name in self.obj_names:
            b = self.obj_qadr[name]
            self._init_obj_qpos[name] = self.d.qpos[b:b + 7].copy()

        if self.render_enabled:
            try:
                self.renderer = mujoco.Renderer(self.m, self.img_size, self.img_size)
            except Exception as e:
                print(f"[警告] 渲染器初始化失败: {e}")
                self.render_enabled = False

    # --------------------------------------------------------------- reset
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        for a in self.arms:
            a.release_all()
            a.reset_arm()
        for name in self.obj_names:
            b = self.obj_qadr[name]
            self.d.qpos[b:b + 7] = self._init_obj_qpos[name]
            self.d.qvel[self.obj_dofadr[name]:self.obj_dofadr[name] + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        for _ in range(80):
            for a in self.arms:
                self.d.ctrl[a.act_arm] = HOME_QPOS
                self.d.ctrl[a.act_grip] = GRIP_OPEN_CTRL
            mujoco.mj_step(self.m, self.d)

    # --------------------------------------------------------------- 物体查询
    def get_object_pose(self, name):
        b = self.obj_qadr[name]
        return self.d.qpos[b:b + 3].copy(), self.d.qpos[b + 3:b + 7].copy()

    def get_body_pos(self, body_name):
        """查询固定 body（无 freejoint，如 cup/pot/carton）的世界位置。"""
        bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.d.xpos[bid].copy()

    # --------------------------------------------------------------- 单臂 6D delta step
    def _resolve_target(self, arm, action):
        """把一条臂的 7 维 action 解析成 (arm_q, grip_goal)。"""
        action = np.asarray(action, float).reshape(ACTION_DIM)
        dpos = np.clip(action[:3], -MAX_DPOS, MAX_DPOS)
        drot = clip_rotvec(action[3:6], MAX_DROT)
        grip = float(np.clip(action[6], 0.0, 1.0))
        arm._grip_cmd = grip

        cur_pos, cur_quat = arm.get_ee_pose()
        tgt_pos = cur_pos + dpos
        abs_q = getattr(arm, "abs_quat", None)
        if abs_q is not None:
            # 倾倒：用「绝对目标姿态」而非「相对当前增量」——否则 IK 每步只追上一点点，
            # 下一步又从（几乎没转的）当前姿态加微小增量，累积被冲掉，腕部永远不转。
            tgt_quat = np.asarray(abs_q, float)
        elif np.linalg.norm(drot) > 1e-9:
            tgt_quat = apply_delta_rotvec(cur_quat, drot)
        elif arm.lock_down:
            err_rot = np.zeros(3)
            mujoco.mju_subQuat(err_rot, arm.grasp_quat, cur_quat)
            corr = clip_rotvec(err_rot, arm.DOWN_CORR_STEP)
            tgt_quat = apply_delta_rotvec(cur_quat, corr)
        else:
            tgt_quat = cur_quat
        ik_target = tgt_pos + arm.bias_world
        # 倾倒等需严格关节连续的阶段：用 local_only IK（不随机重启），防跳支甩飞；
        # 同时大幅提高姿态权重 w_rot——否则 LM 优先满足位置、牺牲旋转，导致命令了
        # 大角度翻转但腕部根本不转（实测 mouth_z 全程 +0.99），罐口始终朝上倒不出。
        pour = getattr(arm, "pour_local", False)
        arm_q = arm.solve_ik(ik_target, tgt_quat, w_rot=(4.0 if pour else 0.5),
                             local_only=pour)
        if arm_q is None:
            arm_q = arm.arm_target
        arm.arm_target = arm_q
        grip_goal = GRIP_CLOSE_CTRL + (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL) * grip
        return grip_goal

    def step_arms(self, actions, sub_steps=16):
        """同时推进若干臂。actions: {suffix: 7d action}。未给的臂保持当前目标。"""
        grip_goals = {}
        for a in self.arms:
            if a.s in actions and actions[a.s] is not None:
                grip_goals[a.s] = self._resolve_target(a, actions[a.s])
            else:
                grip_goals[a.s] = GRIP_CLOSE_CTRL + (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL) * a._grip_cmd
        for _ in range(sub_steps):
            for a in self.arms:
                goal = grip_goals[a.s]
                if a._grip_ctrl_cur < goal:
                    a._grip_ctrl_cur = min(goal, a._grip_ctrl_cur + GRIP_OPEN_CTRL)
                elif a._grip_ctrl_cur > goal:
                    a._grip_ctrl_cur = max(goal, a._grip_ctrl_cur - GRIP_CLOSE_RAMP)
                self.d.ctrl[a.act_arm] = a.arm_target
                self.d.ctrl[a.act_grip] = a._grip_ctrl_cur
            mujoco.mj_step(self.m, self.d)
            for a in self.arms:
                a.grasp_assist()

    def hold_arms(self, grips=None, sub_steps=16):
        """冻结所有臂关节角，仅推进夹爪 ctrl（grips: {suffix: grip}），不漂移。"""
        grips = grips or {}
        for a in self.arms:
            if a.s in grips:
                a._grip_cmd = float(np.clip(grips[a.s], 0.0, 1.0))
        goals = {a.s: GRIP_CLOSE_CTRL + (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL) * a._grip_cmd
                 for a in self.arms}
        for _ in range(sub_steps):
            for a in self.arms:
                goal = goals[a.s]
                if a._grip_ctrl_cur < goal:
                    a._grip_ctrl_cur = min(goal, a._grip_ctrl_cur + GRIP_OPEN_CTRL)
                elif a._grip_ctrl_cur > goal:
                    a._grip_ctrl_cur = max(goal, a._grip_ctrl_cur - GRIP_CLOSE_RAMP)
                self.d.ctrl[a.act_arm] = a.arm_target
                self.d.ctrl[a.act_grip] = a._grip_ctrl_cur
            mujoco.mj_step(self.m, self.d)
            for a in self.arms:
                a.grasp_assist()

    def step_joint(self, arm, arm_q, grip, sub_steps=16):
        """直接以给定关节角 arm_q 驱动指定臂（不走 IK），其余臂保持目标不动。
        用于「持物倾倒」等需严格关节空间插值、绝不让 IK 跳支甩飞的阶段。"""
        arm.arm_target = np.asarray(arm_q, float)
        arm._grip_cmd = float(np.clip(grip, 0.0, 1.0))
        goals = {}
        for a in self.arms:
            goals[a.s] = GRIP_CLOSE_CTRL + (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL) * a._grip_cmd
        for _ in range(sub_steps):
            for a in self.arms:
                goal = goals[a.s]
                if a._grip_ctrl_cur < goal:
                    a._grip_ctrl_cur = min(goal, a._grip_ctrl_cur + GRIP_OPEN_CTRL)
                elif a._grip_ctrl_cur > goal:
                    a._grip_ctrl_cur = max(goal, a._grip_ctrl_cur - GRIP_CLOSE_RAMP)
                self.d.ctrl[a.act_arm] = a.arm_target
                self.d.ctrl[a.act_grip] = a._grip_ctrl_cur
            mujoco.mj_step(self.m, self.d)
            for a in self.arms:
                a.grasp_assist()

    # --------------------------------------------------------------- 渲染录制
    def render_cam(self, cam_name="overview"):
        if not self.render_enabled or self.renderer is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.renderer.update_scene(self.d, camera=cam_name)
        return np.asarray(self.renderer.render(), dtype=np.uint8)[..., :3]

    def start_record(self):
        self._rec_frames = []

    def record_frame(self, cam="overview"):
        self._rec_frames.append(self.render_cam(cam))

    def save_video(self, path, fps=30):
        if not self._rec_frames:
            print("[警告] 无录制帧")
            return
        try:
            import imageio
            imageio.mimwrite(path, self._rec_frames, fps=fps, quality=8)
            print(f"[ok] 视频已保存 {path}（{len(self._rec_frames)} 帧）")
        except Exception as e:
            print(f"[警告] 保存视频失败: {e}")

    def save_png(self, path, cam="overview"):
        try:
            import imageio
            imageio.imwrite(path, self.render_cam(cam))
        except Exception as e:
            print(f"[警告] 保存图片失败: {e}")
