"""Franka 双相机抓取环境（MuJoCo 版，采集 / 评估共用）
========================================================
与 Genesis 版 `dp_common/franka_dp_env.py` **接口完全一致**，但底层换成 MuJoCo：

- 机器人：Franka Emika Panda，固定底座（立柱顶 z=0.40），前方一张桌子。
- 相机：
    * front : 正面第三视角相机（世界固定，俯视桌面）
    * wrist : 腕部相机（固连在 hand 上，随末端运动）
- 物体：桌上随机位置一个几何基元（见 dp_objects）。物体作为 freejoint body
        在 build 时注入场景 XML 占位符后整体编译。
- 动作：末端 6D 位姿 delta + 夹爪开度 action = [dx,dy,dz, drx,dry,drz, grip] (7 维)
        平移 m，姿态 rotvec(rad)，世界系；grip 1=张开 0=闭合。
- 观测：obs = {front_rgb(H,W,3)uint8, wrist_rgb(H,W,3)uint8, agent_pos(10)}
        agent_pos = pos(3)+quat(4)+grip_width(1)+holding(1)+grip_cmd(1)

与 Genesis 版的关键差异
-----------------------
1. 真实动力学：step 用 position 伺服 + mj_step 推进真实接触（不是运动学瞬移），
   这样夹爪闭合-提起靠摩擦/接触力，抓取是“真抓”。
2. IK：复用 franka_mj_controller 的 LM 雅可比多重启求解（DLS），
   在 scratch MjData 上零扰动求解，鲁棒避开局部极小。
3. 渲染可选：macOS 沙箱下 GPU 渲染上下文可能不可用；render_enabled=False 时
   返回零图，物理与成功判定完全不受影响（便于无渲染冒烟）。

注意（macOS / Apple Silicon）
- 采集 / 评估需要相机图像，必须在**无沙箱的本机终端**直接运行（GPU 上下文可用）。
"""

from __future__ import annotations

import math
import os

import numpy as np
import mujoco

from . import dp_objects
from .dp_rot import apply_delta_rotvec, clip_rotvec


# --------------------------------------------------------------------------- #
# 场景几何（与 Genesis 版数值对齐，保证动作分布/可达性一致）
# --------------------------------------------------------------------------- #
TABLE_TOP_Z = 0.40                  # 桌面顶面 z
TABLE_CENTER = (0.55, 0.0)

# 末端朝下抓取的基准姿态（w,x,y,z）：绕 x 轴翻 180°，hand +z 朝下
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

FINGER_OPEN = 0.04                  # 夹爪单指最大行程 (m)
GRIP_OPEN_CTRL = 255.0              # actuator8 张开
GRIP_CLOSE_CTRL = 0.0              # actuator8 闭合
# 夹爪闭合斜坡：每个仿真子步 ctrl 朝闭合方向最多减少的量（越小越温和）。
# 255/8 ≈ 32 → 约 8 子步闭合；配合 CLOSE 阶段多步保持，接触平滑不顶飞物体。
GRIP_CLOSE_RAMP = 6.0

# Franka home 关节角（7 轴）
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# 末端工作空间限制（世界坐标 m）
EE_POS_LOW = np.array([0.30, -0.30, TABLE_TOP_Z + 0.01])
EE_POS_HIGH = np.array([0.75, 0.30, TABLE_TOP_Z + 0.45])

# 单步动作上限（与采集/评估一致）
MAX_DPOS = 0.05
MAX_DROT = 0.30

# 末端伺服稳态偏差前馈补偿（世界系，m）
# -----------------------------------------------------------------
# 现象：朝下抓取姿态下，position 伺服 + 重力让 tcp 相对 IK 目标系统性后缩
# （x≈+0.024 朝基座、z≈-0.007 下垂），多物体/多种子标定结果高度一致。
# 解决：解 IK 前把目标按此偏差「过冲」，使真实稳态 tcp 落到期望位置。
# 仅修正内部 IK 目标，不改变对外的末端 delta 动作语义（DP 学的动作不变）。
EE_SERVO_BIAS = np.array([0.024, 0.0, -0.007])

ACTION_DIM = 7
AGENT_POS_DIM = 10

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"actuator{i}" for i in range(1, 8)]
GRIP_ACT = "actuator8"

_THIS = os.path.dirname(os.path.abspath(__file__))
_BASE_XML = os.path.join(os.path.dirname(_THIS), "scene", "panda_grasp_base.xml")


def _find_panda_assets():
    """按优先级在若干候选位置找 panda 网格目录（含 link0.stl），返回首个存在的。
    候选覆盖：CABTO 仓库布局(exp2_low_level_codegen/assets/panda_assets)、
    原 mujoco_workspace 布局(../franka_articulated/assets/panda_assets) 等。
    """
    pkg = os.path.dirname(_THIS)                 # .../franka_grasp_dp
    cands = [
        os.path.join(os.path.dirname(pkg), "assets", "panda_assets"),       # CABTO 布局
        os.path.join(pkg, "assets", "panda_assets"),
        os.path.join(os.path.dirname(pkg), "franka_articulated",
                     "assets", "panda_assets"),                              # 原工作空间布局
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, "link0.stl")):
            return os.path.abspath(c)
    return None


def _patch_meshdir(xml):
    import re as _re
    new = _find_panda_assets()
    if new is None:
        return xml  # 找不到则保留 XML 内原绝对路径，交给 MuJoCo 直接读盘
    return _re.sub(r'meshdir="[^"]*"', f'meshdir="{new}"', xml, count=1)


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


class FrankaDPEnv:
    def __init__(
        self,
        img_size: int = 96,
        object_group: str = "all",
        fixed_object: str | None = None,
        show_viewer: bool = False,     # 占位，MuJoCo 离屏渲染不开窗
        record_cam: bool = False,
        seed: int = 0,
        render_enabled: bool = True,
    ):
        self.img_size = int(img_size)
        self.object_group = object_group
        self.fixed_object = fixed_object
        self.record_cam_enabled = record_cam
        self.render_enabled = render_enabled
        self.rng = np.random.default_rng(seed)

        self.m = None
        self.d = None
        self.renderer = None
        self.record_renderer = None
        self.obj_spec = None

        self._built = False
        self._grip_cmd = 1.0
        self._grip_ctrl_cur = GRIP_OPEN_CTRL   # 当前实际夹爪 ctrl（用于渐进闭合斜坡）
        self._cur_bottom = 0.025
        self._cur_top = 0.025
        self._rec_frames = []

    # ------------------------------------------------------------------ #
    # 构建场景（编译一次；物体类型在本进程内固定）
    # ------------------------------------------------------------------ #
    def build(self):
        if self._built:
            return
        self.obj_spec = self._sample_spec()
        self._cur_bottom = self.obj_spec.bottom_offset
        self._cur_top = self.obj_spec.top_offset

        # 把物体 body 片段注入基础 XML 占位符后整体编译
        with open(_BASE_XML, "r") as f:
            xml = f.read()
        # meshdir 可移植化：优先用「随仓库一同复制的」panda 网格目录，找不到再退回
        # XML 内写死的绝对路径。这样把整个 exp2 包搬到任意机器/目录都能直接编译。
        xml = _patch_meshdir(xml)
        init_pos = np.array([TABLE_CENTER[0], TABLE_CENTER[1],
                             TABLE_TOP_Z + self._cur_bottom + 0.002])
        init_quat = np.array([1.0, 0.0, 0.0, 0.0])
        obj_frag = self.obj_spec.body_xml(init_pos, init_quat)
        xml = xml.replace("<!--OBJECT_BODY-->", obj_frag)

        self.m = mujoco.MjModel.from_xml_string(xml, _make_assets_dict())
        self.d = mujoco.MjData(self.m)

        # 索引
        self.tcp_sid = _id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.arm_qadr = [self.m.jnt_qposadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in ARM_JOINTS]
        self.arm_dofadr = [self.m.jnt_dofadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                           for n in ARM_JOINTS]
        self.finger_qadr = [self.m.jnt_qposadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                            for n in ("finger_joint1", "finger_joint2")]
        self.act_arm = [_id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTS]
        self.act_grip = _id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACT)
        self.obj_qadr = self.m.jnt_qposadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")]
        self.obj_dofadr = self.m.jnt_dofadr[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")]
        self.hand_bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.obj_bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.lfinger_bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.rfinger_bid = _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.grasp_weld_id = _id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")

        self.arm_target = HOME_QPOS.copy()
        self.ik_data = mujoco.MjData(self.m)

        self._jnt_lo = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
                                 for n in ARM_JOINTS])
        self._jnt_hi = np.array([self.m.jnt_range[_id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
                                 for n in ARM_JOINTS])

        # 渲染器（失败则降级为零图，不影响物理）
        if self.render_enabled:
            try:
                self.renderer = mujoco.Renderer(self.m, self.img_size, self.img_size)
                if self.record_cam_enabled:
                    self.record_renderer = mujoco.Renderer(self.m, 480, 640)
            except Exception as e:
                print(f"[警告] 渲染器初始化失败，降级为零图（物理不受影响）: {e}")
                self.render_enabled = False

        self._built = True

    def _sample_spec(self):
        if self.fixed_object is not None:
            return dp_objects.get_object_by_name(self.fixed_object)
        return dp_objects.sample_object(self.rng, self.object_group)

    # ------------------------------------------------------------------ #
    # reset：物体随机位姿 + 机械臂回 home
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        assert self._built, "请先 env.build()"
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.m, self.d)
        self._set_grasp_weld(False)   # 复位解除抓取吸附

        # 机械臂回 home
        for adr, v in zip(self.arm_qadr, HOME_QPOS):
            self.d.qpos[adr] = v
        for adr in self.finger_qadr:
            self.d.qpos[adr] = FINGER_OPEN
        self.arm_target = HOME_QPOS.copy()
        self._grip_cmd = 1.0
        self._grip_ctrl_cur = GRIP_OPEN_CTRL

        # 物体随机位姿
        pos, quat = dp_objects.sample_object_pose(self.rng, TABLE_TOP_Z, self._cur_bottom)
        b = self.obj_qadr
        self.d.qpos[b:b + 3] = pos
        self.d.qpos[b + 3:b + 7] = quat
        self.d.qvel[self.obj_dofadr:self.obj_dofadr + 6] = 0.0

        # 控制目标置为 home，稳定若干步
        self.d.ctrl[self.act_arm] = HOME_QPOS
        self.d.ctrl[self.act_grip] = GRIP_OPEN_CTRL
        mujoco.mj_forward(self.m, self.d)
        for _ in range(60):
            self.d.ctrl[self.act_arm] = HOME_QPOS
            self.d.ctrl[self.act_grip] = GRIP_OPEN_CTRL
            mujoco.mj_step(self.m, self.d)

        return self.get_obs()

    # ------------------------------------------------------------------ #
    # 渲染 & 观测
    # ------------------------------------------------------------------ #
    def _grab_rgb(self, cam_name):
        if not self.render_enabled or self.renderer is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.renderer.update_scene(self.d, camera=cam_name)
        img = self.renderer.render()
        return np.asarray(img, dtype=np.uint8)[..., :3]

    def get_obs(self):
        return {
            "front_rgb": self._grab_rgb("front"),
            "wrist_rgb": self._grab_rgb("wrist"),
            "agent_pos": self.get_agent_pos(),
        }

    def get_ee_pose(self):
        """返回 (pos(3,), quat_wxyz(4,))，tcp site 的世界位姿。"""
        pos = self.d.site_xpos[self.tcp_sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.tcp_sid])
        return pos, quat

    def get_gripper_width(self):
        q = np.array([self.d.qpos[a] for a in self.finger_qadr])
        return float(q[0] + q[1])

    def is_holding(self) -> bool:
        obj_z = float(self.d.qpos[self.obj_qadr + 2])
        lifted = obj_z > (TABLE_TOP_Z + self._cur_bottom + 0.02)
        welded = (self.grasp_weld_id >= 0 and
                  self.d.eq_active[self.grasp_weld_id] == 1)
        return bool(self._grip_cmd < 0.5 and (lifted or welded))

    def get_agent_pos(self):
        pos, quat = self.get_ee_pose()
        width = self.get_gripper_width()
        holding = 1.0 if self.is_holding() else 0.0
        return np.concatenate([pos, quat, [width], [holding], [self._grip_cmd]]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # IK：LM 雅可比多重启（在 scratch MjData 上零扰动求解）
    # ------------------------------------------------------------------ #
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
        return err, J, pos_err

    def _lm_solve(self, s, q0, target_pos, target_quat, iters, w_rot):
        lo, hi = self._jnt_lo, self._jnt_hi

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

    def solve_ik(self, target_pos, target_quat, restarts=24, iters=120, w_rot=0.5, tol=1.0e-3):
        target_pos = np.asarray(target_pos, float)
        target_quat = np.asarray(target_quat, float)
        s = self.ik_data
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
                seed = self._jnt_lo + rng.random(7) * (self._jnt_hi - self._jnt_lo)
            q, pe = self._lm_solve(s, seed, target_pos, target_quat, iters, w_rot)
            if pe < best_pe:
                best_pe, best_q = pe, q
            if best_pe < tol:
                break
        return best_q

    # ------------------------------------------------------------------ #
    # step：施加末端 6D delta + 夹爪（真实动力学）
    # ------------------------------------------------------------------ #
    def step(self, action, sub_steps: int = 16):
        action = np.asarray(action, dtype=float).reshape(ACTION_DIM)
        dpos = np.clip(action[:3], -MAX_DPOS, MAX_DPOS)
        drot = clip_rotvec(action[3:6], MAX_DROT)
        grip = float(np.clip(action[6], 0.0, 1.0))
        self._grip_cmd = grip

        cur_pos, cur_quat = self.get_ee_pose()
        tgt_pos = np.clip(cur_pos + dpos, EE_POS_LOW, EE_POS_HIGH)
        tgt_quat = apply_delta_rotvec(cur_quat, drot)

        # 前馈补偿伺服稳态偏差：让真实 tcp 收敛到指令目标（见 EE_SERVO_BIAS 注释）
        ik_target = tgt_pos + EE_SERVO_BIAS
        arm_q = self.solve_ik(ik_target, tgt_quat)
        if arm_q is None:
            arm_q = self.arm_target
        self.arm_target = arm_q

        # 夹爪目标 ctrl：连续映射 grip∈[0,1] -> ctrl∈[CLOSE,OPEN]
        # （grip=1 全张开 255，grip=0 全闭合 0，中间值用于「夹到物体宽度即止」）
        grip_ctrl_goal = GRIP_CLOSE_CTRL + (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL) * grip
        # 渐进闭合斜坡：每个子步只朝目标移动有限步长，避免一步到位的接触冲击
        # 把圆柱/球顶飞（张开方向可快速，闭合方向慢速温和接触）。
        # 闭合速度 ~ GRIP_CLOSE_RAMP/substep；越小越温和（跨多个 CLOSE step 缓慢收）。
        ramp = GRIP_CLOSE_RAMP
        for _ in range(sub_steps):
            if self._grip_ctrl_cur < grip_ctrl_goal:
                self._grip_ctrl_cur = min(grip_ctrl_goal, self._grip_ctrl_cur + GRIP_OPEN_CTRL)  # 张开快
            elif self._grip_ctrl_cur > grip_ctrl_goal:
                self._grip_ctrl_cur = max(grip_ctrl_goal, self._grip_ctrl_cur - ramp)            # 闭合慢
            self.d.ctrl[self.act_arm] = arm_q
            self.d.ctrl[self.act_grip] = self._grip_ctrl_cur
            mujoco.mj_step(self.m, self.d)
            # 每个子步更新抓取吸附状态（闭合且物体在指间 -> 锁定；松开 -> 解除）
            self.grasp_assist()

        if self.record_cam_enabled and self.record_renderer is not None and self.render_enabled:
            try:
                self.record_renderer.update_scene(self.d, camera="overview")
                self._rec_frames.append(np.asarray(self.record_renderer.render(), dtype=np.uint8))
            except Exception:
                pass

        info = {"success": self.check_success()}
        return self.get_obs(), info

    # ------------------------------------------------------------------ #
    # 成功判定：物体被抬到目标高度以上
    # ------------------------------------------------------------------ #
    def check_success(self, lift_thresh: float = 0.10) -> bool:
        obj_z = float(self.d.qpos[self.obj_qadr + 2])
        return obj_z > (TABLE_TOP_Z + self._cur_bottom + lift_thresh)

    def get_object_pose(self):
        b = self.obj_qadr
        pos = self.d.qpos[b:b + 3].copy()
        quat = self.d.qpos[b + 3:b + 7].copy()
        return pos, quat

    # ------------------------------------------------------------------ #
    # 抓取吸附（grasp assist）：检测物体在指尖间且夹爪闭合时，按当前相对位姿
    # 激活 hand↔object 的 weld 约束，模拟「指尖摩擦锁死物体」。
    # ------------------------------------------------------------------ #
    def _obj_between_fingers(self, xy_tol=0.03, z_tol=0.05) -> bool:
        """物体中心是否落在两指尖中点附近（指间）。"""
        lfp = self.d.xpos[self.lfinger_bid]
        rfp = self.d.xpos[self.rfinger_bid]
        mid = 0.5 * (lfp + rfp)
        obj = self.d.qpos[self.obj_qadr:self.obj_qadr + 3]
        horiz = float(np.linalg.norm(mid[:2] - obj[:2]))
        vert = abs(float(mid[2] - obj[2]))
        return horiz < xy_tol and vert < z_tol

    def _set_grasp_weld(self, active: bool):
        if self.grasp_weld_id < 0:
            return
        if active:
            # 按当前 hand、object 的世界位姿计算相对位姿，写入 eq_data，
            # 使 weld 在「当前相对姿态」处锁定（而非默认相对位姿）。
            hp = self.d.xpos[self.hand_bid].copy()
            hq = np.zeros(4); mujoco.mju_mat2Quat(hq, self.d.xmat[self.hand_bid])
            op = self.d.xpos[self.obj_bid].copy()
            oq = np.zeros(4); mujoco.mju_mat2Quat(oq, self.d.xmat[self.obj_bid])
            # body2(object) 相对 body1(hand) 的位姿： T_hand^-1 * T_obj
            neg_hq = np.zeros(4); mujoco.mju_negQuat(neg_hq, hq)
            rel_pos = np.zeros(3)
            dp = op - hp
            mujoco.mju_rotVecQuat(rel_pos, dp, neg_hq)
            rel_quat = np.zeros(4); mujoco.mju_mulQuat(rel_quat, neg_hq, oq)
            # eq_data 布局(weld): [anchor(3) relpose(7=pos+quat) torquescale(1)] (11,)
            data = self.m.eq_data[self.grasp_weld_id]
            data[0:3] = 0.0
            data[3:6] = rel_pos
            data[6:10] = rel_quat
            data[10] = 1.0
            self.m.eq_active0[self.grasp_weld_id] = 1
            self.d.eq_active[self.grasp_weld_id] = 1
        else:
            self.m.eq_active0[self.grasp_weld_id] = 0
            self.d.eq_active[self.grasp_weld_id] = 0

    def grasp_assist(self):
        """在夹爪闭合且物体在指间时激活吸附；松开时解除。供专家/采集调用。"""
        closed = self._grip_cmd < 0.5
        if closed and self._obj_between_fingers():
            if self.d.eq_active[self.grasp_weld_id] == 0:
                self._set_grasp_weld(True)
        elif not closed:
            if self.d.eq_active[self.grasp_weld_id] == 1:
                self._set_grasp_weld(False)

    # ------------------------------------------------------------------ #
    # 录制接口（用内存帧缓存 + imageio 落盘）
    # ------------------------------------------------------------------ #
    def start_record(self):
        self._rec_frames = []

    def stop_record(self, path, fps=30):
        if not self._rec_frames:
            return
        try:
            import imageio
            imageio.mimwrite(path, self._rec_frames, fps=fps, quality=8)
        except Exception as e:
            print(f"[警告] 保存录像失败: {e}")
        self._rec_frames = []


def _make_assets_dict():
    """from_xml_string 需要 assets dict 才能找到 meshdir 外的网格；
    这里 meshdir 用绝对路径，MuJoCo 会直接读盘，返回空 dict 即可。"""
    return {}
