"""脚本化专家策略（在 delta 动作空间内工作）
=============================================
为了让采集到的「末端 6D delta」动作分布与策略部署时一致，
专家**不直接瞬移**，而是每一步输出一个受限的 delta 动作，由环境的 IK 执行。

阶段（有限状态机）：
    APPROACH  : 移动到物体正上方（保持预抓取高度），夹爪张开
    DESCEND   : 垂直下探到抓取高度（指尖落到物体腰部，对准物体中心）
    CLOSE     : 闭合夹爪（原地保持，等接触/约束稳定）
    LIFT      : 锁定水平、垂直抬起到目标高度
    DONE      : 完成

每一步 delta：
    dpos = clip(target_pos - cur_pos, ±MAX_DPOS)
    drot = clip(rotvec(target_quat ⊖ cur_quat), ±MAX_DROT)
    grip = 1(张开) / 0(闭合)

几何要点（关键）
----------------
- env.get_ee_pose() 返回 **tcp site**（hand 坐标系 z=0.103，即两指指尖处）。
  因此末端目标 z 直接就是「指尖期望高度」，无需再补 hand→指尖 偏移。
- env.step 内部已对 IK 目标做伺服稳态偏差前馈补偿（EE_SERVO_BIAS），
  所以 tcp 能精确收敛到指令位置，专家可以信任「指令=实际」。
- 抓取高度取物体**几何中心附近**（腰部），保证指尖跨在物体两侧而非顶面以上。
- 各阶段切换阈值按真实可达精度（~0.01）设定，避免过严导致卡死。
"""

from __future__ import annotations

import numpy as np

from .dp_rot import delta_rotvec_between, clip_rotvec
from . import franka_dp_env as E


class ScriptedExpert:
    def __init__(self, env: E.FrankaDPEnv, rng: np.random.Generator | None = None):
        self.env = env
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.phase = "APPROACH"
        self._approach_counter = 0
        self._close_counter = 0
        self._lift_counter = 0
        self._descend_counter = 0
        self._align_counter = 0
        self._grasp_quat = E.GRASP_QUAT.copy()
        self._pre_extra = 0.12        # 预抓取在抓取高度之上再抬的量
        self._lock_xy = None          # LIFT 阶段锁定的水平目标

    def _target_quat(self):
        # 保持末端朝下；几何基元对称，不强行对齐 yaw，IK 更稳。
        return self._grasp_quat

    def _grasp_z(self):
        """抓取时指尖目标高度：物体几何中心（腰部）。

        指尖跨在物体两侧、中心略低于物体顶面，保证闭合时夹住物体侧壁。
        """
        env = self.env
        obj_pos, _ = env.get_object_pose()
        obj_top_z = obj_pos[2] + env._cur_top
        obj_bottom_z = obj_pos[2] - env._cur_bottom
        # 指尖落到「顶面以下 0.6×物高」≈腰部偏下，但不低于底面以上 0.6cm 防撞桌
        grasp_z = obj_top_z - 0.6 * (env._cur_top + env._cur_bottom)
        grasp_z = max(grasp_z, obj_bottom_z + 0.006)
        return grasp_z

    def act(self):
        """返回 (action(7,), done(bool))。"""
        env = self.env
        ee_pos, ee_quat = env.get_ee_pose()
        obj_pos, _ = env.get_object_pose()

        grasp_z = self._grasp_z()
        pre_z = grasp_z + self._pre_extra
        lift_z = E.TABLE_TOP_Z + 0.30   # 抬到桌面上方 30cm（远超成功阈值 10cm）

        tgt_quat = self._target_quat()
        drot = clip_rotvec(delta_rotvec_between(ee_quat, tgt_quat), E.MAX_DROT)

        grip = 1.0
        done = False

        if self.phase == "APPROACH":
            target = np.array([obj_pos[0], obj_pos[1], pre_z])
            dpos = target - ee_pos
            horiz_err = np.linalg.norm(dpos[:2])
            self._approach_counter += 1
            # 水平对准即可下探（精对准留给 DESCEND/ALIGN）；阈值放宽匹配真实精度，
            # 并加超时保护避免个别落点因 IK 残差卡死在 APPROACH。
            if horiz_err < 0.02 or self._approach_counter > 30:
                self.phase = "DESCEND"
            grip = 1.0

        elif self.phase == "DESCEND":
            # 持续用最新物体水平位置 + 抓取高度作目标，边下边微调对准
            target = np.array([obj_pos[0], obj_pos[1], grasp_z])
            dpos = target - ee_pos
            horiz_err = np.linalg.norm(dpos[:2])
            z_err = abs(dpos[2])
            self._descend_counter += 1
            # 到达抓取高度且大致对准 -> 进入精对准；超时也强制进入（避免卡死）
            if (z_err < 0.012 and horiz_err < 0.030) or self._descend_counter > 60:
                self.phase = "ALIGN"
                self._align_counter = 0
            grip = 1.0

        elif self.phase == "ALIGN":
            # 抓取高度上纯水平精对准：闭合前把指尖中心压到物体中心，
            # 保证两指对称接触、不把自由物体顶飞。
            target = np.array([obj_pos[0], obj_pos[1], grasp_z])
            dpos = target - ee_pos
            dpos[2] = np.clip(dpos[2], -0.006, 0.006)  # z 基本保持在抓取面
            horiz_err = np.linalg.norm(dpos[:2])
            self._align_counter += 1
            grip = 1.0
            if horiz_err < 0.006 or self._align_counter >= 20:
                self.phase = "CLOSE"
                self._close_counter = 0

        elif self.phase == "CLOSE":
            # 慢闭合 + tcp 强跟随物体中心：单边接触会推动自由物体，跟随让指尖
            # 始终回到物体中心，配合 env 内慢速闭合斜坡，使两指最终对称夹住。
            target = np.array([obj_pos[0], obj_pos[1], grasp_z])
            dpos = target - ee_pos          # 全增益跟随（不缩放）
            drot = np.zeros(3)
            grip = 0.0
            self._close_counter += 1
            if self._close_counter >= 22:   # 慢闭合需要更多步达到稳定夹持
                self.phase = "LIFT"
                self._lift_counter = 0
                self._lock_xy = ee_pos[:2].copy()   # 锁定当前水平位置，抬升不漂移

        elif self.phase == "LIFT":
            lx, ly = self._lock_xy
            target = np.array([lx, ly, lift_z])
            dpos = target - ee_pos
            grip = 0.0
            self._lift_counter += 1
            if ee_pos[2] >= lift_z - 0.02 or self._lift_counter >= 40:
                self.phase = "DONE"

        else:  # DONE
            dpos = np.zeros(3)
            drot = np.zeros(3)
            grip = 0.0
            done = True

        dpos = np.clip(dpos, -E.MAX_DPOS, E.MAX_DPOS)
        action = np.concatenate([dpos, drot, [grip]]).astype(np.float32)
        return action, done
