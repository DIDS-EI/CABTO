"""机械臂原语接口（primitives）
================================

实验2「代码生成串接原语」范式的执行底座。每个原语是一个**可被代码生成器
调用的最小动作单元**，内部复用 FrankaDPEnv 已调通的 IK + 伺服偏差补偿 +
grasp-assist 抓取链路。原语对外只暴露简单语义（移动到某点 / 开合夹爪 / 抬起），
代码生成器把它们串成一个 action（如 pick(obj)）。

设计要点
--------
- 所有原语都走 env.step(action)（末端 6D delta + grip），因此和 DP 学到的动作
  空间完全一致——代码生成出来的轨迹本身就是合法的 DP 训练/评估轨迹。
- move_to 是闭环伺服：反复发 delta 把 tcp 推向目标，直到收敛或超时。
- 每个原语返回执行轨迹片段（obs/action 序列），便于录像与符号状态记录。
"""

from __future__ import annotations

import numpy as np

from dp_common.franka_dp_env import (
    FrankaDPEnv, TABLE_TOP_Z, MAX_DPOS,
)


class PrimitiveRunner:
    """把原语调用翻译成 env.step 的执行器，并记录轨迹。"""

    def __init__(self, env: FrankaDPEnv, log=None):
        self.env = env
        self.log = log if log is not None else []
        self.traj = []          # [(obs, action), ...] 供录像/数据
        self._last_obs = None

    # ----- 底层：发一个末端 delta 动作 ----- #
    def _apply(self, dpos, grip, drot=(0, 0, 0)):
        action = np.zeros(7)
        action[:3] = np.clip(dpos, -MAX_DPOS, MAX_DPOS)
        action[3:6] = drot
        action[6] = grip
        obs, info = self.env.step(action)
        self._last_obs = obs
        self.traj.append((obs, action.copy()))
        return obs, info

    def _ee(self):
        return self.env.get_ee_pose()[0]

    # ================= 对外原语 ================= #
    def move_to(self, target_xyz, grip=1.0, tol=0.01, max_steps=80, name="move_to", ki=0.0):
        """闭环把 tcp 移动到 target_xyz（保持朝下抓取姿态）。

        env.step 内部对 IK 目标加了固定 EE_SERVO_BIAS（为开环单步 delta 标定的
        稳态补偿）；在多步闭环里该固定偏置会变成恒定稳态误差源，使 tcp 在工作空间
        边缘点系统性偏 ~2cm 收敛不到。可选**积分项 (PI 控制, ki>0)** 累加残余误差，
        闭环自适应消除任意恒定偏置。仅在贴近物体的精对准阶段（descend）开启，
        approach 等粗移动用纯比例 (ki=0) 避免超调把物体碰倒。
        """
        target = np.asarray(target_xyz, float)
        integ = np.zeros(3)
        for i in range(max_steps):
            cur = self._ee()
            err = target - cur
            if np.linalg.norm(err) < tol:
                break
            if ki > 0:
                integ += err
                cmd = err + ki * integ
            else:
                cmd = err
            self._apply(cmd, grip)
        cur = self._ee()
        final_err = float(np.linalg.norm(target - cur))
        self.log.append({"prim": name, "target": target.tolist(),
                         "final": cur.tolist(), "err": final_err,
                         "steps": i + 1})
        return final_err

    def approach(self, target_xyz, height=0.12, grip=1.0):
        """移动到目标正上方 height 处（接近位姿，夹爪张开）。"""
        above = np.array(target_xyz, float).copy()
        above[2] = TABLE_TOP_Z + height
        return self.move_to(above, grip=grip, name="approach")

    def descend_to(self, target_xyz, grip=1.0, tol=0.008):
        """下探到抓取高度（夹爪保持张开）。

        两段式：(1) 先在当前（较高）高度用积分项把 xy 精确对准目标正上方，
        消除工作空间边缘的稳态偏差；(2) 再竖直缓慢下探到抓取高度。
        这样圆柱/球体下探时不会被斜向接近碰倒。
        """
        target = np.asarray(target_xyz, float)
        cur = self._ee()
        # (1) 高处精对准 xy（带积分），z 保持当前高度
        above = np.array([target[0], target[1], cur[2]], float)
        self.move_to(above, grip=grip, tol=0.006, max_steps=40,
                     name="descend_align", ki=0.6)
        # (2) 竖直下探到抓取高度（带积分保持 xy 不漂）
        return self.move_to(target, grip=grip, tol=tol, max_steps=50,
                            name="descend", ki=0.6)

    def align_xy(self, target_xyz, grip=1.0, horiz_tol=0.006, max_steps=25):
        """抓取高度上的纯水平精对准：把指尖中心压到目标 xy（z 基本保持）。
        闭合前调用，保证两指对称接触、不把自由物体顶飞。
        servo 目标是「给定的 target_xyz」（来自 VLM 反投影），不偷看物体真值。
        """
        target = np.asarray(target_xyz, float)
        integ = np.zeros(3)
        KI = 0.6
        for i in range(max_steps):
            cur = self._ee()
            err = target - cur
            integ += err
            dpos = err + KI * integ
            dpos[2] = np.clip(dpos[2], -0.006, 0.006)
            if np.linalg.norm((target - cur)[:2]) < horiz_tol:
                break
            self._apply(dpos, grip)
        cur = self._ee()
        self.log.append({"prim": "align_xy", "horiz_err": float(np.linalg.norm((target - cur)[:2])),
                         "steps": i + 1})

    def close_gripper_following(self, target_xyz, hold=22, grip=0.0):
        """慢闭合 + tcp 跟随给定 xy 目标：单边接触会推动自由物体，持续把指尖
        拉回目标 xy，配合 env 内慢闭合斜坡，使两指最终对称夹住。
        """
        target = np.asarray(target_xyz, float)
        integ = np.zeros(3)
        KI = 0.5
        for _ in range(hold):
            cur = self._ee()
            err = target - cur
            integ += err
            dpos = err + KI * integ
            dpos[2] = np.clip(dpos[2], -0.004, 0.004)
            self._apply(dpos, grip)
        self.log.append({"prim": "close_gripper_following",
                         "holding": bool(self.env.is_holding())})

    def open_gripper(self, hold=4):
        for _ in range(hold):
            self._apply((0, 0, 0), grip=1.0)
        self.log.append({"prim": "open_gripper"})

    def close_gripper(self, hold=10):
        """渐进闭合（env 内部有闭合斜坡 + grasp-assist 吸附）。"""
        for _ in range(hold):
            self._apply((0, 0, 0), grip=0.0)
        self.log.append({"prim": "close_gripper",
                         "holding": bool(self.env.is_holding())})

    def lift(self, height=0.30, grip=0.0):
        """抓住后竖直抬起到 TABLE_TOP_Z + height。"""
        cur = self._ee()
        target = cur.copy()
        target[2] = TABLE_TOP_Z + height
        return self.move_to(target, grip=grip, tol=0.02, max_steps=60, name="lift")
