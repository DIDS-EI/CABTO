"""无把手茶盒的细粒度交接 API，保留接触门控 weld 显式辅助。

只读目标返回独立的纯数据；控制接口不暴露环境、状态写入口或抓放大宏。
本模块不改变场景、接触门限或物体 qpos。factory 默认关闭渲染。
"""
from math import isfinite

import numpy as np

from core.handleless_tea_handover import HandlelessTeaRuntime, box_domain


_GRASP_DOCS = {
    "grasp_pose": "grasp_pose()：只读当前指定盒面抓位的 TCP 世界坐标，返回有限 xyz 元组。",
    "approach": "approach(p)：张爪移动到抓位上方的配置净空；p 仅接受本次 API 的 grasp_pose 读数。返回实测数据。",
    "descend": "descend(p)：从接近位张爪下降到已读取的抓位，不闭爪、不附着；返回实测数据。",
    "close_gripper": "close_gripper()：仅执行闭爪控制并记录接触窗口起点，不激活 weld；返回实测数据。",
    "attach_contact_gated": "attach_contact_gated()：检查闭爪、当帧及最近 0.10 秒双指接触和碰撞审计后，显式激活 assisted_grasp_weld；不执行闭爪过程。返回实测数据。",
}
DOCS = {
    "donor_pick": {
        **_GRASP_DOCS,
        "lift": "lift()：仅将已附着的盒体原点抬到桌面高度加 0.20 米，保持当前 xy；返回实测数据。",
    },
    "present": {
        "target_pose": "target_pose()：只读配置的空中交接目标，返回有限 xyz 元组。",
        "move_held": "move_held(p)：仅移动交出臂已持有的盒体；p 必须来自本次 API 的 target_pose。返回实测数据。",
        "settle": "settle()：交出臂保持闭爪、接收臂保持张爪，执行真实保持控制；返回实测数据。",
    },
    "receive": {
        **_GRASP_DOCS,
        "attach_contact_gated": "attach_contact_gated()：按当帧及最近 0.10 秒双指接触门控显式激活接收臂 weld；实测双侧持有成立才记录 dual_hold_verified，不抬升。返回实测数据。",
    },
    "donor_release": {
        "open_gripper": "open_gripper()：已验证双侧持有时仅张开交出臂夹爪，保持接收臂闭爪；不解除 weld。返回实测数据。",
        "detach": "detach()：张爪控制完成且接收臂仍持有时，显式解除交出臂 weld 并记录释放事件；不撤离。返回实测数据。",
        "withdraw": "withdraw()：已解除附着的交出臂移动到配置安全位，实测到位且接收臂仍持有才记录撤离事件。返回实测数据。",
    },
    "receiver_place": {
        "target_pose": "target_pose()：只读终态盒体原点目标，返回有限 xyz 元组；目标 z 为桌面加盒体半高加 0.003 米。",
        "read": "read()：只读盒体放置几何、桌面支撑接触及释放状态，返回独立字典，不推进仿真。",
        "carry": "carry(p)：仅把接收臂持有的盒体移到目标 xy、桌面加 0.24 米的运输位；p 必须来自本次 API 的 target_pose。返回实测数据。",
        "lower": "lower(p)：仅执行桌面加 0.20 米对齐及原 tracked 下降，不张爪或解除附着；p 必须来自本次 API 的 target_pose。返回实测数据。",
        "open_gripper": "open_gripper()：下降完成后仅执行接收臂张爪控制，不解除 weld；返回实测数据。",
        "detach": "detach()：显式解除接收臂 weld 并执行原释放后保持控制，不退手；返回实测数据。",
        "retreat_vertical": "retreat_vertical()：已解除附着的接收臂从当前 TCP 位竖直上移 0.10 米，不横向撤离；返回实测数据。",
        "withdraw_safe": "withdraw_safe()：竖直退手后的空接收臂移动到配置安全位；实测到位且盒体落稳才记录撤离事件。返回实测数据。",
    },
    "controlled_home": {
        "return_arm_home": "return_arm_home()：仅对无活动 weld 的绑定空臂执行安全位及原执行器 HOME 控制，保留实测 HOME 验收。返回纯数据。",
    },
}

_PHASES = {
    "donor_pick": "pick_donor",
    "present": "present_handover",
    "receive": "receive_baton",
    "donor_release": "donor_release",
    "receiver_place": "receiver_place",
}


def _xyz(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("目标必须是三个有限坐标")
    return result.copy()


class FineHandoverRuntime(HandlelessTeaRuntime):
    def api(self, model):
        kind = model["program_kind"]
        if kind not in DOCS:
            raise ValueError(f"未知交接 API 类型：{kind}")
        args = model["args"]
        raw_arm = args["arm"]
        if type(raw_arm) not in (int, str) or str(raw_arm) not in ("0", "1"):
            raise ValueError("arm 只能是 0 或 1")
        i = int(raw_arm)
        if args.get("obj") != "baton":
            raise ValueError("交接 API 只支持 baton")
        expected_arm = 1 if kind in ("receive", "receiver_place") else 0
        if kind != "controlled_home" and i != expected_arm:
            raise ValueError("program_kind 与绑定机械臂不匹配")
        arm = self.env.arms[i]
        issued = set()
        done = set()
        close_since = None

        def begin():
            if any(a._grasp_enabled for a in self.env.arms):
                raise RuntimeError("细粒度 API 要求自动抓取辅助关闭，weld 仅由显式接口控制")
            if not self.collision_free():
                raise RuntimeError(f"碰撞审计未通过：{self.audit.summary()}")
            self.audit.phase = (_PHASES[kind] if kind != "controlled_home" else
                                ("home_donor_arm" if i == 0 else "home_receiver_arm"))
            if kind != "controlled_home":
                self.home_dirty[i] = True

        def need(step):
            if step not in done:
                raise RuntimeError(f"当前 API 缺少已完成的 {step} 控制证据")

        def observe():
            return {"time": float(self.env.d.time),
                    "object_position": self.pos().tolist(),
                    "tcp_position": arm.get_ee_pose()[0].tolist(),
                    "weld_constraint_active": bool(self.weld_constraint_active(i)),
                    "holding": bool(self.weld_active(i)),
                    "grip_command": float(arm._grip_cmd),
                    "collision_free": bool(self.collision_free()),
                    "home_dirty": bool(self.home_dirty[i]),
                    "mode": "assisted_grasp_weld"}

        def issue(value):
            p = tuple(float(v) for v in _xyz(value))
            issued.add(p)
            return p

        def target(value):
            if type(value) not in (tuple, list) or len(value) != 3:
                raise ValueError("参数必须是本次只读目标接口返回的 xyz")
            if any(type(v) not in (int, float) or not isfinite(v) for v in value):
                raise ValueError("目标坐标必须是有限数值，不接受布尔值")
            if tuple(value) not in issued:
                raise ValueError("目标未经本次只读接口签发，或已被修改")
            return _xyz(value)

        def grasp_ready():
            if self.weld_constraint_active(i):
                raise RuntimeError("抓取控制要求绑定臂尚未附着")
            if i == 1 and not self.weld_active(0):
                raise RuntimeError("接收动作要求交出臂仍持有盒体")
            if i == 0 and self.weld_constraint_active(1):
                raise RuntimeError("交出臂抓取要求接收臂为空")

        def held():
            if not self.weld_active(i) or self.weld_constraint_active(1 - i):
                raise RuntimeError("运输要求绑定臂单独持有盒体")

        def receiver_holds():
            if not self.weld_active(1):
                raise RuntimeError("释放或撤离期间接收臂必须仍持有盒体")

        def check_arrival(p):
            error = float(np.linalg.norm(arm.get_ee_pose()[0] - _xyz(p)))
            if not isfinite(error) or error > .010:
                raise RuntimeError(f"撤离目标未实测到位：{error:.4f} 米")
            if not self.collision_free():
                raise RuntimeError(f"撤离碰撞：{self.audit.summary()}")
            return error

        def grasp_pose():
            return issue(self._finger_target(i, self.handle_world(i)))

        def approach(p):
            p = target(p)
            grasp_ready()
            begin()
            self._move(i, _xyz(p + [0, 0, self.p["approach_clearance"]]),
                       1.0, ki=.2 if i == 0 else .0)
            done.add("approach")
            return observe()

        def descend(p):
            p = target(p)
            need("approach")
            grasp_ready()
            begin()
            self._move(i, p, 1.0, tol=.007 if i == 0 else .008,
                       max_steps=180 if i == 0 else 220, ki=.15 if i == 0 else .0)
            done.add("descend")
            return observe()

        def close_gripper():
            nonlocal close_since
            need("descend")
            grasp_ready()
            begin()
            done.discard("close_gripper")
            close_since = float(self.env.d.time)
            for _ in range(30):
                self.env.hold_arms({arm.s: 0.0}, sub_steps=8)
                self.skills[i]._tick()
            if self.weld_constraint_active(i):
                raise RuntimeError("闭爪不得隐式激活 weld")
            done.add("close_gripper")
            result = observe()
            result["contact_since"] = close_since
            return result

        def attach_contact_gated():
            need("close_gripper")
            grasp_ready()
            begin()
            if not arm._grip_cmd < .5:
                raise RuntimeError("接触门控附着要求闭爪控制仍有效")
            side = ("L", "R")[i]
            contacts = self.audit.recent_handle_contacts(side, close_since)
            simultaneous = {}
            for row in contacts:
                if row["normal_force_n"] > .03 and row["time"] >= float(self.env.d.time) - .10:
                    simultaneous.setdefault(row["time"], set()).add(row["finger_body"])
            bilateral = sum(len(fingers) == 2 for fingers in simultaneous.values())
            current_fingers = simultaneous.get(float(self.env.d.time), set())
            ee_distance = float(np.linalg.norm(arm.get_ee_pose()[0] - self.handle_world(i)))
            if bilateral < 2 or len(current_fingers) != 2 or not isfinite(ee_distance) or ee_distance > .030:
                raise RuntimeError(f"{side} 盒面抓取未过门控：双指帧数={bilateral}，TCP误差={ee_distance:.4f}，接触数={len(contacts)}")
            if not self.collision_free():
                raise RuntimeError(f"{side} 抓取碰撞：{self.audit.summary()}")
            before = self.pos().copy()
            arm._set_weld("baton", True)
            jump = float(np.linalg.norm(self.pos() - before))
            self.events.append({"event": "contact_gated_attach", "arm": i,
                                "time": float(self.env.d.time),
                                "contact_surface": "tea_box_core",
                                "bilateral_contact_steps": bilateral,
                                "tcp_grasp_error_m": ee_distance,
                                "mode": "assisted_grasp_weld", "handles_exist": False,
                                "final_window_s": .10, "activation_pose_jump_m": jump,
                                "both_fingers_contact_at_activation": True})
            for _ in range(12):
                self.env.hold_arms({arm.s: 0.0}, sub_steps=8)
                self.skills[i]._tick()
            if not self.weld_active(i):
                raise RuntimeError("附着后未实测到有效持有")
            if i == 1:
                if not (self.weld_active(0) and self.weld_active(1)):
                    raise RuntimeError("交接未建立实测双侧持有")
                self.events.append({"event": "dual_hold_verified", "time": float(self.env.d.time),
                                    "object_z": float(self.pos()[2])})
            done.add("attach_contact_gated")
            return observe()

        def lift():
            need("attach_contact_gated")
            held()
            begin()
            self._move_object_slow(0, _xyz([self.pos()[0], self.pos()[1], self.h + .20]), step=.006)
            return observe()

        def target_pose():
            if kind == "present":
                return issue(self.p["meet_center_xyz"])
            return issue([*self.p["destination_center_xy"], self.h + self.p["object_half_size"][2] + .003])

        def move_held(p):
            p = target(p)
            held()
            begin()
            self._move_object_slow(0, p, step=.006)
            return observe()

        def settle():
            held()
            begin()
            for _ in range(18):
                self.env.hold_arms({"_L": 0.0, "_R": 1.0}, sub_steps=8)
                self.skills[0]._tick()
            return observe()

        def carry(p):
            p = target(p)
            held()
            begin()
            self._move_object_slow(1, _xyz([p[0], p[1], self.h + .24]), step=.006)
            done.add("carry")
            return observe()

        def lower(p):
            p = target(p)
            need("carry")
            held()
            command_xy = p[:2] + np.asarray(self.p.get("placement_preoffset_xy", [0.0, 0.0]), float)
            if command_xy.shape != (2,) or not np.isfinite(command_xy).all():
                raise ValueError("放置预偏移必须产生两个有限坐标")
            begin()
            self._move_object_slow(1, _xyz([p[0], p[1], self.h + .20]), step=.006)
            sk = self.skills[1]
            sk.record_grab_offset("baton")
            sk.descend_place_tracked(command_xy, self.h, "baton", gap=.003,
                                     max_steps=320, xy_ok=.012, z_step=.018,
                                     suppress_feedforward=True)
            self.events.append({"event": "pre_release_pose", "time": float(self.env.d.time),
                                "object_position": self.pos().tolist()})
            done.add("lower")
            return observe()

        def open_gripper():
            if not self.weld_active(i):
                raise RuntimeError("张爪释放要求绑定臂仍持有盒体")
            if kind == "donor_release":
                receiver_holds()
                if not any(e["event"] == "dual_hold_verified" for e in self.events):
                    raise RuntimeError("缺少真实双侧持有验证事件")
                grips, count, sub_steps = {"_L": 1.0, "_R": 0.0}, 12, 8
            else:
                need("lower")
                held()
                grips, count, sub_steps = {"_R": 1.0}, 8, 4
            begin()
            for _ in range(count):
                self.env.hold_arms(grips, sub_steps=sub_steps)
                self.skills[i]._tick()
            if not self.weld_constraint_active(i):
                raise RuntimeError("张爪不得隐式解除 weld")
            if kind == "donor_release":
                receiver_holds()
            done.add("open_gripper")
            return observe()

        def detach():
            need("open_gripper")
            if not self.weld_constraint_active(i) or not arm._grip_cmd >= .5:
                raise RuntimeError("解除附着要求 weld 仍活动且夹爪张开")
            if kind == "donor_release":
                receiver_holds()
            begin()
            arm._set_weld("baton", False)
            if self.weld_constraint_active(i):
                raise RuntimeError("解除附着后 weld 仍活动")
            if kind == "donor_release":
                receiver_holds()
                self.events.append({"event": "donor_release_after_receiver", "time": float(self.env.d.time),
                                    "receiver_active": bool(self.weld_active(1)), "object_z": float(self.pos()[2])})
            else:
                for _ in range(18):
                    self.env.hold_arms({"_R": 1.0}, sub_steps=8)
                    self.skills[1]._tick()
                self.events.append({"event": "post_release_pose", "time": float(self.env.d.time),
                                    "object_position": self.pos().tolist()})
                if not self.released():
                    raise RuntimeError("放置释放后仍存在活动 weld")
            done.add("detach")
            return observe()

        def withdraw():
            need("detach")
            receiver_holds()
            if self.weld_constraint_active(0):
                raise RuntimeError("交出臂撤离要求已解除附着")
            p = _xyz(self.p["donor_safe_xyz"])
            begin()
            self._move(0, p, 1.0, max_steps=220, ki=.2)
            check_arrival(p)
            receiver_holds()
            self.events.append({"event": "donor_withdrawn", "time": float(self.env.d.time)})
            return observe()

        def retreat_vertical():
            need("detach")
            if not self.released():
                raise RuntimeError("竖直退手要求盒体已解除全部附着")
            p = _xyz(arm.get_ee_pose()[0] + [0, 0, .10])
            begin()
            self._move(1, p, 1.0)
            check_arrival(p)
            done.add("retreat_vertical")
            return observe()

        def withdraw_safe():
            need("retreat_vertical")
            if not self.released():
                raise RuntimeError("接收臂撤离要求盒体已解除全部附着")
            p = _xyz(self.p["receiver_safe_xyz"])
            begin()
            self._move(1, p, 1.0, max_steps=220, ki=.10)
            check_arrival(p)
            geometry = self.placement_geometry()
            if not geometry["placed"]:
                raise RuntimeError(f"盒体未在目的地落稳：{geometry}")
            self.events.append({"event": "receiver_withdrawn", "time": float(self.env.d.time)})
            return observe()

        def read():
            return dict(self.placement_geometry())

        def return_arm_home():
            if self.weld_constraint_active(i):
                raise RuntimeError("空臂复位要求不存在活动 weld，张爪不能替代解除附着")
            begin()
            self._controlled_home(i)
            result = observe()
            result["home"] = bool(self.home(i))
            return result

        values = {
            "grasp_pose": grasp_pose, "approach": approach, "descend": descend,
            "close_gripper": close_gripper, "attach_contact_gated": attach_contact_gated,
            "lift": lift, "target_pose": target_pose, "move_held": move_held,
            "settle": settle, "carry": carry, "lower": lower, "read": read,
            "open_gripper": open_gripper, "detach": detach, "withdraw": withdraw,
            "retreat_vertical": retreat_vertical, "withdraw_safe": withdraw_safe,
            "return_arm_home": return_arm_home,
        }
        return {name: values[name] for name in DOCS[kind]}


def reference_models():
    descriptions = {
        "donor_pick": "交出臂持有茶盒且茶盒位于空中，接收臂为空，源位置不再占用，运动无禁止碰撞。",
        "present": "交出臂持有的茶盒位于配置的空中交接位，接收臂为空，运动无禁止碰撞。",
        "receive": "茶盒由两臂共同持有，接收侧指定盒面的当帧与近期双指接触已验证，交出侧仍保持持有。",
        "donor_release": "茶盒由接收臂持有，交出臂为空且位于安全撤离位；真实事件证明接收验证先于空中释放，运动无禁止碰撞。",
        "receiver_place": "茶盒位于目的地并由桌面真实支撑，接收臂为空且位于安全撤离位，交出臂保持 HOME，运动无禁止碰撞。",
        "controlled_home": "绑定的空机械臂处于经执行器控制及实测验证的 HOME 位，无活动抓取 weld，运动无禁止碰撞。",
    }
    models = box_domain()
    for model in models:
        model["description"] = descriptions[model["program_kind"]]
    return models


def factory(config, out, seed):
    return FineHandoverRuntime(config, out, seed=seed, render=True)
