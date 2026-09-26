"""双臂倒料与收纳的细粒度接口，不改动原状态、目标或审计。

几何读数来自原仿真真值，运动使用既有标定控制器；抓持仍包含显式声明的
weld 辅助。这里不写 qpos、不设置成功标记，也不替代独立物理审计。
运行时构造、save/close 的副作用和判定门限均继承原实现。
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.dual_lift_shelf_bridge import DualLiftRuntime, lift_domain
from core.smooth_held_basin_bridge import SmoothHeldBasinRuntime, domain
from probe_free_tray_packing import FreeTrayRuntime


_PICK_DOCS = {
    "target_pose": "api.target_pose() -> tuple(x,y,z)：只读绑定物体的真值抓取点，z 含原标定的 TCP 高度偏移。",
    "approach": "api.approach(xyz)：张开手爪运动到 xyz 的水平位置、桌面上方 0.24m；不下降到抓取高度、不抓持。",
    "align_xy": "api.align_xy(xyz)：与 approach 使用同一控制器，张开手爪对齐水平位置及桌面上方 0.24m 的高度。",
    "descend_to": "api.descend_to(z, lock_xy)：张开手爪运动至高度 z，水平目标为二元组 lock_xy；不闭合手指。",
    "grasp": "api.grasp(xy)：在当前高度跟踪二元组 xy 并闭合手指；启用原有抓持辅助，满足底层条件时可激活 weld，不负责抬升。",
    "lift": "api.lift()：保持闭合手爪，将末端抬至桌面上方 0.30m 的运输高度；沿用原控制器容差。",
    "record_grab_offset": "api.record_grab_offset() -> None：读取并记录当前已抓物体相对末端的偏移，用于放置跟踪；不移动物体。",
}

_RETURN_DOCS = {
    "safe_pose": "api.safe_pose() -> tuple(x,y,z)：只读绑定物体的标定安全通道原点，z 为桌面上方 0.32m；不是 TCP 位姿。",
    "move_safe": "api.move_safe(value)：value 必须等于绑定安全通道原点；保持抓持搬运物体，源杯采用原点跟踪及精对齐，接收盆采用缓慢闭环运输；不释放。",
    "target_pose": "api.target_pose() -> tuple(x,y,z)：只读返还目标的水平坐标和桌面高度，包含原配置的放置预偏移。",
    "move_above": "api.move_above(value)：保持抓持，将物体运至 value 的水平位置、桌面上方 0.30m；不降低到桌面、不释放。",
    "align_target": "api.align_target(value) -> tuple(x,y)：按当前物体与末端偏移精调水平位置，末端高度目标为桌面上方 0.30m；返回补偿后的 TCP 水平目标。",
    "lower": "api.lower(value)：记录抓持偏移，按物体把手高度跟踪下降至 value 指定的支撑面，沿用 0.003m 间隙并等待 20 个保持周期；不释放。",
    "release": "api.release()：禁用自动抓持辅助，在当前位置张开手爪并解除该臂的 weld，检查绑定物体已不再被抓持；不撤离。",
    "retreat": "api.retreat() -> float：要求绑定物体已释放；张开手爪向该臂外侧移动 0.10m，高度目标为桌面上方 0.28m，检查原运动残差及高度门限；返回残差。",
    "settle": "api.settle()：保持现有手爪命令等待 16 个保持周期，不改写物体位姿或成功状态。",
}

POUR_DOCS = {
    "pick": deepcopy(_PICK_DOCS),
    "present_receiver": {
        "target_pose": "api.target_pose() -> tuple(x,y,z)：只读接料站的标定接收盆把手目标。",
        "move_basin": "api.move_basin(value)：value 必须等于绑定把手目标；通过运输及精对齐控制器移动已抓住的接收盆，不释放。",
        "settle": "api.settle()：保持接收盆抓持，等待 12 个保持周期。",
    },
    "coordinated_pour": {
        "source_pose": "api.source_pose() -> tuple(x,y,z)：只读接料区上方的标定源杯把手目标。",
        "move_source": "api.move_source(value)：保持源杯抓持，将其运输并对齐到绑定目标；不倾倒。运动会使接收盆的既有对齐证据失效。",
        "receiver_pose": "api.receiver_pose() -> tuple(x,y,z)：只读接料站的标定接收盆把手目标。",
        "align_receiver": "api.align_receiver(value)：要求源杯运输证据有效；保持盆抓持，精调至绑定目标并等待，核验当前接料几何。",
        "tip_source": "api.tip_source()：要求当前接料对齐证据有效；记录实际倾倒前关节目标，通过关节执行器使源杯倾斜越过水平，核验小球转移并启用保留监测；不直接移动小球或设置成功。",
        "settle": "api.settle()：保持两侧抓持，等待 16 个保持周期。",
    },
    "separate_receiver": {
        "separate_receiver": "api.separate_receiver()：要求盆仍被抓住且含接住的小球；保持盆中心高度，缓慢移向负 y 安全通道并等待 12 个保持周期，核验小球未离盆；不抓取或释放。",
    },
    "restore_source": {
        "restore_upright": "api.restore_upright()：要求源杯仍被抓住且存在真实倾倒前记录；在当前末端位置求解直立 IK，插值关节执行器目标，恢复姿态伺服并上移 0.025m、保持和核验直立角度；不抓取、释放或写 qpos。",
    },
    "return_source": deepcopy(_RETURN_DOCS),
    "return_basin": deepcopy(_RETURN_DOCS),
    "controlled_home": {
        "controlled_home": "api.controlled_home()：要求该臂为空且任务物体已释放；张开手爪经过配置的外侧笛卡尔点，再插值关节执行器目标至 HOME 并保持、核验；这是空臂复位运动，不是抓放操作，不写 qpos。",
    },
}

STORAGE_DOCS = {
    "pick": deepcopy(_PICK_DOCS),
    "place": {
        "target_pose": "api.target_pose() -> tuple(x,y,z)：只读当前自由托盘姿态下的绑定槽位及托盘底面支撑高度。",
        "move_above": "api.move_above(p)：保持抓持，补偿当前物体与末端偏移，运至 p 的水平位置，末端高度目标为桌面上方 0.30m；不释放。",
        "align_target": "api.align_target(p) -> tuple(x,y)：按实际物体位姿精调槽位上方的水平位置，末端高度目标为桌面上方 0.30m；返回补偿后的 TCP 水平目标。",
        "lower": "api.lower(p)：记录抓持偏移并跟踪物体下降至 p 指定的支撑面，沿用 0.003m 间隙；不释放。",
        "place_release": "api.place_release()：在当前位置禁用自动抓持辅助，张开手爪、解除该臂 weld 并保持；不撤离。",
        "retreat": "api.retreat()：张开手爪竖直移至桌面上方 0.30m 的目标高度；沿用原控制器容差。",
        "settle": "api.settle()：保持现有手爪命令等待 16 个保持周期。",
    },
    "park": {
        "raise_clear": "api.raise_clear()：要求空手；保持当前水平目标，张开手爪竖直运动至 max(当前高度, 桌面高度 + 0.34m)，沿用原控制器容差并记录显式撤离证据；不横移至停泊点。",
        "target_pose": "api.target_pose() -> tuple(x,y,z)：只读共享工作区外的绑定停泊目标，不运动、不建立撤离证据。",
        "move_to": "api.move_to(p) -> float：要求空手且本次 API 绑定已有显式竖直撤离证据；用原基础控制器单段运动至 p，不隐含抬升、抓取或放置；返回运动残差。",
        "settle": "api.settle()：保持现有手爪命令等待 16 个保持周期，不建立撤离证据。",
    },
    "dual_grasp": {
        "target_handles": "api.target_handles() -> tuple(left_xyz,right_xyz)：只读双侧托盘把手的真值坐标并缓存该组几何读数。",
        "move_both_above": "api.move_both_above(value)：value 为当前缓存的完整双把手坐标组；同步移动两只张开的手至把手水平位置、桌面上方 0.29m。",
        "move_both_to": "api.move_both_to(value)：value 为当前缓存的完整双把手坐标组；同步移动两只张开的手至把手位置，不闭合、不附着。",
        "close_both": "api.close_both()：禁用自动抓持辅助，在当前把手位置闭合两侧手爪；本接口不激活托盘 weld。",
        "attach_both": "api.attach_both()：要求已缓存把手几何且两侧手爪闭合；检查 TCP 和手指中点误差，激活两个已声明的托盘抓持辅助 weld、记录事件并保持。",
        "settle": "api.settle()：保持现有手爪命令等待 12 个保持周期。",
    },
    "cooperative_move": {
        "target_tray_pose": "api.target_tray_pose() -> tuple(x,y,z)：只读本动作绑定的托盘底部目标。",
        "move_tray_to": "api.move_tray_to(value)：value 必须等于绑定目标；保持双侧抓持，同步平移两臂，闭环修正托盘位置并检查物品仍在托盘内；不释放。",
        "settle": "api.settle()：保持双侧抓持等待 12 个保持周期。",
    },
    "dual_release": {
        "open_both": "api.open_both()：在当前笛卡尔目标处张开两侧手爪并推进仿真，不发出横向或竖直运动目标；底层原有开爪逻辑可能解除抓持 weld。",
        "detach_both": "api.detach_both()：要求两侧手爪张开；显式禁用两个托盘抓持辅助 weld，记录事件并核验均已解除，不移动两臂。",
        "settle": "api.settle()：保持当前命令等待 20 个保持周期，使释放的托盘在支撑面上稳定。",
    },
    "dual_withdraw": {
        "raise_both": "api.raise_both()：保持两侧手爪张开，同步将各末端的高度目标增加 0.13m，水平目标不变；不横向停泊。",
        "settle": "api.settle()：保持张开手爪等待 12 个保持周期。",
    },
}


class FinePourRuntime(SmoothHeldBasinRuntime):
    """仅拆开返还接口，其余控制、事件、状态及独立审计均继承。"""

    def api(self, model):
        kind = model["program_kind"]
        if kind not in ("return_source", "return_basin"):
            return super().api(model)
        self.audit.phase = model["name"]
        i = int(model["args"]["arm"])
        obj = model["args"]["obj"]
        arm = self.env.arms[i]
        sk = self.scout.skills[i]

        # 对应原 smooth_held_basin_bridge.py 的九个局部函数，保留控制语义。
        def safe_pose():
            return tuple(map(float, self._safe_origin(obj)))

        def move_safe(value):
            point = self._xyz(value)
            if np.linalg.norm(point - self._safe_origin(obj)) > 1e-6:
                raise ValueError("安全通道坐标必须保持原值")
            return self._move_object_slow(i, obj, point) if obj == "basin" else self._go_object_origin(i, obj, point)

        def target_pose():
            key = "source_return_preoffset_xy" if obj == "canL" else "basin_return_preoffset_xy"
            xy = self._return_origin_xy(obj) + np.asarray(self.b.get(key, [0.0, 0.0]), float)
            return (float(xy[0]), float(xy[1]), float(self.h))

        def move_above(value):
            p = self._xyz(value)
            target = [p[0], p[1], self.h + .30]
            return self._move_object_slow(i, obj, target) if obj == "basin" else self._go_object_origin(i, obj, target)

        def align_target(value):
            p = self._xyz(value)
            return tuple(map(float, sk.refine_above(p[:2], high_z=self.h + .30, obj_name=obj)))

        def lower(value):
            p = self._xyz(value)
            sk.record_grab_offset(obj)
            saved = sk.block_half
            try:
                sk.block_half = self._handle_height(obj)
                result = sk.descend_place_tracked(p[:2], p[2], obj, gap=.003, suppress_feedforward=True)
                self.scout.hold(20)
                return result
            finally:
                sk.block_half = saved

        def release():
            arm._grasp_enabled = False
            for _ in range(8):
                self.env.hold_arms({arm.s: 1.0}, sub_steps=4)
                sk._tick()
            arm.release_all()
            if arm.is_holding(obj):
                raise RuntimeError("释放后物体 weld 仍处于激活状态")

        def retreat():
            if arm.is_holding(obj):
                raise RuntimeError("撤离要求绑定物体已释放")
            saved = arm.bias_world.copy()
            arm.bias_world = np.zeros(3)
            try:
                ee = sk.ee()
                side = 1.0 if i == 0 else -1.0
                target = [ee[0], ee[1] + side * .10, self.h + .28]
                err = sk.move_to(target, grip=1.0, tol=.015, max_steps=180, ki=.15)
            finally:
                arm.bias_world = saved
            if err > .04 or sk.ee()[2] < self.h + .23:
                raise RuntimeError(f"向外撤离失败：err={err:.4f}, z={sk.ee()[2]:.4f}")
            return float(err)

        def settle():
            self.scout.hold(16)

        return {
            "safe_pose": safe_pose,
            "move_safe": move_safe,
            "target_pose": target_pose,
            "move_above": move_above,
            "align_target": align_target,
            "lower": lower,
            "release": release,
            "retreat": retreat,
            "settle": settle,
        }


class FineStorageRuntime(DualLiftRuntime):
    """停泊的竖直撤离由调用方显式触发，不再藏在横移接口内。"""

    def api(self, model):
        if model["program_kind"] != "park":
            return super().api(model)
        # 绕过 DualLiftRuntime 的预撤离 wrapper，只取原单段停泊控制器。
        base_api = FreeTrayRuntime.api(self, model)
        i = int(model["args"]["arm"])
        arm = self.env.arms[i]
        cleared = False

        def raise_clear():
            nonlocal cleared
            cleared = False
            if any(arm.is_holding(name) for name in self.env.obj_names):
                raise RuntimeError("竖直撤离要求空手")
            ee = arm.get_ee_pose()[0]
            safe = np.array([ee[0], ee[1], max(ee[2], self.h + .34)])
            self.scout.go(i, safe, 1.0)
            self.scout.events.append({
                "event": "physical_prepark_vertical_clearance", "arm": i,
                "sim_time": float(self.env.d.time), "target": safe.tolist(),
                "explicit_api": "raise_clear",
            })
            cleared = True

        def move_to(p):
            if not cleared:
                raise RuntimeError("停泊运动缺少本次 API 绑定的显式竖直撤离证据")
            return base_api["move_to"](p)

        return {
            "raise_clear": raise_clear,
            "target_pose": base_api["target_pose"],
            "move_to": move_to,
            "settle": base_api["settle"],
        }


_POUR_DESCRIPTIONS = {
    "pick_basin": "接收臂抓持接收盆，盆已离开桌面处于运输高度；抓持偏移可用于后续位置跟踪。",
    "present_basin": "接收盆保持被抓持，位于标定接料站且满足接料几何。",
    "pick_source_cup": "源臂抓持含小球的源杯，源杯已离开桌面处于运输高度；抓持偏移可用于后续位置跟踪。",
    "pour_into_held_basin": "小球由实际倾斜越过水平的源杯转移到仍被另一臂抓持的接收盆；两侧抓持保持，小球留在盆内。",
    "separate_receiver_after_pour": "接收盆仍被抓持，位于负 y 安全通道并保留小球，与源杯分离。",
    "restore_source_upright": "源杯仍被源臂抓持且恢复运输要求的直立姿态，小球仍留在接收盆内。",
    "return_source_cup": "源杯在标定返还位置直立落桌且已解除抓持，源臂空手并撤至外侧安全高度，小球仍留在接收盆内。",
    "home_source_arm": "源杯已返还，源臂空手张开并处于 HOME 关节姿态，小球仍留在接收盆内。",
    "return_basin": "接收盆在标定返还位置落桌且已解除抓持，小球保留在盆内，接收臂空手并撤至外侧安全高度。",
    "home_receiver_arm": "源杯与接收盆已返还，两臂空手张开并处于 HOME 关节姿态，小球仍留在接收盆内。",
}

_STORAGE_DESCRIPTIONS = {
    "pick": "绑定物品被指定机械臂抓持并抬离桌面，抓持偏移可用于放置跟踪。",
    "place": "绑定物品已在自由托盘对应槽位内稳定支撑且解除抓持，指定手爪张开并竖直撤离放置面。",
    "park": "指定机械臂空手张开，已脱离物体周围的竖直障碍区并停泊在共享工作区外。",
    "dual_grasp": "两个物品仍在桌上托盘内，两臂在把手处闭合并通过已声明的双 weld 辅助共同抓持托盘。",
    "dual_release": "托盘由货架支撑、双侧抓持 weld 均已解除且两只手爪张开，物品仍在托盘内。",
    "dual_withdraw": "托盘已释放在货架上且物品仍被容纳，两只张开的手均高于把手并满足横向运动的净空条件。",
}
_STORAGE_WAYPOINT_DESCRIPTIONS = {
    "dual_lift_tray": "两臂保持共同抓持，托盘在原桌面水平位置达到标定运输高度，两个物品仍在托盘内。",
    "dual_translate_tray": "两臂保持共同抓持，托盘在运输高度位于货架上方，两个物品仍在托盘内。",
    "dual_lower_tray": "两臂保持共同抓持，托盘位于货架目标位置并获得货架支撑，两个物品仍在托盘内。",
}


def factory(task, config, out, seed=0):
    """构造原生命周期的运行时；storage 继续使用原 packing 配置。"""
    if task == "pour":
        return FinePourRuntime(config, Path(out), seed)
    if task == "storage":
        return FineStorageRuntime(config, Path(out), seed)
    raise ValueError(f"不支持的任务：{task!r}；应为 pour 或 storage")


def reference_models(task):
    """仅重写参考动作的终态描述，保留参数、前提和增删效果。"""
    if task == "pour":
        models = deepcopy(domain())
        for model in models:
            model["description"] = _POUR_DESCRIPTIONS[model["name"]]
        return models
    if task == "storage":
        models = deepcopy(lift_domain())
        for model in models:
            if model["program_kind"] == "cooperative_move":
                model["description"] = _STORAGE_WAYPOINT_DESCRIPTIONS[model["name"]]
            else:
                model["description"] = _STORAGE_DESCRIPTIONS[model["program_kind"]]
        return models
    raise ValueError(f"不支持的任务：{task!r}；应为 pour 或 storage")


def docs_for(task, m):
    """返回绑定 program_kind 的独立文档副本，不包含策略代码答案。"""
    if task == "pour":
        docs = POUR_DOCS
    elif task == "storage":
        docs = STORAGE_DOCS
    else:
        raise ValueError(f"不支持的任务：{task!r}；应为 pour 或 storage")
    return deepcopy(docs[m["program_kind"]])
