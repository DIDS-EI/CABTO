"""统一相机挂载的轻量回归测试：仅编译小型 MJCF 和执行 mj_forward。"""

import hashlib
import importlib.util
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# 直接载入待测文件，避免项目入口引入模型、渲染器或控制循环。
_SOURCE = Path(__file__).resolve().parent / "core" / "unified_camera_rig.py"
_SPEC = importlib.util.spec_from_file_location("_camera_rig_under_test", _SOURCE)
rig = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rig)


CONFIG = {"table_center": [0.2, -0.1, 0.0], "table_top": 0.4}
SINGLE = ("hand",)
DUAL = ("hand_L", "hand_R")
CAMERA_NAMES = {
    "hand": "rig_wrist_single",
    "hand_L": "rig_wrist_left",
    "hand_R": "rig_wrist_right",
}


def small_xml(hands=SINGLE, object_pos="0.2 -0.1 0.46"):
    bodies = []
    motors = []
    for i, name in enumerate(hands):
        bodies.append(f"""
        <body name="carrier_{i}" pos="{0.1 + 0.3 * i} 0.15 1.0"
              euler="0.12 -0.18 0.25">
          <freejoint name="free_{i}"/>
          <geom name="carrier_geom_{i}" type="sphere" size="0.025" mass="0.2"/>
          <body name="{name}" pos="0.03 0.01 0.04" quat="0 1 0 0">
            <joint name="hinge_{i}" type="hinge" axis="0 1 0" range="-1 1"
                   damping="0.3" frictionloss="0.02"/>
            <geom name="hand_geom_{i}" type="box" size="0.04 0.025 0.03" mass="0.4"/>
            <camera name="legacy_hand_{i}" pos="0.01 0.02 -0.03" fovy="43"/>
          </body>
        </body>""")
        motors.append(f'<motor name="motor_{i}" joint="hinge_{i}" '
                      'ctrllimited="true" ctrlrange="-2 2" gear="0.7"/>')
    return f"""<mujoco model="tiny_camera_rig">
      <compiler angle="radian"/>
      <option timestep="0.003" gravity="0 0 -9.81"/>
      <default><geom friction="0.6 0.02 0.001" solref="0.03 1"/></default>
      <worldbody>
        <camera name="legacy_global" pos="1 -1 2" quat="0.9238795325 0 0.3826834324 0" fovy="51"/>
        <geom name="table" type="box" pos="0.2 -0.1 0.35" size="0.5 0.4 0.05"/>
        {''.join(bodies)}
        <body name="object" pos="{object_pos}">
          <freejoint name="object_free"/>
          <geom name="object_geom" type="sphere" size="0.025" mass="0.08"/>
        </body>
      </worldbody>
      <actuator>{''.join(motors)}</actuator>
      <equality><joint name="test_equality" joint1="hinge_0" active="false"/></equality>
    </mujoco>"""


def make_env(xml, img_size=320):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return SimpleNamespace(m=m, d=d, img_size=img_size)


def augmented_env(hands=SINGLE, config=None, profile=None):
    xml, meta = rig.augment_xml(small_xml(hands), config or CONFIG, profile)
    size = (profile or rig.PROFILE)["width"]
    return make_env(xml, size), meta


def fake_exp4_module(fail=False):
    # 仅替身构造器负责真实 MJCF 编译；不导入完整 Exp4Env 及其依赖。
    module = ModuleType("exp4_env")

    class Exp4Env:
        def __init__(self, xml, img_size=64, *, marker=None):
            self.xml = xml
            self.img_size = img_size
            self.marker = marker
            if fail:
                raise RuntimeError("模拟构造失败")
            env = make_env(xml, img_size)
            self.m, self.d = env.m, env.d

    module.Exp4Env = Exp4Env
    return module


class UnifiedCameraRigTests(unittest.TestCase):
    def assert_close(self, actual, expected, atol=2e-10):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)

    def test_single_arm_adds_exactly_two_cameras(self):
        xml = small_xml()
        augmented, meta = rig.augment_xml(xml, CONFIG)
        before, after = make_env(xml), make_env(augmented)
        self.assertEqual(after.m.ncam, before.m.ncam + 2)
        self.assertEqual(meta["cameras"], ["rig_wrist_single", "rig_global"])
        self.assertEqual(after.m.camera("rig_wrist_single").bodyid[0], after.m.body("hand").id)
        self.assertEqual(after.m.camera("rig_global").bodyid[0], 0)

    def test_dual_arm_adds_exactly_three_cameras(self):
        xml = small_xml(DUAL)
        augmented, meta = rig.augment_xml(xml, CONFIG)
        before, after = make_env(xml), make_env(augmented)
        self.assertEqual(after.m.ncam, before.m.ncam + 3)
        self.assertEqual(meta["cameras"], ["rig_wrist_left", "rig_wrist_right", "rig_global"])
        for hand in DUAL:
            self.assertEqual(after.m.camera(CAMERA_NAMES[hand]).bodyid[0], after.m.body(hand).id)

    def test_legacy_cameras_keep_attributes_parents_and_calibration(self):
        xml = small_xml(DUAL)
        augmented, meta = rig.augment_xml(xml, CONFIG)
        before, after = make_env(xml), make_env(augmented)

        def camera_nodes(text):
            return {c.get("name"): dict(c.attrib) for c in ET.fromstring(text).iter("camera")}

        originals, installed = camera_nodes(xml), camera_nodes(augmented)
        self.assertEqual(meta["old_cameras_preserved"], list(originals))
        for name, attributes in originals.items():
            with self.subTest(camera=name):
                self.assertEqual(installed[name], attributes)
                i, j = before.m.camera(name).id, after.m.camera(name).id
                for key in ("cam_pos", "cam_quat", "cam_fovy", "cam_mode", "cam_bodyid", "cam_targetbodyid"):
                    np.testing.assert_array_equal(getattr(before.m, key)[i], getattr(after.m, key)[j])
                self.assert_close(before.d.cam_xpos[i], after.d.cam_xpos[j])
                self.assert_close(before.d.cam_xmat[i], after.d.cam_xmat[j])

    def test_repeated_installation_is_rejected(self):
        for hands in (SINGLE, DUAL):
            with self.subTest(hands=hands):
                augmented, _ = rig.augment_xml(small_xml(hands), CONFIG)
                with self.assertRaisesRegex(ValueError, "already installed"):
                    rig.augment_xml(augmented, CONFIG)

    def test_existing_rig_camera_name_is_rejected(self):
        xml = small_xml().replace('name="legacy_global"', 'name="rig_custom"')
        with self.assertRaisesRegex(ValueError, "already installed"):
            rig.augment_xml(xml, CONFIG)

    def test_missing_worldbody_and_hands_are_rejected(self):
        for xml, message in (("<mujoco/>", "Missing worldbody"),
                             ("<mujoco><worldbody/></mujoco>", "No Panda hand")):
            with self.subTest(xml=xml), self.assertRaisesRegex(ValueError, message):
                rig.augment_xml(xml, CONFIG)

    def test_all_hand_names_share_identical_local_extrinsics(self):
        custom = deepcopy(rig.PROFILE)
        custom["wrist"].update(position_hand_m=[0.05, 0.01, 0.03],
                               look_at_hand_m=[0.01, -0.02, 0.4], fovy_deg=58.0)
        for profile in (rig.PROFILE, custom):
            measurements = []
            for hands in (SINGLE, DUAL):
                env, meta = augmented_env(hands, profile=profile)
                for mount in meta["wrist_mounts"]:
                    with self.subTest(hand=mount["parent_body"], fovy=mount["fovy_deg"]):
                        cal = rig.calibration(env, mount["name"])
                        local = cal["measured_T_body_camera"]
                        self.assertEqual(cal["parent_body"], mount["parent_body"])
                        self.assert_close(local["position"], profile["wrist"]["position_hand_m"])
                        self.assert_close(local["rotation"], mount["T_hand_camera"]["rotation"])
                        self.assertEqual(cal["fovy_deg"], profile["wrist"]["fovy_deg"])
                        measurements.append(local)
            for local in measurements[1:]:
                self.assert_close(local["position"], measurements[0]["position"])
                self.assert_close(local["rotation"], measurements[0]["rotation"])

    def test_global_camera_ignores_object_xml_and_config_positions(self):
        results = []
        for position in ([0.2, -0.1, 0.46], [9.0, -7.0, 4.0]):
            config = deepcopy(CONFIG)
            config.update(object_pos=position, objects={"object": {"position": position}})
            xml, meta = rig.augment_xml(small_xml(object_pos=" ".join(map(str, position))), config)
            env = make_env(xml)
            results.append((env, meta, rig.calibration(env, "rig_global")))
        self.assertFalse(np.allclose(results[0][0].d.xpos[results[0][0].m.body("object").id],
                                     results[1][0].d.xpos[results[1][0].m.body("object").id]))
        self.assertEqual(results[0][1]["global_position_world"], results[1][1]["global_position_world"])
        self.assertEqual(results[0][2], results[1][2])

    def test_global_camera_uses_table_frame_and_custom_profile(self):
        profile = deepcopy(rig.PROFILE)
        profile["global"].update(offset_table_xy_m=[-0.12, 0.23],
                                 height_above_table_m=1.7, fovy_deg=48.0)
        for config, expected in (
            (CONFIG, [0.08, 0.13, 2.1]),
            ({"table": {"center_xy": [-0.5, 0.7], "top_z": 0.8}}, [-0.62, 0.93, 2.5]),
        ):
            with self.subTest(config=config):
                env, meta = augmented_env(config=config, profile=profile)
                cal = rig.calibration(env, "rig_global")
                self.assert_close(meta["global_position_world"], expected)
                self.assert_close(cal["T_world_camera"]["position"], expected)
                self.assert_close(cal["T_world_camera"]["rotation"], np.eye(3))
                self.assertEqual(cal["fovy_deg"], 48.0)

    def test_table_config_schemas_are_equivalent(self):
        flat = rig.augment_xml(small_xml(), CONFIG)
        nested = rig.augment_xml(small_xml(), {"table": {"center_xy": [0.2, -0.1], "top_z": 0.4}})
        self.assertEqual(flat, nested)

    def test_augmentation_does_not_mutate_input_profile_or_config(self):
        profile, config = deepcopy(rig.PROFILE), deepcopy(CONFIG)
        original_profile, original_config = deepcopy(profile), deepcopy(config)
        _, meta = rig.augment_xml(small_xml(), config, profile)
        self.assertEqual(profile, original_profile)
        self.assertEqual(config, original_config)
        meta["profile"]["wrist"]["position_hand_m"][0] = 999
        self.assertEqual(profile, original_profile)
        self.assertEqual(rig.PROFILE, original_profile)

    def test_only_camera_xml_elements_are_added(self):
        def without_cameras(node):
            return (node.tag, sorted(node.attrib.items()), (node.text or "").strip(),
                    tuple(without_cameras(child) for child in node if child.tag != "camera"))

        for hands in (SINGLE, DUAL):
            with self.subTest(hands=hands):
                xml = small_xml(hands)
                augmented, _ = rig.augment_xml(xml, CONFIG)
                self.assertEqual(without_cameras(ET.fromstring(xml)),
                                 without_cameras(ET.fromstring(augmented)))

    def test_physics_mass_inertia_and_all_geom_arrays_are_unchanged(self):
        for hands in (SINGLE, DUAL):
            with self.subTest(hands=hands):
                xml = small_xml(hands)
                augmented, _ = rig.augment_xml(xml, CONFIG)
                before, after = make_env(xml), make_env(augmented)
                report = rig.physics_comparison(before.m, after.m)
                self.assertTrue(report["matched"])
                self.assertTrue(report["only_sensor_configuration_changed"])
                self.assertEqual(report["new_ncam"] - report["old_ncam"], len(hands) + 1)
                self.assertGreater(before.m.nu, 0)
                self.assertGreater(before.m.neq, 0)
                keys = {"body_mass", "body_inertia", "body_ipos", "body_iquat"}
                keys.update(key for key in dir(before.m)
                            if key.startswith("geom_") and isinstance(getattr(before.m, key), np.ndarray))
                for key in sorted(keys):
                    np.testing.assert_array_equal(getattr(before.m, key), getattr(after.m, key), err_msg=key)
                self.assert_close(before.d.qM, after.d.qM)
                self.assert_close(before.d.qfrc_bias, after.d.qfrc_bias)

    def test_physics_comparison_detects_mass_and_geom_changes(self):
        for key in ("body_mass", "geom_size"):
            with self.subTest(array=key):
                reference, actual = make_env(small_xml()), make_env(small_xml())
                getattr(actual.m, key).flat[-1] += 0.01
                with self.assertRaises(AssertionError):
                    rig.physics_comparison(reference.m, actual.m)

    def test_wrist_world_pose_follows_parent_but_local_pose_stays_fixed(self):
        for hands in (SINGLE, DUAL):
            env, _ = augmented_env(hands)
            global_before = rig.calibration(env, "rig_global")
            before = {hand: rig.calibration(env, CAMERA_NAMES[hand]) for hand in hands}
            for i, hand in enumerate(hands):
                adr = int(env.m.jnt_qposadr[env.m.joint(f"free_{i}").id])
                env.d.qpos[adr:adr + 3] += [0.17, -0.08, 0.13]
                angle = 0.65 + i * 0.2
                env.d.qpos[adr + 3:adr + 7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
                env.d.qpos[env.m.jnt_qposadr[env.m.joint(f"hinge_{i}").id]] = 0.3
            mujoco.mj_forward(env.m, env.d)
            for hand in hands:
                with self.subTest(hand=hand):
                    old = before[hand]
                    new = rig.calibration(env, CAMERA_NAMES[hand])
                    for key in ("position", "rotation"):
                        self.assertFalse(np.allclose(old["T_world_camera"][key], new["T_world_camera"][key]))
                        self.assert_close(old["measured_T_body_camera"][key], new["measured_T_body_camera"][key])
                    bid = env.m.body(hand).id
                    body_r = env.d.xmat[bid].reshape(3, 3)
                    local = old["measured_T_body_camera"]
                    self.assert_close(new["T_world_camera"]["position"],
                                      env.d.xpos[bid] + body_r @ local["position"])
                    self.assert_close(new["T_world_camera"]["rotation"], body_r @ local["rotation"])
            self.assertEqual(global_before, rig.calibration(env, "rig_global"))

    def test_project_ray_and_plane_round_trip(self):
        for hands in (SINGLE, DUAL):
            env, meta = augmented_env(hands)
            for name in meta["cameras"]:
                cal = rig.calibration(env, name)
                for point in ([0.1, -0.1, 0.4], [0.3, 0.0, 0.4], [-0.1, 0.2, 0.4]):
                    with self.subTest(hands=hands, camera=name, point=point):
                        uv = rig.project(cal, point)
                        self.assertIsNotNone(uv)
                        origin, direction = rig.ray(cal, uv)
                        displacement = np.asarray(point) - origin
                        self.assert_close(np.linalg.norm(direction), 1.0)
                        self.assert_close(direction, displacement / np.linalg.norm(displacement))
                        self.assert_close(rig.intersect_plane(cal, uv, point[2]), point)
                for uv in ([123.0, 256.0], [360.0, 360.0], [540.0, 470.0]):
                    with self.subTest(camera=name, uv=uv):
                        point = rig.intersect_plane(cal, uv, CONFIG["table_top"])
                        self.assert_close(rig.project(cal, point), uv, atol=1e-9)

    def test_optical_axis_is_minus_z_and_image_axes_have_correct_sign(self):
        env, meta = augmented_env(DUAL)
        for name in meta["cameras"]:
            with self.subTest(camera=name):
                cal = rig.calibration(env, name)
                r = np.asarray(cal["T_world_camera"]["rotation"])
                position = np.asarray(cal["T_world_camera"]["position"])
                k = np.asarray(cal["K"])
                center = k[:2, 2]
                self.assert_close(cal["optical_axis_world"], -r[:, 2])
                self.assert_close(rig.ray(cal, center)[1], -r[:, 2])
                self.assert_close(rig.project(cal, position - 2 * r[:, 2]), center)
                right = rig.project(cal, position + r @ np.array([0.2, 0, -2]))
                up = rig.project(cal, position + r @ np.array([0, 0.2, -2]))
                self.assert_close(right, center + [0.1 * k[0, 0], 0])
                self.assert_close(up, center + [0, -0.1 * k[1, 1]])

    def test_points_behind_camera_or_at_camera_origin_are_rejected(self):
        env, meta = augmented_env(DUAL)
        for name in meta["cameras"]:
            cal = rig.calibration(env, name)
            r = np.asarray(cal["T_world_camera"]["rotation"])
            position = np.asarray(cal["T_world_camera"]["position"])
            for point in (position, position + 0.5 * r[:, 2], position + r @ [0.2, -0.1, 1]):
                with self.subTest(camera=name, point=point):
                    self.assertIsNone(rig.project(cal, point))

    def test_plane_behind_camera_or_through_origin_is_rejected(self):
        env, _ = augmented_env()
        cal = rig.calibration(env, "rig_global")
        height = cal["T_world_camera"]["position"][2]
        for z in (height, height + 1):
            with self.subTest(z=z), self.assertRaisesRegex(ValueError, "behind camera"):
                rig.intersect_plane(cal, [360, 360], z)

    def test_plane_parallel_to_ray_is_rejected(self):
        cal = {"K": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
               "T_world_camera": {"position": [0, 0, 1],
                                  "rotation": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]}}
        with self.assertRaisesRegex(ValueError, "parallel"):
            rig.intersect_plane(cal, [50, 50], 0.4)

    def test_calibration_intrinsics_and_missing_camera(self):
        env, meta = augmented_env()
        env.img_size = 256
        for name in meta["cameras"]:
            cal = rig.calibration(env, name)
            expected_f = 128 / np.tan(np.deg2rad(cal["fovy_deg"]) / 2)
            self.assert_close(cal["K"], [[expected_f, 0, 128], [0, expected_f, 128], [0, 0, 1]])
            self.assertEqual(cal["image_size"], [256, 256])
        with self.assertRaisesRegex(ValueError, "missing_camera"):
            rig.calibration(env, "missing_camera")

    def test_axes_are_right_handed_and_look_toward_target(self):
        eye, target = np.array([0.1, 0.2, 0.3]), np.array([-0.1, 0.1, 0.8])
        r = rig.axes(eye, target)
        self.assert_close(r.T @ r, np.eye(3))
        self.assert_close(np.linalg.det(r), 1)
        self.assert_close(-r[:, 2], (target - eye) / np.linalg.norm(target - eye))
        with self.assertRaisesRegex(ValueError, "parallel"):
            rig.axes([0, 1, 0], [0, 0, 0])

    def test_constructor_patch_success_restores_and_records_metadata(self):
        module = fake_exp4_module()
        original = module.Exp4Env.__init__
        profile = deepcopy(rig.PROFILE)
        profile.update(width=96, height=96)
        xml = small_xml()
        previous = sys.modules.get("exp4_env")
        with patch.dict(sys.modules, {"exp4_env": module}):
            with rig.install_for_construction(CONFIG, profile):
                self.assertIsNot(module.Exp4Env.__init__, original)
                env = module.Exp4Env(xml, 17, marker="位置参数")
                keyword_env = module.Exp4Env(xml=xml, marker="关键字参数")
            self.assertIs(module.Exp4Env.__init__, original)
            for instance in (env, keyword_env):
                self.assertEqual(instance.img_size, 96)
                self.assertEqual(instance.camera_rig_original_xml, xml)
                self.assertEqual(instance.camera_rig_xml, instance.xml)
                meta = instance.camera_rig_metadata
                self.assertTrue(meta["physics_validation"]["matched"])
                self.assertEqual(meta["source_xml_sha256"], hashlib.sha256(xml.encode()).hexdigest())
                self.assertEqual(meta["rig_xml_sha256"], hashlib.sha256(instance.xml.encode()).hexdigest())
                self.assertNotEqual(meta["source_xml_sha256"], meta["rig_xml_sha256"])
            self.assertEqual(env.marker, "位置参数")
            self.assertEqual(keyword_env.marker, "关键字参数")
            unpatched = module.Exp4Env(xml)
            self.assertEqual(unpatched.img_size, 64)
            self.assertFalse(hasattr(unpatched, "camera_rig_metadata"))
        self.assertIs(sys.modules.get("exp4_env"), previous)

    def test_constructor_patch_finally_restores_after_constructor_failure(self):
        module = fake_exp4_module(fail=True)
        original = module.Exp4Env.__init__
        with patch.dict(sys.modules, {"exp4_env": module}):
            with self.assertRaisesRegex(RuntimeError, "模拟构造失败"):
                with rig.install_for_construction(CONFIG):
                    module.Exp4Env(small_xml())
            self.assertIs(module.Exp4Env.__init__, original)

    def test_constructor_patch_finally_restores_after_context_body_failure(self):
        module = fake_exp4_module()
        original = module.Exp4Env.__init__
        with patch.dict(sys.modules, {"exp4_env": module}):
            with self.assertRaisesRegex(RuntimeError, "上下文失败"):
                with rig.install_for_construction(CONFIG):
                    raise RuntimeError("上下文失败")
            self.assertIs(module.Exp4Env.__init__, original)

    def test_constructor_patch_finally_restores_after_validation_failure(self):
        for failure in ("non_square", "already_installed", "physics_mismatch"):
            with self.subTest(failure=failure):
                module = fake_exp4_module()
                if failure == "physics_mismatch":
                    base = module.Exp4Env.__init__

                    def corrupt(self, xml, img_size=64, *, marker=None):
                        base(self, xml, img_size, marker=marker)
                        self.m.body_mass[-1] += 0.1

                    module.Exp4Env.__init__ = corrupt
                original = module.Exp4Env.__init__
                profile = deepcopy(rig.PROFILE)
                xml = small_xml()
                if failure == "non_square":
                    profile["height"] = profile["width"] + 1
                elif failure == "already_installed":
                    xml, _ = rig.augment_xml(xml, CONFIG)
                error = AssertionError if failure == "physics_mismatch" else ValueError
                with patch.dict(sys.modules, {"exp4_env": module}):
                    with self.assertRaises(error):
                        with rig.install_for_construction(CONFIG, profile):
                            module.Exp4Env(xml)
                    self.assertIs(module.Exp4Env.__init__, original)

    def test_nested_constructor_contexts_restore_outer_then_original(self):
        module = fake_exp4_module()
        original = module.Exp4Env.__init__
        with patch.dict(sys.modules, {"exp4_env": module}):
            with rig.install_for_construction(CONFIG):
                outer = module.Exp4Env.__init__
                with self.assertRaisesRegex(RuntimeError, "内层失败"):
                    with rig.install_for_construction(CONFIG):
                        self.assertIsNot(module.Exp4Env.__init__, outer)
                        raise RuntimeError("内层失败")
                self.assertIs(module.Exp4Env.__init__, outer)
                env = module.Exp4Env(small_xml())
                self.assertTrue(env.camera_rig_metadata["physics_validation"]["matched"])
            self.assertIs(module.Exp4Env.__init__, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
