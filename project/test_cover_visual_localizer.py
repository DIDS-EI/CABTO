"""使用合成 RGB 图和内存 MuJoCo 模型测试；无需 MLX、渲染器或场景资产。"""
import builtins
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import mujoco
import numpy as np


_original_import = builtins.__import__


def _import_without_mlx(name, *args, **kwargs):
    if name.split(".")[0] in {"mlx", "mlx_vlm", "mlx_lm"}:
        raise AssertionError("轻量测试不应导入 MLX: " + name)
    return _original_import(name, *args, **kwargs)


# 导入真实实现，仅阻止无关的模型依赖，不替换视觉算法或物理引擎。
with patch("builtins.__import__", side_effect=_import_without_mlx):
    from core import cover_visual_localizer as localizer


class RefineRGBTests(unittest.TestCase):
    BACKGROUND = (50, 100, 180)
    RED = (220, 25, 20)
    WOOD = (180, 130, 60)
    CENTER = (160, 140)

    def setUp(self):
        guard = patch("builtins.__import__", side_effect=_import_without_mlx)
        guard.start()
        self.addCleanup(guard.stop)

    def image(self):
        return np.full((280, 340, 3), self.BACKGROUND, dtype=np.uint8)

    def apple(self):
        image = self.image()
        cv2.circle(image, self.CENTER, 20, self.RED, -1)
        return image

    def assert_center(self, actual, expected, tolerance=0.75):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)

    def test_rgb_and_coarse_only_with_explicit_semantic_prior(self):
        signature = inspect.signature(localizer.refine_rgb)
        self.assertEqual(tuple(signature.parameters), ("image", "coarse_uv", "name"))
        self.assertTrue(all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                            for p in signature.parameters.values()))
        with patch.object(localizer, "mujoco", spec=[]), patch.object(
            localizer, "intersect_plane", side_effect=AssertionError("不应访问三维投影")
        ), patch.object(localizer, "Perception", spec=[]):
            refined, _, _ = localizer.refine_rgb(self.apple(), (166, 136), "apple")
        self.assert_center(refined, self.CENTER)
        for extra in ("env", "depth", "segmentation", "geom_id", "object_position"):
            with self.subTest(extra=extra), self.assertRaises(TypeError):
                localizer.refine_rgb(self.apple(), self.CENTER, "apple", **{extra: None})

    def test_rgb_not_bgr_channel_order(self):
        image = self.apple()
        refined, _, _ = localizer.refine_rgb(image, self.CENTER, "apple")
        self.assert_center(refined, self.CENTER)
        with self.assertRaisesRegex(ValueError, "No RGB component"):
            localizer.refine_rgb(image[:, :, ::-1].copy(), self.CENTER, "apple")

    def test_zero_sized_images_rejected(self):
        for shape in ((0, 340, 3), (280, 0, 3), (0, 0, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                localizer.refine_rgb(np.empty(shape, dtype=np.uint8), (0, 0), "apple")

    def test_blank_images_rejected_for_all_priors(self):
        for name in ("apple", "pan", "board", "bowl", "potato", "shrimp"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "No RGB component"):
                localizer.refine_rgb(self.image(), self.CENTER, name)

    def test_out_of_bounds_coarse_rejected(self):
        for point in ((-1, 140), (160, -1), (340, 140), (160, 280), (999, 999)):
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, "outside frame"):
                localizer.refine_rgb(self.apple(), point, "apple")

    def test_nonfinite_coarse_rejected(self):
        for point in ((np.nan, 140), (160, np.nan), (np.inf, 140), (160, -np.inf)):
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, "outside frame"):
                localizer.refine_rgb(self.apple(), point, "apple")

    def test_malformed_coarse_rejected(self):
        for point in ([], [160], [160, 140, 0], [[160, 140]], 160):
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, "outside frame"):
                localizer.refine_rgb(self.apple(), point, "apple")

    def test_unknown_semantic_prior_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown semantic color prior"):
            localizer.refine_rgb(self.apple(), self.CENTER, "unknown")

    def test_red_disc_with_offcenter_black_hole_recovers_center(self):
        image = self.apple()
        hole = (167, 140)
        cv2.circle(image, hole, 7, (0, 0, 0), -1)
        refined, diagnosis, component = localizer.refine_rgb(image, (166, 136), "apple")
        self.assert_center(refined, self.CENTER)
        self.assertEqual(diagnosis["method"], "maximum_inscribed_disc")
        x0, y0, _, _ = diagnosis["roi"]
        self.assertEqual(component[hole[1] - y0, hole[0] - x0], 1)

    def test_round_pan_with_thin_handle_recovers_disc_center(self):
        image = self.image()
        cv2.circle(image, self.CENTER, 32, (25, 25, 25), -1)
        cv2.rectangle(image, (188, 137), (230, 143), (25, 25, 25), -1)
        refined, diagnosis, _ = localizer.refine_rgb(image, (170, 145), "pan")
        self.assert_center(refined, self.CENTER, tolerance=1.0)
        self.assertIn(diagnosis["method"], ("round_edge_hough_center", "maximum_inscribed_disc"))

    def test_wood_rectangle_recovers_center(self):
        image = self.image()
        cv2.rectangle(image, (120, 115), (200, 165), self.WOOD, -1)
        refined, diagnosis, _ = localizer.refine_rgb(image, (173, 132), "board")
        self.assert_center(refined, self.CENTER)
        self.assertEqual(diagnosis["method"], "component_centroid")
        self.assertEqual(diagnosis["component_area"], 81 * 51)

    def test_white_bowl_outer_ellipse_recovers_center(self):
        image = self.image()
        cv2.ellipse(image, self.CENTER, (35, 26), 0, 0, 360, (235, 235, 235), -1)
        cv2.ellipse(image, self.CENTER, (24, 16), 0, 0, 360, self.BACKGROUND, -1)
        refined, diagnosis, _ = localizer.refine_rgb(image, (168, 135), "bowl")
        self.assert_center(refined, self.CENTER)
        self.assertEqual(diagnosis["method"], "white_component_outer_ellipse_center")

    def test_ambiguous_matching_components_are_rejected(self):
        image=self.image()
        cv2.circle(image,(148,140),6,self.RED,-1)
        cv2.circle(image,(172,140),6,self.RED,-1)
        with self.assertRaisesRegex(ValueError,"Ambiguous"):
            localizer.refine_rgb(image,(160,140),"apple")

    def test_nearest_matching_component_beats_larger_distractor(self):
        image = self.image()
        cv2.circle(image, (148, 140), 6, self.RED, -1)
        cv2.circle(image, (182, 140), 10, self.RED, -1)
        refined, _, _ = localizer.refine_rgb(image, (150, 140), "apple")
        self.assert_center(refined, (148, 140))

    def test_multiple_remote_colors_do_not_replace_local_target(self):
        image = self.apple()
        for center, color in (((50, 50), self.RED), ((275, 50), self.WOOD),
                              ((50, 230), (25, 25, 25)), ((280, 225), (240, 240, 240))):
            cv2.circle(image, center, 28, color, -1)
        refined, _, _ = localizer.refine_rgb(image, (166, 136), "apple")
        self.assert_center(refined, self.CENTER)

    def test_only_remote_targets_are_rejected(self):
        image = self.image()
        for center, color in (((50, 50), self.RED), ((275, 50), self.WOOD),
                              ((50, 230), (25, 25, 25)), ((280, 225), (240, 240, 240))):
            cv2.circle(image, center, 18, color, -1)
        for name in ("apple", "board", "pan", "bowl", "potato", "shrimp"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "No RGB component"):
                localizer.refine_rgb(image, self.CENTER, name)

    def test_component_inside_roi_but_too_far_from_coarse_rejected(self):
        image = self.image()
        # 圆完全位于 ROI 内，但圆心距离超过 36 * 0.85。
        cv2.circle(image, (188, 157), 5, self.RED, -1)
        with self.assertRaisesRegex(ValueError, "No RGB component"):
            localizer.refine_rgb(image, self.CENTER, "apple")

    def test_tiny_color_noise_rejected(self):
        image = self.image()
        cv2.circle(image, self.CENTER, 3, self.RED, -1)
        with self.assertRaisesRegex(ValueError, "No RGB component"):
            localizer.refine_rgb(image, self.CENTER, "apple")

    def test_roi_clipped_components_rejected_on_each_edge(self):
        for dx, dy in ((20, 0), (-20, 0), (0, 20), (0, -20)):
            with self.subTest(offset=(dx, dy)):
                image = self.image()
                cv2.circle(image, (160 + dx, 140 + dy), 20, self.RED, -1)
                with self.assertRaisesRegex(ValueError, "crosses ROI border"):
                    localizer.refine_rgb(image, self.CENTER, "apple")

    def test_frame_clipped_components_rejected_on_each_edge(self):
        for center in ((8, 140), (331, 140), (160, 8), (160, 271)):
            with self.subTest(center=center):
                image = self.image()
                cv2.circle(image, center, 20, self.RED, -1)
                with self.assertRaisesRegex(ValueError, "crosses ROI border"):
                    localizer.refine_rgb(image, center, "apple")

    def test_whole_image_translation_preserves_refinement(self):
        for name in ("apple", "pan", "board"):
            image = self.image()
            if name == "apple":
                cv2.circle(image, self.CENTER, 20, self.RED, -1)
                cv2.circle(image, (167, 140), 7, (0, 0, 0), -1)
            elif name == "pan":
                cv2.circle(image, self.CENTER, 32, (25, 25, 25), -1)
                cv2.rectangle(image, (188, 137), (230, 143), (25, 25, 25), -1)
            else:
                cv2.rectangle(image, (120, 115), (200, 165), self.WOOD, -1)
            coarse = np.array((166.25, 136.5))
            original, diagnosis, mask = localizer.refine_rgb(image, coarse, name)
            for delta in ((19, 13), (-17, -11)):
                with self.subTest(name=name, delta=delta):
                    dx, dy = delta
                    moved = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                                           (340, 280), flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=self.BACKGROUND)
                    refined, shifted_diagnosis, shifted_mask = localizer.refine_rgb(
                        moved, coarse + delta, name)
                    self.assert_center(refined - original, delta, tolerance=1e-6)
                    np.testing.assert_array_equal(shifted_mask, mask)
                    np.testing.assert_array_equal(
                        np.array(shifted_diagnosis["roi"]) - diagnosis["roi"],
                        (dx, dy, dx, dy))

    def test_inputs_unchanged_and_diagnostics_consistent(self):
        image = self.apple()
        coarse = np.array((166.0, 136.0))
        before_image, before_coarse = image.copy(), coarse.copy()
        refined, diagnosis, mask = localizer.refine_rgb(image, coarse, "apple")
        np.testing.assert_array_equal(image, before_image)
        np.testing.assert_array_equal(coarse, before_coarse)
        np.testing.assert_array_equal(diagnosis["refined_uv"], refined)
        np.testing.assert_array_equal(diagnosis["coarse_uv"], coarse)
        self.assertEqual(diagnosis["semantic_prior"], "apple")
        x0, y0, x1, y1 = diagnosis["roi"]
        self.assertEqual(mask.shape, (y1 - y0, x1 - x0))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))


class InstallCameraTests(unittest.TestCase):
    def setUp(self):
        # front 不是第一台相机，避免把固定下标写入误判为正确实现。
        self.m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <camera name="side" pos="2 1 2" quat="1 0 0 0" fovy="60"/>
            <camera name="front" pos="1 -1 2" euler="10 20 30" fovy="55"/>
            <geom name="floor" type="plane" size="2 2 0.1" friction="0.7 0.02 0.003"/>
            <body name="object" pos="0.2 0.1 0.8">
              <freejoint/>
              <geom name="box" type="box" size="0.05 0.06 0.07"
                    pos="0.01 0.02 0.03" euler="4 5 6" mass="0.3"
                    rgba="0.7 0.3 0.2 1" friction="0.6 0.03 0.004"/>
            </body>
          </worldbody>
        </mujoco>
        """)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:3] = (0.25, -0.12, 0.9)
        self.d.qvel[:] = np.linspace(0.01, 0.06, self.m.nv)
        self.d.time = 1.25
        mujoco.mj_forward(self.m, self.d)
        self.env = SimpleNamespace(m=self.m, d=self.d)
        self.front = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "front")

    def test_install_changes_only_front_camera_configuration(self):
        before = {}
        for name in dir(self.m):
            if name.startswith("cam_"):
                value = getattr(self.m, name)
                if isinstance(value, np.ndarray):
                    before[name] = value.copy()
        self.assertGreater(self.front, 0)
        self.assertFalse(np.array_equal(before["cam_pos"][self.front], localizer.CAMERA["position"]))
        self.assertFalse(np.array_equal(before["cam_quat"][self.front], localizer.CAMERA["quat"]))
        self.assertNotEqual(before["cam_fovy"][self.front], localizer.CAMERA["fovy"])
        report = localizer.install_camera(self.env)
        expected_changes = {"cam_pos": "position", "cam_quat": "quat", "cam_fovy": "fovy"}
        for name, snapshot in before.items():
            with self.subTest(array=name):
                expected = snapshot.copy()
                if name in expected_changes:
                    expected[self.front] = localizer.CAMERA[expected_changes[name]]
                np.testing.assert_array_equal(getattr(self.m, name), expected)
        for array, key in expected_changes.items():
            np.testing.assert_array_equal(report["before"][key], before[array][self.front])
        self.assertEqual(report["after"], localizer.CAMERA)
        np.testing.assert_allclose(self.d.cam_xpos[self.front], localizer.CAMERA["position"],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.d.cam_xmat[self.front].reshape(3, 3), np.eye(3),
                                   rtol=0, atol=1e-12)

    def test_install_preserves_qpos_qvel_time_and_non_camera_model_arrays(self):
        before = {}
        for name in dir(self.m):
            if not name.startswith("_") and not name.startswith("cam_"):
                value = getattr(self.m, name)
                if isinstance(value, np.ndarray):
                    before[name] = value.copy()
        self.assertTrue({"geom_pos", "geom_quat", "geom_size", "geom_friction",
                         "geom_rgba", "body_mass", "body_inertia"}.issubset(before))
        qpos, qvel, time = self.d.qpos.copy(), self.d.qvel.copy(), self.d.time
        geom_xpos, geom_xmat = self.d.geom_xpos.copy(), self.d.geom_xmat.copy()
        localizer.install_camera(self.env)
        np.testing.assert_array_equal(self.d.qpos, qpos)
        np.testing.assert_array_equal(self.d.qvel, qvel)
        self.assertEqual(self.d.time, time)
        np.testing.assert_array_equal(self.d.geom_xpos, geom_xpos)
        np.testing.assert_array_equal(self.d.geom_xmat, geom_xmat)
        for name, snapshot in before.items():
            with self.subTest(array=name):
                np.testing.assert_array_equal(getattr(self.m, name), snapshot)

    def test_install_is_idempotent(self):
        localizer.install_camera(self.env)
        qpos = self.d.qpos.copy()
        report = localizer.install_camera(self.env)
        for array, key in (("cam_pos", "position"), ("cam_quat", "quat"), ("cam_fovy", "fovy")):
            np.testing.assert_array_equal(getattr(self.m, array)[self.front], localizer.CAMERA[key])
            np.testing.assert_array_equal(report["before"][key], localizer.CAMERA[key])
        np.testing.assert_array_equal(self.d.qpos, qpos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
