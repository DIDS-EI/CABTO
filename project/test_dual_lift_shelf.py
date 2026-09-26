"""双臂抬自由托盘上架的短时单元测试。

只用 build_scene、判定 stub 和 render=False 的真实 MuJoCo 初始化；不运行 LLM、
不执行长 rollout，也不把 qpos 写入当作执行证据。
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from probe_dual_lift_shelf import DualLiftShelfScout
from probe_dual_lift_codegen import collision_signature, physical_config
from probe_feasible_curriculum import build_scene


ROOT = Path(__file__).resolve().parent
NEW_CONFIG_PATH = ROOT / "scenes/packing_dual_lift_shelf_storage_v3.json"
VISUAL_V2_CONFIG_PATH = ROOT / "test_fixtures/storage/packing_dual_lift_shelf_storage_v2.json"
LEGACY_CONFIG_PATH = ROOT / "test_fixtures/storage/packing_dual_lift_shelf_v1.json"
FREE_CONFIG_PATH = ROOT / "test_fixtures/storage/packing_free_tray_v1.json"
FROZEN_CONFIG_PATH = ROOT / "test_fixtures/storage/feasible_dual_v1.json"
FREE_REFERENCE_XML = ROOT / "test_fixtures/storage/free_reference.xml"
FROZEN_REFERENCE_XML = ROOT / "test_fixtures/storage/frozen_reference.xml"

NEW_CONFIG = json.loads(NEW_CONFIG_PATH.read_text())
FREE_CONFIG = json.loads(FREE_CONFIG_PATH.read_text())
FROZEN_CONFIG = json.loads(FROZEN_CONFIG_PATH.read_text())
PACKING = NEW_CONFIG["packing"]
ARMS = [("_L", -np.pi / 2), ("_R", np.pi / 2)]
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _packing_xml(config):
    return build_scene(deepcopy(config), "packing")


def _tray_node(xml):
    return ET.fromstring(xml).find("./worldbody/body[@name='tray']")


def _portable_xml_hash(xml):
    """Hash scene semantics without the machine-specific absolute mesh directory."""
    root=ET.fromstring(xml)
    root.find("compiler").set("meshdir","<PANDA_ASSETS>")
    return hashlib.sha256(ET.tostring(root,encoding="utf-8")).hexdigest()


def _quat_z(degrees):
    radians = math.radians(degrees)
    return np.array([math.cos(radians / 2), 0.0, 0.0, math.sin(radians / 2)])


def _shelf_scout(pos=None, quat=None, support=True, welds=(False, False), grips=(1.0, 1.0)):
    scout = DualLiftShelfScout.__new__(DualLiftShelfScout)
    scout.p = deepcopy(PACKING)
    state = {
        "pos": np.asarray(pos if pos is not None else [*PACKING["shelf_xy"], PACKING["shelf_top_z"]], float),
        "quat": np.asarray(quat if quat is not None else IDENTITY, float),
        "support": support,
        "welds": list(welds),
    }
    scout.tray_pose = lambda: (state["pos"].copy(), state["quat"].copy())
    scout.tray_supported_by_shelf = lambda: state["support"]
    scout.tray_weld_active = lambda i: state["welds"][i]
    scout.arms = [SimpleNamespace(_grip_cmd=float(g)) for g in grips]
    return scout, state


def _goal_scout(outside_item):
    scout = DualLiftShelfScout.__new__(DualLiftShelfScout)
    scout.p = deepcopy(PACKING)
    tray_pos = np.array([*PACKING["shelf_xy"], PACKING["shelf_top_z"]], float)
    floor_center_z = tray_pos[2] + PACKING["tray_wall"] + PACKING["item_half_size"][2]
    poses = {
        "tray": (tray_pos, IDENTITY.copy()),
        "item0": (tray_pos + np.array([0.0, 0.025, floor_center_z - tray_pos[2]]), IDENTITY.copy()),
        "item1": (tray_pos + np.array([0.0, -0.025, floor_center_z - tray_pos[2]]), IDENTITY.copy()),
    }
    # 物体中心仍在盘内，但旋转后的一个或多个局部角点越过 y 内边界。
    poses[outside_item] = (
        tray_pos + np.array([0.0, 0.060, floor_center_z - tray_pos[2]]),
        _quat_z(45),
    )
    scout.tray_pose = lambda: (poses["tray"][0].copy(), poses["tray"][1].copy())
    scout.env = SimpleNamespace(
        get_object_pose=lambda name: (poses[name][0].copy(), poses[name][1].copy())
    )
    scout.tray_on_shelf = lambda: True
    scout.released = lambda name: True
    return scout


class SceneConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml, cls.names = _packing_xml(NEW_CONFIG)
        cls.root = ET.fromstring(cls.xml)
        cls.tray = cls.root.find("./worldbody/body[@name='tray']")

    def test_new_scene_has_free_tray(self):
        self.assertIn("tray", self.names)
        self.assertIsNotNone(self.tray)
        self.assertIsNotNone(self.tray.find("./freejoint[@name='tray_free']"))

    def test_new_scene_has_two_arm_specific_inactive_tray_welds(self):
        equalities = self.root.find("equality")
        expected = {
            "grasp_weld_L_tray": "hand_L",
            "grasp_weld_R_tray": "hand_R",
        }
        for name, hand in expected.items():
            weld = equalities.find(f"./weld[@name='{name}']")
            self.assertIsNotNone(weld)
            self.assertEqual(weld.get("body1"), hand)
            self.assertEqual(weld.get("body2"), "tray")
            self.assertEqual(weld.get("active"), "false")

    def test_left_handle_and_joint_are_physically_contiguous(self):
        handle = self.tray.find("./geom[@name='tray_handle_L']")
        joint = self.tray.find("./geom[@name='tray_handle_joint_L']")
        self.assertIsNotNone(handle)
        self.assertIsNotNone(joint)
        hp, hs = np.fromstring(handle.get("pos"), sep=" "), np.fromstring(handle.get("size"), sep=" ")
        jp, js = np.fromstring(joint.get("pos"), sep=" "), np.fromstring(joint.get("size"), sep=" ")
        self.assertAlmostEqual(jp[1] + js[1], hp[1] - hs[1], places=9)
        self.assertAlmostEqual(jp[1] - js[1], PACKING["tray_inner_half_size"][1] + PACKING["tray_wall"], places=9)

    def test_right_handle_and_joint_are_physically_contiguous(self):
        handle = self.tray.find("./geom[@name='tray_handle_R']")
        joint = self.tray.find("./geom[@name='tray_handle_joint_R']")
        self.assertIsNotNone(handle)
        self.assertIsNotNone(joint)
        hp, hs = np.fromstring(handle.get("pos"), sep=" "), np.fromstring(handle.get("size"), sep=" ")
        jp, js = np.fromstring(joint.get("pos"), sep=" "), np.fromstring(joint.get("size"), sep=" ")
        self.assertAlmostEqual(jp[1] - js[1], hp[1] + hs[1], places=9)
        self.assertAlmostEqual(jp[1] + js[1], -(PACKING["tray_inner_half_size"][1] + PACKING["tray_wall"]), places=9)

    def test_shelf_is_fixed_and_top_matches_config(self):
        shelf = self.root.find("./worldbody/body[@name='shelf']")
        self.assertIsNotNone(shelf)
        self.assertIsNone(shelf.find("joint"))
        self.assertIsNone(shelf.find("freejoint"))
        geom = shelf.find("geom[@name='shelf_collision_core']")
        if geom is None:geom=shelf.find("geom")
        body_z = np.fromstring(shelf.get("pos"), sep=" ")[2]
        geom_z = np.fromstring(geom.get("pos"), sep=" ")[2]
        half_z = np.fromstring(geom.get("size"), sep=" ")[2]
        self.assertAlmostEqual(body_z + geom_z + half_z, PACKING["shelf_top_z"], places=9)

    def test_total_tray_geom_mass_matches_config(self):
        masses = [float(g.get("mass")) for g in self.tray.findall("geom")]
        self.assertTrue(masses)
        self.assertTrue(all(m > 0.0 for m in masses))
        self.assertAlmostEqual(sum(masses), PACKING["tray_mass"], places=6)

    def test_storage_style_has_green_toy_brick_details(self):
        item0 = self.root.find("./worldbody/body[@name='item0']")
        item1 = self.root.find("./worldbody/body[@name='item1']")
        self.assertIsNotNone(item0.find("./geom[@name='item0_collision_core']"))
        self.assertEqual(len([g for g in item0.findall("geom") if "stud" in g.get("name", "")]), 2)
        self.assertIsNotNone(item1.find("./geom[@name='item1_hinge_bar']"))
        self.assertEqual(len([g for g in item1.findall("geom") if "stud" in g.get("name", "")]), 2)
        self.assertTrue(all(g.get("material", "").startswith("mat_toy_green") for body in (item0,item1) for g in body.findall("geom")))

    def test_cardboard_box_has_physical_connected_flaps_tape_and_label(self):
        for side in ("left","right"):
            flap=self.tray.find(f"./geom[@name='carton_flap_{side}']")
            hinge=self.tray.find(f"./geom[@name='carton_hinge_{side}']")
            self.assertIsNotNone(flap);self.assertIsNotNone(hinge)
            self.assertGreater(float(flap.get("mass")),0);self.assertGreater(float(hinge.get("mass")),0)
            self.assertNotEqual(flap.get("contype"),"0")
        for name in ("carton_tape","carton_label"):
            self.assertIsNotNone(self.tray.find(f"./site[@name='{name}']"))
        self.assertEqual(self.tray.find("./geom[@name='tray_floor']").get("material"), "mat_cardboard_inner")
        self.assertTrue(all(self.tray.find(f"./geom[@name='tray_wall_x_{sign}']").get("material")=="mat_cardboard_outer" for sign in (-1,1)))

    def test_flaps_are_smaller_and_overlap_wall_hinges(self):
        p=PACKING;ix=p["tray_inner_half_size"][0];w=p["tray_wall"]
        for side,sign in (("left",-1),("right",1)):
            flap=self.tray.find(f"./geom[@name='carton_flap_{side}']");hinge=self.tray.find(f"./geom[@name='carton_hinge_{side}']")
            fs=np.fromstring(flap.get("size"),sep=" ");fp=np.fromstring(flap.get("pos"),sep=" ");hp=np.fromstring(hinge.get("pos"),sep=" ")
            self.assertLess(fs[0],.035);self.assertLess(fs[1],p["tray_inner_half_size"][1])
            self.assertAlmostEqual(hp[0],sign*(ix+w),places=9);self.assertAlmostEqual(hp[2],p["tray_height"],places=9)
            self.assertLess(abs(abs(fp[0])-abs(hp[0]))-fs[0],.001)

    def test_box_panels_are_thicker_than_v2(self):
        v2=json.loads(VISUAL_V2_CONFIG_PATH.read_text())
        self.assertGreater(PACKING["tray_wall"],v2["packing"]["tray_wall"])
        self.assertGreater(PACKING["tray_height"],v2["packing"]["tray_height"])

    def test_shelf_is_wood_metal_open_frame(self):
        shelf = self.root.find("./worldbody/body[@name='shelf']")
        self.assertIsNotNone(shelf.find("./geom[@name='shelf_collision_core']"))
        self.assertEqual(shelf.find("./geom[@name='shelf_collision_core']").get("rgba"),"0 0 0 0")
        self.assertIsNotNone(shelf.find("./site[@name='shelf_top']"))
        self.assertIsNotNone(shelf.find("./site[@name='shelf_lower']"))
        self.assertEqual(len([g for g in shelf.findall("site") if g.get("name", "").startswith("shelf_leg_")]), 4)

    def test_v3_physical_box_is_not_mislabeled_visual_only(self):
        v2=json.loads(VISUAL_V2_CONFIG_PATH.read_text())
        self.assertNotEqual(physical_config(NEW_CONFIG),physical_config(v2))
        self.assertNotEqual(collision_signature(NEW_CONFIG),collision_signature(v2))

    def test_initial_bricks_are_intentionally_asymmetric(self):
        p=PACKING;xy=np.asarray(p["sources_xy"],float);yaw=np.asarray(p["source_yaws_deg"],float)
        self.assertGreater(abs(xy[0,0]-xy[1,0]),.05)
        self.assertGreater(abs(abs(xy[0,1])-abs(xy[1,1])),.01)
        self.assertTrue(np.all(np.abs(yaw)>15));self.assertNotAlmostEqual(abs(yaw[0]),abs(yaw[1]))

    def test_building_new_scene_does_not_modify_its_config_file(self):
        before = NEW_CONFIG_PATH.read_bytes()
        _packing_xml(NEW_CONFIG)
        self.assertEqual(before, NEW_CONFIG_PATH.read_bytes())

    def test_old_free_tray_xml_matches_existing_reference_hash(self):
        self.assertTrue(FREE_REFERENCE_XML.is_file())
        config_before = FREE_CONFIG_PATH.read_bytes()
        xml, _ = _packing_xml(FREE_CONFIG)
        self.assertEqual(_portable_xml_hash(xml),_portable_xml_hash(FREE_REFERENCE_XML.read_text()))
        self.assertEqual(config_before, FREE_CONFIG_PATH.read_bytes())

    def test_old_frozen_xml_matches_existing_reference_hash(self):
        self.assertTrue(FROZEN_REFERENCE_XML.is_file())
        config_before = FROZEN_CONFIG_PATH.read_bytes()
        xml, _ = _packing_xml(FROZEN_CONFIG)
        self.assertEqual(_portable_xml_hash(xml),_portable_xml_hash(FROZEN_REFERENCE_XML.read_text()))
        self.assertEqual(config_before, FROZEN_CONFIG_PATH.read_bytes())

    def test_old_scenes_have_no_new_handle_shelf_or_tray_weld_elements(self):
        for config in (FREE_CONFIG, FROZEN_CONFIG):
            with self.subTest(version=config["version"]):
                xml, _ = _packing_xml(config)
                root = ET.fromstring(xml)
                self.assertIsNone(root.find("./worldbody/body[@name='shelf']"))
                self.assertNotIn("tray_handle_", xml)
                self.assertNotIn("grasp_weld_L_tray", xml)
                self.assertNotIn("grasp_weld_R_tray", xml)


class HandleAndFootprintTests(unittest.TestCase):
    def test_handle_world_rotates_left_handle_with_positive_yaw(self):
        scout = DualLiftShelfScout.__new__(DualLiftShelfScout)
        scout.p = deepcopy(PACKING)
        origin = np.array([0.4, -0.2, 0.7])
        scout.tray_pose = lambda: (origin.copy(), _quat_z(90))
        expected = origin + np.array([-PACKING["lift_handle_y"], 0.0, PACKING["lift_handle_z"]])
        np.testing.assert_allclose(scout.handle_world(0), expected, atol=1e-12)

    def test_handle_world_rotates_right_handle_with_positive_yaw(self):
        scout = DualLiftShelfScout.__new__(DualLiftShelfScout)
        scout.p = deepcopy(PACKING)
        origin = np.array([0.4, -0.2, 0.7])
        scout.tray_pose = lambda: (origin.copy(), _quat_z(90))
        expected = origin + np.array([PACKING["lift_handle_y"], 0.0, PACKING["lift_handle_z"]])
        np.testing.assert_allclose(scout.handle_world(1), expected, atol=1e-12)

    def test_footprint_accepts_all_four_corners_on_shelf(self):
        scout, _ = _shelf_scout()
        self.assertTrue(scout.tray_footprint_on_shelf())

    def test_footprint_rejects_when_only_center_is_on_shelf(self):
        scout, _ = _shelf_scout(quat=_quat_z(45))
        self.assertTrue(np.allclose(scout.tray_pose()[0][:2], PACKING["shelf_xy"]))
        self.assertFalse(scout.tray_footprint_on_shelf())


class ShelfPredicateTests(unittest.TestCase):
    def test_tray_on_shelf_accepts_complete_goal_state(self):
        scout, _ = _shelf_scout()
        self.assertTrue(scout.tray_on_shelf())

    def test_tray_on_shelf_requires_xy(self):
        pos = np.array([PACKING["shelf_xy"][0] + 0.026, PACKING["shelf_xy"][1], PACKING["shelf_top_z"]])
        scout, _ = _shelf_scout(pos=pos)
        self.assertFalse(scout.tray_on_shelf())

    def test_tray_on_shelf_requires_z(self):
        pos = np.array([*PACKING["shelf_xy"], PACKING["shelf_top_z"] + 0.013])
        scout, _ = _shelf_scout(pos=pos)
        self.assertFalse(scout.tray_on_shelf())

    def test_tray_on_shelf_checks_full_rotation_including_yaw(self):
        scout, _ = _shelf_scout(quat=_quat_z(PACKING["shelf_tilt_tolerance_deg"] + 1.0))
        self.assertFalse(scout.tray_on_shelf())

    def test_tray_on_shelf_requires_support_contact(self):
        scout, _ = _shelf_scout(support=False)
        self.assertFalse(scout.tray_on_shelf())

    def test_tray_on_shelf_requires_left_weld_inactive(self):
        scout, _ = _shelf_scout(welds=(True, False))
        self.assertFalse(scout.tray_on_shelf())

    def test_tray_on_shelf_requires_right_weld_inactive(self):
        scout, _ = _shelf_scout(welds=(False, True))
        self.assertFalse(scout.tray_on_shelf())

    def test_tray_on_shelf_requires_left_hand_open(self):
        scout, _ = _shelf_scout(grips=(0.0, 1.0))
        self.assertFalse(scout.tray_on_shelf())

    def test_tray_on_shelf_requires_right_hand_open(self):
        scout, _ = _shelf_scout(grips=(1.0, 0.0))
        self.assertFalse(scout.tray_on_shelf())

    def test_support_contact_requires_tray_shelf_pair_and_small_distance(self):
        scout = DualLiftShelfScout.__new__(DualLiftShelfScout)
        scout.env = SimpleNamespace(
            m=SimpleNamespace(geom_bodyid=np.array([0, 1, 2], dtype=int)),
            d=SimpleNamespace(contact=[SimpleNamespace(geom1=0, geom2=1, dist=0.001)]),
        )
        names = {0: "tray", 1: "shelf", 2: "table"}
        with patch("probe_dual_lift_shelf.mujoco.mj_id2name", side_effect=lambda m, kind, bid: names[bid]):
            self.assertTrue(scout.tray_supported_by_shelf())
            scout.env.d.contact = [SimpleNamespace(geom1=0, geom2=1, dist=0.003)]
            self.assertFalse(scout.tray_supported_by_shelf())
            scout.env.d.contact = [SimpleNamespace(geom1=0, geom2=2, dist=0.0)]
            self.assertFalse(scout.tray_supported_by_shelf())

    def test_tray_weld_active_reads_each_arm_specific_equality(self):
        scout = DualLiftShelfScout.__new__(DualLiftShelfScout)
        scout.arms = [SimpleNamespace(obj_weld={"tray": 0}), SimpleNamespace(obj_weld={"tray": 1})]
        scout.env = SimpleNamespace(d=SimpleNamespace(eq_active=np.array([1, 0], dtype=np.uint8)))
        self.assertTrue(scout.tray_weld_active(0))
        self.assertFalse(scout.tray_weld_active(1))


class GoalContainmentTests(unittest.TestCase):
    def test_goal_fails_when_item0_rotated_corner_leaves_tray(self):
        scout = _goal_scout("item0")
        checks = scout.items_contained()
        self.assertFalse(checks["item0"]["inside"])
        self.assertTrue(checks["item1"]["inside"])
        self.assertFalse(scout.goal_now())

    def test_goal_fails_when_item1_rotated_corner_leaves_tray(self):
        scout = _goal_scout("item1")
        checks = scout.items_contained()
        self.assertTrue(checks["item0"]["inside"])
        self.assertFalse(checks["item1"]["inside"])
        self.assertFalse(scout.goal_now())

    def test_goal_requires_each_item_released(self):
        scout = _goal_scout("item0")
        scout.items_contained = lambda: {"item0": {"inside": True}, "item1": {"inside": True}}
        scout.released = lambda name: name != "item1"
        self.assertFalse(scout.goal_now())


class DualMoveResidualTests(unittest.TestCase):
    @staticmethod
    def _scout(positions):
        scout = DualLiftShelfScout.__new__(DualLiftShelfScout)
        arms = []
        for i, pos in enumerate(positions):
            arm = SimpleNamespace(
                s=("_L", "_R")[i],
                bias_world=np.array([0.01, 0.02, 0.03]),
                get_ee_pose=lambda p=np.asarray(pos, float): (p.copy(), IDENTITY.copy()),
            )
            arms.append(arm)
        scout.arms = arms
        scout.env = SimpleNamespace(step_arms=Mock(), record_frame=Mock())
        scout.hold = Mock()
        return scout

    def test_dual_move_rejects_left_residual_over_30mm(self):
        scout = self._scout(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "residuals"):
            scout.dual_move_targets(([0.031, 0.0, 0.0], [1.0, 0.0, 0.0]), grip=0.0, max_steps=0)

    def test_dual_move_rejects_right_residual_over_30mm(self):
        scout = self._scout(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "residuals"):
            scout.dual_move_targets(([0.0, 0.0, 0.0], [1.0, 0.031, 0.0]), grip=0.0, max_steps=0)

    def test_dual_move_accepts_both_residuals_below_30mm_and_restores_bias(self):
        scout = self._scout(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
        before = [arm.bias_world.copy() for arm in scout.arms]
        residuals = scout.dual_move_targets(
            ([0.029, 0.0, 0.0], [1.0, 0.0, 0.029]), grip=0.0, max_steps=0
        )
        self.assertTrue(all(r < 0.030 for r in residuals))
        for arm, expected in zip(scout.arms, before):
            np.testing.assert_array_equal(arm.bias_world, expected)


class RealMujocoInitializationTests(unittest.TestCase):
    def test_new_scene_initializes_headless_without_qpos_execution_edits(self):
        """仅检查 render=False 初态；不写 qpos，不把初态当作任务执行证据。"""
        xml, names = _packing_xml(NEW_CONFIG)
        from exp4_env import Exp4Env

        env = Exp4Env(xml, ARMS, names, render=False)
        env.reset()
        tray_pos, _ = env.get_object_pose("tray")
        self.assertAlmostEqual(float(tray_pos[2]), NEW_CONFIG["table"]["top_z"], delta=0.03)
        self.assertGreaterEqual(mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, "shelf"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_JOINT, "tray_free"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld_L_tray"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld_R_tray"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
