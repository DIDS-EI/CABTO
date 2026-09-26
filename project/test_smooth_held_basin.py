"""Smooth held-basin 的轻量回归测试：无 LLM、无长仿真。"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
FORMAL_ROOT = ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"
STAGE2_ROOT = ROOT.parent / "CABTO/exp4_bt_tasks/stage2_scripted"
sys.path.insert(0, str(FORMAL_ROOT))
sys.path.insert(0, str(STAGE2_ROOT))
sys.path.insert(0, str(ROOT))

from core.held_basin_pour_bridge import HeldBasinRuntime
import core.smooth_held_basin_bridge as smooth
from core.smooth_held_basin_bridge import SmoothHeldBasinRuntime, domain, goal
from exp4_env import Exp4Env, HOME_QPOS
from formal_bt import ModelLibrary, action_id, symbolic_dry_run, validate_model
from probe_feasible_curriculum import build_scene
import probe_held_basin_collision as collision
from probe_held_basin_collision import Audit
from probe_smooth_held_basin_reference import direct_return


V2_PATH = ROOT / "scenes/held_basin_pour_smooth_v2.json"
V1_PATH = ROOT / "test_fixtures/held_basin_pour_v1.json"


def load(path):
    return json.loads(path.read_text())


def model_by(name):
    matches = [model for model in domain() if model["name"] == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one model named {name}, got {len(matches)}")
    return matches[0]


def xyz(node):
    return np.asarray(list(map(float, node.get("pos").split())))


class V2SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load(V2_PATH)
        cls.v1 = load(V1_PATH)
        cls.xml, cls.names = build_scene(cls.config, "pour")
        cls.root = ET.fromstring(cls.xml)

    def test_v2_version_and_free_objects(self):
        self.assertEqual(self.config["version"], "held_basin_pour_smooth_v2")
        self.assertEqual(self.names, ["canL", "ballL", "basin"])

    def test_left_base_consumes_distinct_v2_xy(self):
        node = self.root.find("./worldbody/body[@name='link0_L']")
        np.testing.assert_allclose(xyz(node)[:2], [0.18, 0.60])
        self.assertNotEqual(self.config["robots"][0]["base_xy"], self.v1["robots"][0]["base_xy"])

    def test_right_base_consumes_distinct_v2_xy(self):
        node = self.root.find("./worldbody/body[@name='link0_R']")
        np.testing.assert_allclose(xyz(node)[:2], [0.56, -0.60])
        self.assertNotEqual(self.config["robots"][1]["base_xy"], self.v1["robots"][1]["base_xy"])

    def test_v2_bases_are_offset_in_both_x_and_y(self):
        left, right = (np.asarray(row["base_xy"]) for row in self.config["robots"])
        self.assertNotEqual(left[0], right[0])
        self.assertNotEqual(left[1], right[1])

    def test_basin_radius_is_consumed_by_bottom_geometry(self):
        basin = self.root.find("./worldbody/body[@name='basin']")
        bottom = basin.find("geom[@type='cylinder']")
        radius = float(bottom.get("size").split()[0])
        b = self.config["pour"]["held_basin"]
        self.assertAlmostEqual(radius, b["basin_inner_radius"] + b["basin_wall"])
        self.assertAlmostEqual(b["basin_inner_radius"], 0.15)

    def test_basin_handle_offset_is_consumed_by_body_origin(self):
        basin = self.root.find("./worldbody/body[@name='basin']")
        b = self.config["pour"]["held_basin"]
        center = xyz(basin)[:2] - [b["basin_handle_offset"], 0.0]
        np.testing.assert_allclose(center, b["basin_start_center_xy"])
        self.assertAlmostEqual(b["basin_handle_offset"], 0.27)

    def test_basin_handle_bridge_consumes_radius_and_offset(self):
        basin = self.root.find("./worldbody/body[@name='basin']")
        handles = [g for g in basin.findall("geom") if g.get("material") == "mat_carton"]
        self.assertEqual(len(handles), 3)
        bridge = next(g for g in handles if np.isclose(float(g.get("size").split()[1]), .009)
                      and np.isclose(float(g.get("size").split()[2]), .009))
        b = self.config["pour"]["held_basin"]
        length = b["basin_handle_offset"] - b["basin_inner_radius"]
        self.assertAlmostEqual(float(bridge.get("size").split()[0]), length / 2)
        self.assertAlmostEqual(float(bridge.get("pos").split()[0]), -length / 2)

    def test_source_cup_wall_consumes_configured_height(self):
        can = self.root.find("./worldbody/body[@name='canL']")
        wall_half_heights = [float(g.get("size").split()[2]) for g in can.findall("geom")
                             if g.get("type") == "box" and g.get("material") == "mat_red"
                             and len(g.get("size").split()) == 3
                             and np.isclose(float(g.get("size").split()[0]), .002)
                             and np.isclose(float(g.get("size").split()[2]), .04)]
        self.assertEqual(len(wall_half_heights), 32)
        self.assertTrue(all(np.isclose(2 * value, self.config["pour"]["can_height"])
                            for value in wall_half_heights))

    def test_source_transport_height_is_v2_value(self):
        runtime = HeldBasinRuntime.__new__(HeldBasinRuntime)
        runtime.b = self.config["pour"]["held_basin"]
        target = runtime._source_target()
        self.assertAlmostEqual(target[2], 0.76)
        self.assertNotEqual(target[2], self.v1["pour"]["held_basin"]["source_transport_z"])

    def test_building_v2_does_not_change_v1_file(self):
        before = V1_PATH.read_bytes()
        build_scene(deepcopy(self.config), "pour")
        self.assertEqual(V1_PATH.read_bytes(), before)

    def test_v1_legacy_values_remain_intact(self):
        self.assertEqual(self.v1["version"], "held_basin_pour_v1")
        self.assertEqual(self.v1["robots"][0]["base_xy"], [0.3, 0.55])
        self.assertEqual(self.v1["robots"][1]["base_xy"], [0.3, -0.55])
        self.assertEqual(self.v1["pour"]["held_basin"]["basin_handle_offset"], 0.16)
        self.assertEqual(self.v1["pour"]["held_basin"]["basin_inner_radius"], 0.13)

    def test_real_v2_scene_compiles_headless_without_stepping(self):
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("编译测试不应推进仿真")):
            env = Exp4Env(
                self.xml,
                [(row["suffix"], row["yaw"]) for row in self.config["robots"]],
                self.names,
                render=False,
            )
        self.assertFalse(env.render_enabled)
        self.assertIsNone(env.renderer)
        self.assertEqual(env.obj_names, ["canL", "ballL", "basin"])


class FormalModelTests(unittest.TestCase):
    def test_all_ten_models_validate(self):
        models = domain()
        self.assertEqual(len(models), 10)
        self.assertEqual([validate_model(model) for model in models],
                         [action_id(model) for model in models])

    def test_ten_models_have_unique_grounded_ids(self):
        ids = [action_id(model) for model in domain()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_formal_bt_dry_run_reaches_complete_goal_in_exactly_ten_steps(self):
        start = {
            "empty(0)", "empty(1)", "at_source(canL)", "at_source(basin)",
            "content_in_source()", "source_returned()", "basin_returned()",
            "home(0)", "home(1)", "hand_raised(0)", "hand_raised(1)",
            "source_upright()", "collision_free()",
        }
        optional = {}
        try:
            import py_trees  # noqa: F401
        except ModuleNotFoundError:
            optional["py_trees"] = SimpleNamespace(common=SimpleNamespace(Status=object))
        try:
            import tabulate  # noqa: F401
        except ModuleNotFoundError:
            optional["tabulate"] = SimpleNamespace(tabulate=lambda *_args, **_kwargs: "")
        with patch.dict(sys.modules, optional):
            report = symbolic_dry_run(ModelLibrary(domain()).build(start, goal()), start, goal(), max_steps=10)
        self.assertTrue(report["reached_goal"], report)
        self.assertEqual(report["steps"], 10)
        self.assertEqual(report["scope"], "symbolic_only")
        self.assertTrue(goal() <= set(report["final_state"]))

    def test_collision_free_is_a_precondition_of_every_action(self):
        self.assertTrue(all("collision_free()" in model["pre"] for model in domain()))

    def test_pick_source_deletes_initial_returned_home_and_raised(self):
        self.assertTrue({"source_returned()", "home(0)", "hand_raised(0)"}
                        <= set(model_by("pick_source_cup")["del"]))

    def test_pick_basin_deletes_initial_returned_home_and_raised(self):
        self.assertTrue({"basin_returned()", "home(1)", "hand_raised(1)"}
                        <= set(model_by("pick_basin")["del"]))

    def test_pour_deletes_source_upright(self):
        self.assertIn("source_upright()", model_by("pour_into_held_basin")["del"])

    def test_source_return_depends_on_restored_upright(self):
        self.assertIn("source_upright()", model_by("return_source_cup")["pre"])

    def test_source_home_depends_on_source_return(self):
        home = model_by("home_source_arm")
        self.assertTrue({"source_returned()", "empty(0)", "hand_raised(0)"} <= set(home["pre"]))

    def test_basin_return_depends_on_source_home(self):
        self.assertIn("home(0)", model_by("return_basin")["pre"])

    def test_receiver_home_depends_on_basin_return_and_source_home(self):
        pre = set(model_by("home_receiver_arm")["pre"])
        self.assertTrue({"basin_returned()", "home(0)", "empty(1)", "hand_raised(1)"} <= pre)

    def test_final_goal_has_all_and_only_required_predicates(self):
        self.assertEqual(goal(), {
            "source_returned()", "basin_returned()", "empty(0)", "empty(1)",
            "home(0)", "home(1)", "sphere_in_basin()", "caught_by_held_basin()",
            "payload_retained()", "collision_free()",
        })


class AuditTests(unittest.TestCase):
    @staticmethod
    def sample_pair(a, b, distance=-0.001):
        names = {0: a, 1: b}
        contact = SimpleNamespace(geom1=0, geom2=1, dist=distance)
        runtime = SimpleNamespace(env=SimpleNamespace(m=object(), d=SimpleNamespace(contact=[contact], time=1.25)))
        audit = Audit(runtime)
        with patch.object(collision, "body_name", side_effect=lambda _m, gid: names[gid]):
            audit.sample()
        return audit

    def test_audit_classifies_inter_arm_contact(self):
        audit = self.sample_pair("hand_L", "finger_R")
        self.assertEqual([row["kind"] for row in audit.rows], ["inter_arm"])

    def test_audit_classifies_left_arm_basin_contact(self):
        audit = self.sample_pair("link3_L", "basin")
        self.assertEqual([row["kind"] for row in audit.rows], ["left_arm_basin"])

    def test_audit_classifies_right_arm_source_contact(self):
        audit = self.sample_pair("finger_R", "canL")
        self.assertEqual([row["kind"] for row in audit.rows], ["right_arm_source"])

    def test_audit_classifies_container_container_contact(self):
        audit = self.sample_pair("canL", "basin")
        self.assertEqual([row["kind"] for row in audit.rows], ["container_container"])

    def test_audit_ignores_left_arm_own_source_contact(self):
        self.assertEqual(self.sample_pair("finger_L", "canL").rows, [])

    def test_audit_ignores_right_arm_own_basin_contact(self):
        self.assertEqual(self.sample_pair("hand_R", "basin").rows, [])

    def test_audit_ignores_normal_sphere_basin_contact(self):
        self.assertEqual(self.sample_pair("ballL", "basin").rows, [])

    def test_negative_penetration_history_keeps_collision_free_false(self):
        runtime = SmoothHeldBasinRuntime.__new__(SmoothHeldBasinRuntime)
        runtime.b = {"forbidden_penetration_tolerance_m": 0.0002}
        runtime.audit = SimpleNamespace(rows=[{"distance_m": -0.0003}])
        self.assertFalse(runtime.collision_free())
        runtime.audit.rows.append({"distance_m": 0.01})
        self.assertFalse(runtime.collision_free())


class ReturnGeometryTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SmoothHeldBasinRuntime.__new__(SmoothHeldBasinRuntime)
        self.runtime.h = .4
        self.runtime.b = {"return_rotation_tolerance_deg": 12.0}
        self.runtime.initial = {"canL": np.array([.18, .27, .522]),
                                "basin": np.array([.60, -.27, .522])}
        self.runtime.initial_quat = {"canL": np.array([1., 0., 0., 0.]),
                                     "basin": np.array([1., 0., 0., 0.])}
        self.poses = {name: [pos.copy(), np.array([1., 0., 0., 0.])]
                      for name, pos in self.runtime.initial.items()}
        self.runtime.env = SimpleNamespace(
            get_object_pose=lambda name: tuple(value.copy() for value in self.poses[name]),
            obj_bid={"canL": 0, "basin": 1},
            m=object(),
            d=SimpleNamespace(xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1))),
        )
        self.released = {"canL": True, "basin": True}
        self.runtime.scout = SimpleNamespace(released=lambda name: self.released[name])

    def geometry(self, obj, bottom=.4):
        corners = np.array([[0., 0., bottom], [1., 1., bottom + .1]])
        with patch.object(smooth, "geom_corners", return_value=corners):
            return self.runtime.return_geometry(obj)

    def test_return_geometry_accepts_xy_bottom_full_rotation_and_release(self):
        result = self.geometry("canL")
        self.assertTrue(result["returned"], result)
        self.assertEqual(result["rotation_error_deg"], 0.0)
        self.assertTrue(result["released"])

    def test_return_geometry_rejects_bad_xy(self):
        self.poses["canL"][0][0] += .021
        result = self.geometry("canL")
        self.assertGreater(result["xy_error_m"], .020)
        self.assertFalse(result["returned"])

    def test_return_geometry_rejects_bad_bottom_height(self):
        result = self.geometry("basin", bottom=.411)
        self.assertAlmostEqual(result["bottom_gap_m"], .011)
        self.assertFalse(result["returned"])

    def test_return_geometry_rejects_full_yaw_rotation_error(self):
        angle = math.radians(13)
        self.poses["basin"][1] = np.array([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
        result = self.geometry("basin")
        self.assertAlmostEqual(result["rotation_error_deg"], 13.0)
        self.assertFalse(result["returned"])

    def test_return_geometry_rejects_unreleased_object(self):
        self.released["canL"] = False
        result = self.geometry("canL")
        self.assertFalse(result["released"])
        self.assertFalse(result["returned"])


class HomeTests(unittest.TestCase):
    def make_runtime(self):
        runtime = SmoothHeldBasinRuntime.__new__(SmoothHeldBasinRuntime)
        qpos = np.concatenate([HOME_QPOS.copy(), HOME_QPOS.copy()])
        arms = []
        for i in (0, 1):
            arm = SimpleNamespace(arm_qadr=np.arange(i * 7, i * 7 + 7), _grip_cmd=1.0)
            arms.append(arm)
        runtime.env = SimpleNamespace(arms=arms, d=SimpleNamespace(qpos=qpos))
        runtime.home_dirty = [False, False]; runtime.home_events = set()
        released = {"canL": True, "basin": False}
        runtime.scout = SimpleNamespace(released=lambda name: released[name])
        return runtime, released

    def test_home_only_requires_this_arm_object_released(self):
        runtime, released = self.make_runtime()
        self.assertFalse(released["basin"])
        self.assertTrue(runtime.home(0))

    def test_home_rejects_this_arm_object_still_held(self):
        runtime, released = self.make_runtime()
        released["canL"] = False
        self.assertFalse(runtime.home(0))

    def test_home_requires_open_gripper(self):
        runtime, _ = self.make_runtime()
        runtime.env.arms[0]._grip_cmd = 0.0
        self.assertFalse(runtime.home(0))

    def test_home_requires_actual_qpos_near_home(self):
        runtime, released = self.make_runtime()
        released["basin"] = True
        runtime.env.d.qpos[7] += .11
        self.assertFalse(runtime.home(1))

    def test_dirty_home_requires_controlled_home_event(self):
        runtime, _ = self.make_runtime()
        runtime.home_dirty[0] = True
        self.assertFalse(runtime.home(0))
        runtime.home_events.add(0)
        self.assertTrue(runtime.home(0))


class ControlledHomeTests(unittest.TestCase):
    def make_runtime(self):
        runtime = SmoothHeldBasinRuntime.__new__(SmoothHeldBasinRuntime)
        runtime.b = {"home_cartesian": [[.22, .34, .76], [.56, -.34, .76]],
                     "controlled_home_steps": 4}
        runtime.h = .4
        runtime.return_events = []; runtime.home_events = set(); runtime.home_dirty = [True, False]
        qpos = np.linspace(-1., 1., 14)
        start_qpos = qpos.copy()
        arm = SimpleNamespace(arm_target=HOME_QPOS + .4, arm_qadr=np.arange(7),
                              is_holding=Mock(return_value=False))
        skill = SimpleNamespace(move_to=Mock(return_value=.0), _tick=Mock())
        runtime.env = SimpleNamespace(arms=[arm], d=SimpleNamespace(qpos=qpos, time=2.0),
                                      step_joint=Mock())
        runtime.scout = SimpleNamespace(released=Mock(return_value=True), skills=[skill])
        runtime.home = Mock(return_value=True)
        return runtime, arm, start_qpos

    def test_controlled_home_uses_step_joint_without_writing_qpos(self):
        runtime, _, before = self.make_runtime()
        runtime._controlled_home(0)
        np.testing.assert_array_equal(runtime.env.d.qpos, before)
        self.assertEqual(runtime.env.step_joint.call_count, 4 + 12)

    def test_controlled_home_interpolation_reaches_home_endpoint(self):
        runtime, arm, _ = self.make_runtime()
        start = arm.arm_target.copy()
        runtime._controlled_home(0)
        first = runtime.env.step_joint.call_args_list[0].args[1]
        endpoint = runtime.env.step_joint.call_args_list[3].args[1]
        np.testing.assert_allclose(first, start + (HOME_QPOS - start) / 4)
        np.testing.assert_allclose(endpoint, HOME_QPOS)
        np.testing.assert_allclose(runtime.env.step_joint.call_args_list[-1].args[1], HOME_QPOS)

    def test_controlled_home_rejects_unreleased_task_object(self):
        runtime, _, _ = self.make_runtime()
        runtime.scout.released.return_value = False
        with self.assertRaisesRegex(RuntimeError, "requires this arm to release"):
            runtime._controlled_home(0)
        runtime.env.step_joint.assert_not_called()


class RestrictedReturnApiTests(unittest.TestCase):
    def make_runtime(self):
        config = load(V2_PATH)
        runtime = SmoothHeldBasinRuntime.__new__(SmoothHeldBasinRuntime)
        runtime.p = config["pour"]
        runtime.b = runtime.p["held_basin"]
        runtime.h = config["table"]["top_z"]
        runtime.initial = {"canL": np.array([.18, .27, .522]),
                           "basin": np.array([.60, -.27, .522])}
        runtime.audit = SimpleNamespace(phase="initial")
        arms, skills = [], []
        for _ in (0, 1):
            arm = SimpleNamespace(is_holding=Mock(return_value=False), release_all=Mock(),
                                  bias_world=np.zeros(3), arm_target=HOME_QPOS.copy(),
                                  _grasp_enabled=True, s=("_L" if len(arms)==0 else "_R"))
            skill = SimpleNamespace(
                ee=Mock(return_value=np.array([.3, 0., .7])),
                move_to=Mock(return_value=.0), refine_above=Mock(return_value=np.zeros(3)),
                record_grab_offset=Mock(), block_half=.04,
                descend_place_tracked=Mock(return_value=.0), place_release=Mock(), _tick=Mock(),
            )
            arms.append(arm)
            skills.append(skill)
        runtime.env = SimpleNamespace(arms=arms, d=SimpleNamespace(time=0.0), step_joint=Mock(), hold_arms=Mock())
        runtime.scout = SimpleNamespace(skills=skills, hold=Mock(), released=Mock(return_value=True))
        runtime._go_object_origin = Mock(return_value=.0)
        runtime._move_object_slow = Mock(return_value=.0)
        runtime.return_geometry = Mock(return_value={"returned": False})
        return runtime

    def test_return_api_exposes_only_atomic_return_object(self):
        runtime = self.make_runtime()
        self.assertEqual(set(runtime.api(model_by("return_source_cup"))), {"return_object"})
        self.assertEqual(set(runtime.api(model_by("return_basin"))), {"return_object"})

    def test_inherited_target_api_rejects_forged_present_target(self):
        runtime = self.make_runtime()
        runtime.scout.transport = Mock()
        api = runtime.api(model_by("present_basin"))
        target = api["target_pose"]()
        with self.assertRaisesRegex(ValueError, "target_pose"):
            api["move_basin"]((target[0], target[1] + 1e-3, target[2]))
        runtime.scout.transport.assert_not_called()

    def test_source_return_object_consumes_preoffset(self):
        runtime = self.make_runtime()
        runtime.api(model_by("return_source_cup"))["return_object"]()
        expected = (np.asarray(runtime.b["source_drop_center_xy"]) +
                    np.array([runtime.p["handle_offset"], 0.0]) +
                    runtime.b["source_return_preoffset_xy"])
        second_target = runtime._go_object_origin.call_args_list[1].args[2]
        np.testing.assert_allclose(second_target[:2], expected)

    def test_basin_return_object_consumes_preoffset(self):
        runtime = self.make_runtime()
        runtime.api(model_by("return_basin"))["return_object"]()
        expected = runtime.initial["basin"][:2] + runtime.b["basin_return_preoffset_xy"]
        second_target = runtime._move_object_slow.call_args_list[1].args[2]
        np.testing.assert_allclose(second_target[:2], expected)

    def test_atomic_return_rejects_unreleased_grasp_before_retreat(self):
        runtime = self.make_runtime()
        runtime.env.arms[0].is_holding.return_value = True
        with self.assertRaisesRegex(RuntimeError, "object weld remained active|release object before retreat"):
            runtime.api(model_by("return_source_cup"))["return_object"]()

    def test_reference_return_program_calls_atomic_return(self):
        call = Mock()
        direct_return({"return_object": call})
        call.assert_called_once_with()


class FinalGoalTests(unittest.TestCase):
    def test_stable_goal_accepts_only_complete_goal_with_sphere_still_in_basin(self):
        runtime = SmoothHeldBasinRuntime.__new__(SmoothHeldBasinRuntime)
        runtime.goal = goal()
        runtime.audit = SimpleNamespace(phase="initial")
        runtime.state = Mock(return_value=set(goal()))
        runtime.scout = SimpleNamespace(stable=lambda predicate: predicate(),
                                        sphere_in_basin=Mock(return_value=True))
        self.assertTrue(runtime.stable_goal())
        self.assertEqual(runtime.audit.phase, "final_stability")

    def test_stable_goal_rejects_each_missing_final_requirement(self):
        for missing in goal():
            with self.subTest(missing=missing):
                runtime = SmoothHeldBasinRuntime.__new__(SmoothHeldBasinRuntime)
                runtime.goal = goal()
                runtime.audit = SimpleNamespace(phase="initial")
                runtime.state = Mock(return_value=set(goal()) - {missing})
                runtime.scout = SimpleNamespace(stable=lambda predicate: predicate(),
                                                sphere_in_basin=Mock(return_value=True))
                self.assertFalse(runtime.stable_goal())


if __name__ == "__main__":
    unittest.main(verbosity=2)
