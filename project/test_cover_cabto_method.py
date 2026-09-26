"""Cover CABTO 轻量回归：夹具仅存在于本文件，不加载 MLX、不步进仿真。"""

from copy import deepcopy
from itertools import combinations
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call

import mujoco
import numpy as np

from core.cover_cabto_api import (
    API_DOCS,
    Perception,
    FineCover,
    intersect_plane,
    project,
    transition_check,
)
from core.cover_cabto_grounding import INITIAL, PAIRS, PolicyProgram, state_valid
from policy_codegen import PolicyRejected


class TransitionCheckTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "name": "pick",
            "args": {"object": "shrimp", "support": "bowl"},
            "pre": ["empty()", "at_source(shrimp)"],
            "add": ["holding(shrimp)"],
            "del": ["empty()", "at_source(shrimp)"],
        }
        self.before = set(INITIAL)
        self.after = {"holding(shrimp)", "at_source(apple)", "at_source(potato)"}

    def test_exact_transition_is_accepted(self):
        result = transition_check(self.model, self.before, self.after)
        for key in ("valid_trial", "paper_line20", "strict_CP_transition_match",
                    "goal_ok", "effect_ok"):
            self.assertTrue(result[key], key)
        self.assertEqual(result["violations"], {
            "missing_pre": [], "missing_add": [], "remaining_del": [],
            "missing_preserved_pre": [], "unexpected_added": [], "unexpected_removed": [],
        })

    def test_missing_pre_rejected_even_when_effects_match(self):
        result = transition_check(self.model, self.before - {"empty()"}, self.after)
        self.assertEqual(result["violations"]["missing_pre"], ["empty()"])
        self.assertFalse(result["valid_trial"])
        self.assertTrue(result["strict_CP_transition_match"])
        self.assertTrue(result["goal_ok"])
        self.assertFalse(result["effect_ok"])

    def test_missing_add_is_reported(self):
        result = transition_check(self.model, self.before, self.after - {"holding(shrimp)"})
        self.assertEqual(result["violations"]["missing_add"], ["holding(shrimp)"])
        self.assertEqual(result["violations"]["unexpected_removed"], ["holding(shrimp)"])
        self.assertFalse(result["goal_ok"])
        self.assertFalse(result["paper_line20"])
        self.assertFalse(result["effect_ok"])

    def test_remaining_delete_rejected_even_when_goal_holds(self):
        result = transition_check(self.model, self.before, self.after | {"at_source(shrimp)"})
        self.assertEqual(result["violations"]["remaining_del"], ["at_source(shrimp)"])
        self.assertTrue(result["goal_ok"])
        self.assertTrue(result["paper_line20"])
        self.assertFalse(result["strict_CP_transition_match"])
        self.assertFalse(result["effect_ok"])

    def test_preserved_precondition_cannot_disappear(self):
        self.model["pre"].append("at_source(apple)")
        result = transition_check(self.model, self.before, self.after - {"at_source(apple)"})
        self.assertEqual(result["violations"]["missing_preserved_pre"], ["at_source(apple)"])
        self.assertTrue(result["valid_trial"])
        self.assertTrue(result["goal_ok"])
        self.assertFalse(result["paper_line20"])
        self.assertFalse(result["effect_ok"])

    def test_frame_rejects_undeclared_addition(self):
        result = transition_check(self.model, self.before, self.after | {"home_completed()"})
        self.assertEqual(result["violations"]["unexpected_added"], ["home_completed()"])
        self.assertTrue(result["goal_ok"])
        self.assertTrue(result["paper_line20"])
        self.assertFalse(result["strict_CP_transition_match"])
        self.assertFalse(result["effect_ok"])

    def test_frame_rejects_undeclared_removal(self):
        result = transition_check(self.model, self.before, self.after - {"at_source(potato)"})
        self.assertEqual(result["violations"]["unexpected_removed"], ["at_source(potato)"])
        self.assertTrue(result["goal_ok"])
        self.assertTrue(result["paper_line20"])
        self.assertFalse(result["strict_CP_transition_match"])
        self.assertFalse(result["effect_ok"])

    def test_exception_rejects_an_otherwise_exact_transition(self):
        for error in (RuntimeError("执行失败"), "执行失败", ""):
            with self.subTest(exception=repr(error)):
                result = transition_check(self.model, self.before, self.after, exception=error)
                self.assertTrue(result["valid_trial"])
                self.assertTrue(result["goal_ok"])
                self.assertTrue(result["strict_CP_transition_match"])
                self.assertFalse(any(result["violations"].values()))
                self.assertFalse(result["effect_ok"])

    def test_transition_does_not_mutate_inputs(self):
        snapshot = deepcopy((self.model, self.before, self.after))
        transition_check(self.model, self.before, self.after)
        self.assertEqual((self.model, self.before, self.after), snapshot)


class PolicyProgramTests(unittest.TestCase):
    def test_unknown_api_is_rejected(self):
        with self.assertRaisesRegex(PolicyRejected, "unsupported API attribute api.unknown_api"):
            PolicyProgram("def policy(api):\n    api.unknown_api()", API_DOCS["pick"])

    def test_whole_action_macros_are_rejected_in_every_phase(self):
        for phase, allowed in API_DOCS.items():
            for macro in ("pick", "place", "home", "pick_and_place"):
                with self.subTest(phase=phase, macro=macro):
                    self.assertNotIn(macro, allowed)
                    with self.assertRaisesRegex(PolicyRejected, "unsupported API attribute"):
                        PolicyProgram(f"def policy(api):\n    api.{macro}()", allowed)

    def test_api_from_another_phase_is_rejected(self):
        with self.assertRaisesRegex(PolicyRejected, "unsupported API attribute api.carry"):
            PolicyProgram("def policy(api):\n    api.carry((0.4, 0.0, 0.5))", API_DOCS["pick"])

    def test_finite_program_passes_point_to_documented_api(self):
        point = (0.4, 0.0, 0.5)
        methods = {
            "grasp_point": Mock(return_value=point),
            "approach": Mock(return_value=None),
        }
        program = PolicyProgram(
            "def policy(api):\n    p = api.grasp_point()\n    api.approach(p)\n    return p",
            API_DOCS["pick"],
        )
        result = program.run(methods)
        methods["grasp_point"].assert_called_once_with()
        methods["approach"].assert_called_once_with(point)
        self.assertEqual(result["return_value"], point)
        self.assertEqual([row["method"] for row in result["trace"]], ["grasp_point", "approach"])
        self.assertTrue(all(row["status"] == "returned" for row in result["trace"]))

    def test_bad_signature_is_rejected_before_any_api_executes(self):
        calls = []

        def open_gripper():
            calls.append("open")

        def approach(point):
            calls.append(point)

        program = PolicyProgram(
            "def policy(api):\n    api.open_gripper()\n    api.approach(0.4, 0.0, 0.5)",
            API_DOCS["pick"],
        )
        with self.assertRaisesRegex(PolicyRejected, "api.approach"):
            program.run({"open_gripper": open_gripper, "approach": approach})
        self.assertEqual(calls, [])

    def test_api_exception_stops_execution_and_retains_failure_trace(self):
        error = RuntimeError("闭爪失败")
        close = Mock(side_effect=error)
        lift = Mock(return_value=None)
        program = PolicyProgram(
            "def policy(api):\n    api.close_gripper()\n    api.lift()", API_DOCS["pick"],
        )
        with self.assertRaises(RuntimeError) as raised:
            program.run({"close_gripper": close, "lift": lift})
        self.assertIs(raised.exception, error)
        close.assert_called_once_with()
        lift.assert_not_called()
        self.assertEqual(len(error.policy_trace), 1)
        self.assertEqual(error.policy_trace[0]["method"], "close_gripper")
        self.assertEqual(error.policy_trace[0]["status"], "failed")


class FineCoverSafetyTests(unittest.TestCase):
    def setUp(self):
        # 跳过真实机器人构造；只替换运行时，执行的仍是 FineCover 原方法。
        self.cover = FineCover.__new__(FineCover)
        self.cover.obj = "shrimp"
        self.cover.support = "bowl"
        self.r = SimpleNamespace(
            arm=SimpleNamespace(_grip_cmd=0.0, hand_bid=0),
            env=SimpleNamespace(d=SimpleNamespace(time=1.25, xmat=np.eye(3).reshape(1, 9))),
            c={"transport_z": 0.7, "lift_speed_m_s": 0.1},
            initial={"shrimp": np.array([0.4, 0.0, 0.4])},
            force_now={"left_finger": 0.1, "right_finger": 0.2},
            mark=Mock(), wait=Mock(), move=Mock(),
            ee=Mock(return_value=np.array([0.4, 0.0, 0.5])),
            grip_point=Mock(return_value=np.array([0.4, 0.0, 0.5])),
            pos=Mock(return_value=np.array([0.4, 0.0, 0.48])),
            active_welds=Mock(return_value=[]), released=Mock(return_value=False),
            events=[], completed=[], target="shrimp",
            grasp_relative=None, transport_monitor=False, current_contact_gap=0.2,
        )
        self.cover.r = self.r

    def test_close_rejects_absent_unilateral_or_insufficient_force(self):
        for left, right in ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.049, 0.2), (0.2, 0.049)):
            with self.subTest(left=left, right=right):
                self.r.force_now = {"left_finger": left, "right_finger": right}
                with self.assertRaisesRegex(RuntimeError, "No simultaneous two-finger force"):
                    self.cover.close_gripper()
                self.r.ee.assert_not_called()
                self.r.move.assert_not_called()
                self.assertEqual(self.r.events, [])
                self.assertIsNone(self.r.grasp_relative)
                self.assertFalse(self.r.transport_monitor)
        self.assertEqual(self.r.wait.call_args_list, [call(0.35, 0.0)] * 5)

    def test_close_accepts_bilateral_threshold_without_moving_arm(self):
        self.r.force_now = {"left_finger": 0.05, "right_finger": 0.05}
        self.cover.close_gripper()
        self.r.mark.assert_called_once_with("grasp_shrimp")
        self.assertEqual(self.r.wait.call_args_list, [call(0.35, 0.0), call(0.10, 0.0)])
        self.r.move.assert_not_called()
        self.assertEqual(self.r.events, [{
            "event": "bilateral_grasp_contact", "object": "shrimp",
            "normal_forces_N": {"left_finger": 0.05, "right_finger": 0.05}, "time": 1.25,
        }])
        np.testing.assert_allclose(self.r.grasp_relative, [0.0, 0.0, -0.02], atol=1e-12)
        self.assertTrue(self.r.transport_monitor)
        self.assertEqual(self.r.current_contact_gap, 0.0)

    def test_close_rejects_inconsistent_grasp_geometry(self):
        self.r.grip_point.return_value = np.array([0.45, 0.0, 0.5])
        with self.assertRaisesRegex(RuntimeError, "Grasp geometry inconsistent"):
            self.cover.close_gripper()
        self.assertEqual(self.r.events, [])
        self.assertFalse(self.r.transport_monitor)

    def test_close_rejects_active_weld_even_with_bilateral_force(self):
        self.r.active_welds.return_value = ["shrimp"]
        with self.assertRaisesRegex(RuntimeError, "must never activate a weld"):
            self.cover.close_gripper()
        self.assertEqual(self.r.events, [])
        self.assertFalse(self.r.transport_monitor)

    def test_lift_rejects_open_gripper_before_any_motion(self):
        for grip in (0.5, 1.0):
            with self.subTest(grip=grip):
                self.r.arm._grip_cmd = grip
                with self.assertRaisesRegex(RuntimeError, "Lift requires closed gripper"):
                    self.cover.lift()
                self.r.mark.assert_not_called()
                self.r.move.assert_not_called()
                self.r.wait.assert_not_called()

    def test_lift_rejects_payload_that_did_not_rise(self):
        self.r.pos.return_value = self.r.initial["shrimp"].copy()
        with self.assertRaisesRegex(RuntimeError, "Payload did not lift"):
            self.cover.lift()
        self.r.mark.assert_called_once_with("lift_shrimp")
        self.r.move.assert_called_once_with([0.4, 0.0, 0.7], 0.0, 0.1)

    def test_withdraw_rejects_unreleased_payload_before_any_motion(self):
        with self.assertRaisesRegex(RuntimeError, "Withdraw requires explicit release first"):
            self.cover.withdraw()
        self.r.released.assert_called_once_with()
        self.r.mark.assert_not_called()
        self.r.move.assert_not_called()
        self.r.wait.assert_not_called()
        self.assertEqual(self.r.target, "shrimp")
        self.assertEqual(self.r.completed, [])


class CameraGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 仅编译微型相机模型并计算静态外参；没有机器人、渲染器或 mj_step。
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody>
                <camera name="unused" pos="9 9 9" fovy="30"/>
                <camera name="down" pos="1 2 3" fovy="90"/>
                <camera name="turned" pos="1 2 3" xyaxes="0 1 0 -1 0 0" fovy="60"/>
                <camera name="side" pos="1 2 3" xyaxes="0 1 0 0 0 1" fovy="90"/>
            </worldbody></mujoco>
        """)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cls.env = SimpleNamespace(m=model, d=data)

    def test_project_optical_axis_to_image_center(self):
        np.testing.assert_allclose(project(self.env, [1, 2, 1], "down", 640), [320, 320])

    def test_project_uses_depth_and_image_y_sign(self):
        np.testing.assert_allclose(project(self.env, [2, 3, 1], "down", 640), [480, 160])
        np.testing.assert_allclose(project(self.env, [2, 3, -1], "down", 640), [400, 240])

    def test_rotated_camera_and_fovy_have_analytic_expected_values(self):
        expected_uv = np.full(2, 320 + 160 * np.sqrt(3))
        np.testing.assert_allclose(project(self.env, [2, 3, 1], "turned", 640), expected_uv)
        np.testing.assert_allclose(
            intersect_plane(self.env, expected_uv, "turned", 640, 1), [2, 3, 1], atol=1e-12,
        )

    def test_intersect_plane_has_independent_expected_world_point(self):
        np.testing.assert_allclose(
            intersect_plane(self.env, [480, 160], "down", 640, 1), [2, 3, 1], atol=1e-12,
        )

    def test_project_intersect_round_trip_across_cameras_and_sizes(self):
        for camera in ("down", "turned"):
            for size in (320, 640):
                for point in ([1, 2, 1], [1.2, 1.7, 0.5], [0.7, 2.1, -0.4]):
                    with self.subTest(camera=camera, size=size, point=point):
                        uv = project(self.env, point, camera, size)
                        np.testing.assert_allclose(
                            intersect_plane(self.env, uv, camera, size, point[2]), point, atol=1e-12,
                        )
        self.assertEqual(self.env.d.time, 0.0)

    def test_project_rejects_points_on_or_behind_camera_plane(self):
        for z in (3, 4):
            with self.subTest(z=z):
                with self.assertRaisesRegex(ValueError, "Point behind camera"):
                    project(self.env, [1, 2, z], "down", 640)

    def test_intersect_rejects_parallel_ray(self):
        with self.assertRaisesRegex(ValueError, "Ray parallel to calibrated plane"):
            intersect_plane(self.env, [320, 320], "side", 640, 1)

    def test_intersect_rejects_plane_at_or_behind_camera(self):
        for z in (3, 4):
            with self.subTest(z=z):
                with self.assertRaisesRegex(ValueError, "Plane behind camera"):
                    intersect_plane(self.env, [320, 320], "down", 640, z)


class StateValidityTests(unittest.TestCase):
    def test_initial_state_is_valid(self):
        self.assertTrue(state_valid(INITIAL))

    def test_single_held_object_with_other_objects_at_source_is_valid(self):
        for obj, _ in PAIRS:
            with self.subTest(object=obj):
                state = (INITIAL - {"empty()", f"at_source({obj})"}) | {f"holding({obj})"}
                self.assertTrue(state_valid(state))

    def test_partial_and_complete_placements_allow_empty_home(self):
        for count in (1, 2, 3):
            with self.subTest(placed=count):
                state = set(INITIAL) | {"home_completed()"}
                for obj, _ in PAIRS[:count]:
                    state.remove(f"at_source({obj})")
                    state.add(f"at_target({obj})")
                self.assertTrue(state_valid(state))

    def test_empty_and_holding_are_mutually_exclusive(self):
        for obj, _ in PAIRS:
            with self.subTest(object=obj):
                state = (INITIAL - {f"at_source({obj})"}) | {f"holding({obj})"}
                self.assertFalse(state_valid(state))

    def test_multiple_held_objects_are_rejected(self):
        for pair in combinations([obj for obj, _ in PAIRS], 2):
            with self.subTest(objects=pair):
                state = INITIAL - {"empty()"} - {f"at_source({obj})" for obj in pair}
                state.update(f"holding({obj})" for obj in pair)
                self.assertFalse(state_valid(state))

    def test_neither_empty_nor_holding_is_rejected(self):
        self.assertFalse(state_valid(INITIAL - {"empty()"}))

    def test_home_completed_while_holding_is_rejected(self):
        for obj, _ in PAIRS:
            with self.subTest(object=obj):
                state = (INITIAL - {"empty()", f"at_source({obj})"}) | {f"holding({obj})"}
                self.assertTrue(state_valid(state))
                self.assertFalse(state_valid(state | {"home_completed()"}))

    def test_each_object_requires_exactly_one_location_predicate(self):
        for obj, _ in PAIRS:
            atoms = [f"{pred}({obj})" for pred in ("at_source", "holding", "at_target")]
            for count in (0, 2, 3):
                for selected in combinations(atoms, count):
                    with self.subTest(object=obj, selected=selected):
                        state = (INITIAL - {f"at_source({obj})"}) | set(selected)
                        # 保持夹爪与 holding 一致，单独测试物体位置的互斥/完备约束。
                        if f"holding({obj})" in state:
                            state.remove("empty()")
                        self.assertFalse(state_valid(state))


class PerceptionParsingTests(unittest.TestCase):
    def test_point_alias_keeps_absolute_coordinates(self):
        uv, field = Perception.parse_pixel({"raw": '{"point":[340,270]}'}, 720)
        np.testing.assert_array_equal(uv, [340,270]); self.assertEqual(field, "point")

    def test_point_2d_in_fenced_list_is_not_lost(self):
        uv, field = Perception.parse_pixel({"raw": '```json\n[{"point_2d":[90,150]}]\n```'}, 720)
        np.testing.assert_array_equal(uv, [90,150]); self.assertEqual(field, "point_2d")

    def test_parse_rejects_out_of_range_instead_of_clamping(self):
        with self.assertRaises(ValueError):
            Perception.parse_pixel({"raw": '{"point":[900,150]}'}, 720)

    def test_parse_rejects_truncated_repeated_array(self):
        with self.assertRaises(ValueError):
            Perception.parse_pixel({"raw": '[[530,642,0,0,0'}, 1000)

    def test_static_cache_reads_only_preobserved_value(self):
        runtime = SimpleNamespace(pos=Mock(side_effect=AssertionError("must not read GT")))
        perception = Perception(runtime, "qwen_zoom", ".")
        perception.static_cache["bowl"] = np.array([.31,.16,.408])
        result = perception.locate("bowl", .408, "support_floor_center")
        np.testing.assert_array_equal(result,[.31,.16,.408])
        runtime.pos.assert_not_called()
        self.assertFalse(perception.records[-1]["fallback"])

    def test_cache_is_not_a_live_mutable_reference(self):
        perception = Perception(SimpleNamespace(), "qwen_zoom", ".")
        perception.static_cache["bowl"] = np.array([.31,.16,.408])
        result = perception.locate("bowl",.408,"support_floor_center");result[0]=.8
        self.assertEqual(perception.static_cache["bowl"][0],.31)


if __name__ == "__main__":
    unittest.main()
