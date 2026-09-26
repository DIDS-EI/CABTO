"""双臂抬盘 codegen 的纯符号、stub 状态与受限 API 单元测试。

不调用 LLM，不启动 MuJoCo，不运行长仿真。
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parent
FORMAL_ROOT = ROOT.parent / "CABTO/exp4_bt_tasks/stage3_cabto"
sys.path.insert(0, str(FORMAL_ROOT))
sys.path.insert(0, str(ROOT))

from core.dual_lift_shelf_bridge import DualLiftRuntime, FreeTrayRuntime, lift_domain, lift_goal
from formal_bt import ModelLibrary, action_id, symbolic_dry_run, validate_model
import probe_dual_lift_codegen as generated


START = {
    "empty(0)", "empty(1)", "at_source(item0)", "at_source(item1)",
    "slot_free(0)", "slot_free(1)", "tray_at_table()",
    "hands_safe_for_lateral(0)", "hands_safe_for_lateral(1)",
}
CUSTOM_PHASES = (
    "dual_grasp_tray", "dual_lift_tray", "dual_translate_tray",
    "dual_lower_tray", "dual_release_tray", "dual_withdraw_hands",
)


def model_by(name, arm=None):
    rows = [m for m in lift_domain() if m["name"] == name]
    if arm is not None:
        rows = [m for m in rows if m["args"]["arm"] == str(arm)]
    if len(rows) != 1:
        raise AssertionError(f"expected one model {name}/{arm}, got {len(rows)}")
    return rows[0]


def _runtime_stub(*, pos=(0.30, 0.0, 0.40), welds=(False, False), grips=(1.0, 1.0),
                  support=True, footprint=True, strict_shelf=False,
                  ee_positions=None, finger_positions=None):
    runtime = DualLiftRuntime.__new__(DualLiftRuntime)
    runtime.h = 0.40
    runtime._tray_xy0 = np.array([0.30, 0.0])
    runtime.p = {
        "carry_tray_base_z": 0.62,
        "shelf_xy": [0.58, 0.0],
        "shelf_top_z": 0.52,
        "shelf_tilt_tolerance_deg": 8.0,
        "lift_handle_z": 0.075,
    }
    state = {
        "pos": np.asarray(pos, float),
        "welds": list(welds),
        "support": support,
        "footprint": footprint,
        "strict_shelf": strict_shelf,
    }
    handles = [np.array([0.30, 0.112, 0.475]), np.array([0.30, -0.112, 0.475])]
    ee_positions = handles if ee_positions is None else [np.asarray(p, float) for p in ee_positions]
    finger_positions = handles if finger_positions is None else [np.asarray(p, float) for p in finger_positions]
    arms = []
    for i in (0, 1):
        arm = SimpleNamespace(_grip_cmd=float(grips[i]))
        arm.get_ee_pose = Mock(return_value=(np.asarray(ee_positions[i], float), np.array([1., 0., 0., 0.])))
        arm._finger_mid = Mock(return_value=np.asarray(finger_positions[i], float))

        def set_weld(name, active, i=i):
            if name != "tray":
                raise AssertionError(name)
            state["welds"][i] = bool(active)

        arm._set_weld = Mock(side_effect=set_weld)
        arms.append(arm)
    scout = SimpleNamespace(
        tray_weld_active=Mock(side_effect=lambda i: state["welds"][i]),
        tray_supported_by_shelf=Mock(side_effect=lambda: state["support"]),
        tray_footprint_on_shelf=Mock(side_effect=lambda: state["footprint"]),
        tray_on_shelf=Mock(side_effect=lambda: state["strict_shelf"]),
        handle_world=Mock(side_effect=lambda i: handles[i].copy()),
        dual_move_targets=Mock(return_value=[0.0, 0.0]),
        dual_translate_tray=Mock(),
        hold=Mock(),
        events=[],
    )
    runtime.scout = scout
    runtime.env = SimpleNamespace(
        arms=arms,
        d=SimpleNamespace(time=1.25),
        hold_arms=Mock(),
        record_frame=Mock(),
    )
    runtime.tray_pose = Mock(side_effect=lambda: (state["pos"].copy(), np.array([1., 0., 0., 0.])))
    return runtime, state, handles


def observed_state(runtime, base=()):
    with patch.object(FreeTrayRuntime, "state", return_value=set(base)):
        return runtime.state()


class FormalModelTests(unittest.TestCase):
    def test_all_fourteen_action_models_validate(self):
        models = lift_domain()
        self.assertEqual(len(models), 14)
        ids = [validate_model(model) for model in models]
        self.assertEqual(ids, [action_id(model) for model in models])
        self.assertEqual(len(ids), len(set(ids)))

    def test_formal_bt_symbolic_dry_run_reaches_goal_in_fourteen_actions(self):
        models = lift_domain()
        start, goal = set(START), lift_goal()
        report = symbolic_dry_run(ModelLibrary(models).build(start, goal), start, goal, max_steps=32)
        self.assertTrue(report["reached_goal"], report)
        self.assertEqual(report["steps"], 14)
        self.assertEqual(report["scope"], "symbolic_only")
        self.assertTrue(goal <= set(report["final_state"]))
        names = [item.split("(", 1)[0] for item in report["action_ids"]]
        for phase in CUSTOM_PHASES:
            self.assertEqual(names.count(phase), 1, report)

    def test_dual_grasp_preserves_table_fact_and_deletes_clearance_facts(self):
        model = model_by("dual_grasp_tray")
        deleted = set(model["del"])
        self.assertNotIn("tray_at_table()", deleted)
        self.assertTrue({
            "empty(0)", "empty(1)", "parked(0)", "parked(1)",
            "hands_safe_for_lateral(0)", "hands_safe_for_lateral(1)",
        } <= deleted)
        after = (set(model["pre"]) | set(model["add"])) - deleted
        self.assertIn("tray_at_table()", after)
        self.assertIn("dual_holding_tray()", after)

    def test_lift_model_starts_the_cooperative_move_chain(self):
        model = model_by("dual_lift_tray")
        self.assertEqual(set(model["pre"]), {
            "packed(0)", "packed(1)", "dual_holding_tray()", "tray_at_table()",
        })
        self.assertEqual(set(model["add"]), {"tray_lifted()"})
        self.assertEqual(set(model["del"]), {"tray_at_table()"})

    def test_translate_model_consumes_lifted_and_adds_above(self):
        model = model_by("dual_translate_tray")
        self.assertEqual(set(model["pre"]), {
            "packed(0)", "packed(1)", "dual_holding_tray()", "tray_lifted()",
        })
        self.assertEqual(set(model["add"]), {"tray_above_shelf()"})
        self.assertEqual(set(model["del"]), {"tray_lifted()"})

    def test_lower_model_consumes_above_and_adds_supported_held(self):
        model = model_by("dual_lower_tray")
        self.assertEqual(set(model["pre"]), {
            "packed(0)", "packed(1)", "dual_holding_tray()", "tray_above_shelf()",
        })
        self.assertEqual(set(model["add"]), {"tray_on_shelf_held()"})
        self.assertEqual(set(model["del"]), {"tray_above_shelf()"})

    def test_release_requires_packed_dual_held_supported_tray(self):
        model = model_by("dual_release_tray")
        self.assertEqual(set(model["pre"]), {
            "packed(0)", "packed(1)", "dual_holding_tray()", "tray_on_shelf_held()",
        })
        self.assertEqual(set(model["add"]), {
            "empty(0)", "empty(1)", "tray_on_shelf()", "hands_near_shelf()",
        })
        self.assertEqual(set(model["del"]), {"dual_holding_tray()", "tray_on_shelf_held()"})

    def test_withdraw_adds_raised_and_safe_clearance(self):
        model = model_by("dual_withdraw_hands")
        self.assertEqual(set(model["add"]), {
            "hands_raised()", "hands_safe_for_lateral(0)", "hands_safe_for_lateral(1)",
        })
        self.assertEqual(set(model["del"]), {"hands_near_shelf()"})

    def test_table_park_and_shelf_park_have_distinct_physical_contexts(self):
        for arm in (0, 1):
            with self.subTest(arm=arm):
                table = model_by("park", arm)
                shelf = model_by("park_after_shelf", arm)
                self.assertIn("tray_at_table()", table["pre"])
                self.assertNotIn(f"hands_safe_for_lateral({arm})", table["pre"])
                self.assertIn(f"hands_safe_for_lateral({arm})", shelf["pre"])
                self.assertIn("tray_on_shelf()", shelf["pre"])
                self.assertEqual(shelf["program_kind"], "park")


class RuntimeStateTests(unittest.TestCase):
    def test_two_active_welds_and_two_closed_grippers_are_dual_holding(self):
        runtime, _, _ = _runtime_stub(welds=(True, True), grips=(0.0, 0.0))
        self.assertIn("dual_holding_tray()", observed_state(runtime))

    def test_table_location_is_preserved_while_dually_held_before_lift(self):
        runtime, _, _ = _runtime_stub(pos=(0.30, 0.0, 0.40), welds=(True, True), grips=(0.0, 0.0))
        state = observed_state(runtime)
        self.assertIn("dual_holding_tray()", state)
        self.assertIn("tray_at_table()", state)

    def test_one_active_weld_is_not_dual_holding(self):
        for welds in ((True, False), (False, True)):
            with self.subTest(welds=welds):
                runtime, _, _ = _runtime_stub(welds=welds, grips=(0.0, 0.0))
                self.assertNotIn("dual_holding_tray()", observed_state(runtime))

    def test_open_gripper_prevents_dual_holding_even_with_two_welds(self):
        runtime, _, _ = _runtime_stub(welds=(True, True), grips=(0.0, 1.0))
        self.assertNotIn("dual_holding_tray()", observed_state(runtime))

    def test_lift_above_and_supported_held_predicates_are_phase_partitioned(self):
        cases = (
            (np.array([0.30, 0.0, 0.62]), "tray_lifted()"),
            (np.array([0.58, 0.0, 0.62]), "tray_above_shelf()"),
            (np.array([0.58, 0.0, 0.52]), "tray_on_shelf_held()"),
        )
        phase_atoms = {"tray_lifted()", "tray_above_shelf()", "tray_on_shelf_held()"}
        runtime, state, _ = _runtime_stub(welds=(True, True), grips=(0.0, 0.0))
        for pos, expected in cases:
            with self.subTest(expected=expected):
                state["pos"] = pos
                self.assertEqual(observed_state(runtime) & phase_atoms, {expected})

    def test_supported_held_requires_contact_and_full_footprint(self):
        runtime, state, _ = _runtime_stub(
            pos=(0.58, 0.0, 0.52), welds=(True, True), grips=(0.0, 0.0),
        )
        for support, footprint in ((False, True), (True, False)):
            with self.subTest(support=support, footprint=footprint):
                state["support"], state["footprint"] = support, footprint
                self.assertNotIn("tray_on_shelf_held()", observed_state(runtime))

    def test_final_tray_on_shelf_is_not_inferred_when_strict_scout_rejects(self):
        runtime, _, _ = _runtime_stub(
            pos=(0.58, 0.0, 0.52), welds=(False, False), grips=(1.0, 1.0),
            support=True, footprint=True, strict_shelf=False,
        )
        self.assertNotIn("tray_on_shelf()", observed_state(runtime))
        runtime.scout.tray_on_shelf.assert_called_once_with()

    def test_vertical_withdraw_clears_near_predicate_and_adds_raised(self):
        ee = (np.array([0.58, 0.112, 0.725]), np.array([0.58, -0.112, 0.725]))
        runtime, _, _ = _runtime_stub(pos=(0.58, 0.0, 0.52), welds=(False, False),
                                      grips=(1.0, 1.0), strict_shelf=True, ee_positions=ee)
        state = observed_state(runtime)
        self.assertNotIn("hands_near_shelf()", state)
        self.assertIn("hands_raised()", state)
        self.assertIn("hands_safe_for_lateral(0)", state)
        self.assertIn("hands_safe_for_lateral(1)", state)

    def test_final_tray_on_shelf_comes_from_strict_scout_predicate(self):
        runtime, _, _ = _runtime_stub(
            pos=(0.58, 0.0, 0.52), welds=(False, False), grips=(1.0, 1.0),
            strict_shelf=True,
        )
        self.assertIn("tray_on_shelf()", observed_state(runtime))
        runtime.scout.tray_on_shelf.assert_called_once_with()


class RestrictedApiTests(unittest.TestCase):
    def test_dual_grasp_api_rejects_forged_handle_tuple(self):
        runtime, _, _ = _runtime_stub()
        api = runtime.api(model_by("dual_grasp_tray"))
        handles = api["target_handles"]()
        forged_point = np.asarray(handles[0]) + np.array([1e-3, 0.0, 0.0])
        forged = (tuple(map(float, forged_point)), handles[1])
        for method in ("move_both_above", "move_both_to"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "exact current tuple"):
                    api[method](forged)
        runtime.scout.dual_move_targets.assert_not_called()

    def test_cooperative_move_api_rejects_forged_bound_target_pose(self):
        runtime, _, _ = _runtime_stub(welds=(True, True), grips=(0.0, 0.0))
        for phase in ("dual_lift_tray", "dual_translate_tray", "dual_lower_tray"):
            with self.subTest(phase=phase):
                api = runtime.api(model_by(phase))
                target = api["target_tray_pose"]()
                forged = (target[0], target[1], target[2] + 1e-3)
                with self.assertRaisesRegex(ValueError, "bound target_tray_pose"):
                    api["move_tray_to"](forged)
        runtime.scout.dual_translate_tray.assert_not_called()

    def test_attach_rejects_unclosed_hand_before_welding(self):
        runtime, _, handles = _runtime_stub(grips=(1.0, 0.0))
        with self.assertRaisesRegex(RuntimeError, "both grippers closed"):
            runtime._attach_both(handles)
        for arm in runtime.env.arms:
            arm._set_weld.assert_not_called()
        runtime.scout.hold.assert_not_called()

    def test_attach_rejects_bad_geometry_before_welding(self):
        far = (np.array([0.50, 0.112, 0.475]), np.array([0.30, -0.112, 0.475]))
        runtime, _, handles = _runtime_stub(grips=(0.0, 0.0), ee_positions=far)
        with self.assertRaisesRegex(RuntimeError, "geometry check failed"):
            runtime._attach_both(handles)
        for arm in runtime.env.arms:
            arm._set_weld.assert_not_called()
        runtime.scout.hold.assert_not_called()

    def test_attach_accepts_only_closed_geometrically_aligned_pair(self):
        runtime, state, handles = _runtime_stub(grips=(0.0, 0.0))
        runtime._attach_both(handles)
        self.assertEqual(state["welds"], [True, True])
        for arm in runtime.env.arms:
            arm._set_weld.assert_called_once_with("tray", True)
        runtime.scout.hold.assert_called_once_with(12)
        self.assertEqual([event["arm"] for event in runtime.scout.events], [0, 1])


class ExecuteTransitionEvidenceTests(unittest.TestCase):
    def test_missing_required_transition_call_rejects_custom_program(self):
        required = {
            "dual_grasp_tray": "attach_both",
            "dual_lift_tray": "move_tray_to",
            "dual_translate_tray": "move_tray_to",
            "dual_lower_tray": "move_tray_to",
            "dual_release_tray": "detach_both",
            "dual_withdraw_hands": "raise_both",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, expected in required.items():
                with self.subTest(name=name):
                    base_record = {
                        "trace": [{"method": "settle", "status": "returned"}],
                        "diagnostics": {},
                        "effect_ok": True,
                    }
                    with patch.object(generated, "base_execute", return_value=deepcopy(base_record)), \
                         patch.object(generated, "make_llm_backend", side_effect=AssertionError("不得调用 LLM")) as llm:
                        rec = generated.execute(object(), model_by(name), object(), Path(directory))
                    self.assertFalse(rec["effect_ok"])
                    self.assertEqual(rec["diagnostics"]["required_transition_call"], expected)
                    self.assertFalse(rec["diagnostics"]["required_transition_observed"])
                    self.assertEqual(rec["diagnostics"]["missing_transition_evidence"], expected)
                    llm.assert_not_called()
                    saved = json.loads((Path(directory) / "feedback.json").read_text())
                    self.assertFalse(saved["effect_ok"])

    def test_observed_required_transition_call_preserves_base_success(self):
        model = model_by("dual_release_tray")
        base_record = {
            "trace": [{"method": "detach_both", "status": "returned"}],
            "diagnostics": {},
            "effect_ok": True,
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(generated, "base_execute", return_value=deepcopy(base_record)):
            rec = generated.execute(object(), model, object(), Path(directory))
        self.assertTrue(rec["effect_ok"])
        self.assertTrue(rec["diagnostics"]["required_transition_observed"])
        self.assertNotIn("missing_transition_evidence", rec["diagnostics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
