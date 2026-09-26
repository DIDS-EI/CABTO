"""剩余 CABTO 轻量回归：真实符号规划、假控制运行时，不加载 MLX 或步进仿真。"""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# 直接运行及导入时均避免为依赖新增字节码文件。
sys.dont_write_bytecode = True

import numpy as np

from core import remaining_blocks_api as blocks
from core import remaining_dual_api as dual
from core import remaining_handover_api as handover
from formal_bt import ModelLibrary, PlanningError, action_id, symbolic_dry_run, validate_model
from policy_codegen import PolicyProgram, PolicyRejected

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
RETURN_METHODS = {
    "safe_pose", "move_safe", "target_pose", "move_above", "align_target",
    "lower", "release", "retreat", "settle",
}
PARK_METHODS = {"raise_clear", "target_pose", "move_to", "settle"}


class PourInterfaceTests(unittest.TestCase):
    def check_return(self, kind, arm, obj):
        model = {"name": kind, "program_kind": kind, "args": {"arm": arm, "obj": obj}}
        runtime = dual.FinePourRuntime.__new__(dual.FinePourRuntime)
        runtime.audit = SimpleNamespace(phase=None)
        runtime.env = SimpleNamespace(arms=[SimpleNamespace(), SimpleNamespace()])
        runtime.scout = SimpleNamespace(skills=[SimpleNamespace(), SimpleNamespace()])
        docs = dual.docs_for("pour", model)
        api = runtime.api(model)
        self.assertEqual(set(docs), RETURN_METHODS)
        self.assertEqual(set(api), RETURN_METHODS)
        self.assertTrue(all(callable(method) for method in api.values()))
        self.assertNotIn("return_object", "\n".join(docs.values()))
        with self.assertRaisesRegex(PolicyRejected, "unsupported API attribute"):
            PolicyProgram("def policy(api):\n    api.return_object()", docs)

    def test_return_source_has_only_fine_grained_interfaces(self):
        self.check_return("return_source", "0", "canL")

    def test_return_basin_has_only_fine_grained_interfaces(self):
        self.check_return("return_basin", "1", "basin")

    def test_return_docs_are_independent_copies(self):
        before = deepcopy(dual.POUR_DOCS)
        for kind in ("return_source", "return_basin"):
            docs = dual.docs_for("pour", {"program_kind": kind})
            docs.clear()
            self.assertEqual(set(dual.docs_for("pour", {"program_kind": kind})), RETURN_METHODS)
        self.assertEqual(dual.POUR_DOCS, before)
        self.assertIsNot(dual.POUR_DOCS["return_source"], dual.POUR_DOCS["return_basin"])


class StorageParkTests(unittest.TestCase):
    def setUp(self):
        self.point = (.35, .45, .80)
        self.base = {
            "target_pose": Mock(return_value=self.point),
            "move_to": Mock(return_value=.002),
            "settle": Mock(),
        }
        base_patch = patch.object(dual.FreeTrayRuntime, "api", return_value=self.base)
        self.base_binding = base_patch.start()
        self.addCleanup(base_patch.stop)
        self.arm = SimpleNamespace(
            is_holding=Mock(return_value=False),
            get_ee_pose=Mock(return_value=(np.array([.3, .2, .60]), None)),
        )
        self.runtime = dual.FineStorageRuntime.__new__(dual.FineStorageRuntime)
        self.runtime.h = .5
        self.runtime.env = SimpleNamespace(
            arms=[self.arm], obj_names=["item0", "item1", "tray"],
            d=SimpleNamespace(time=1.25),
        )
        self.runtime.scout = SimpleNamespace(go=Mock(), events=[])
        self.model = {"name": "park", "program_kind": "park", "args": {"arm": "0"}}
        self.api = self.runtime.api(self.model)

    def test_park_docs_and_runtime_have_exactly_four_interfaces(self):
        self.assertEqual(set(dual.docs_for("storage", self.model)), PARK_METHODS)
        self.assertEqual(set(self.api), PARK_METHODS)
        self.base_binding.assert_called_once_with(self.runtime, self.model)

    def test_move_without_raise_clear_is_rejected_before_base_motion(self):
        with self.assertRaisesRegex(RuntimeError, "显式竖直撤离证据"):
            self.api["move_to"](self.point)
        self.base["move_to"].assert_not_called()
        self.runtime.scout.go.assert_not_called()
        self.assertEqual(self.runtime.scout.events, [])

    def test_target_pose_and_settle_cannot_supply_clearance(self):
        self.assertEqual(self.api["target_pose"](), self.point)
        self.api["settle"]()
        with self.assertRaisesRegex(RuntimeError, "显式竖直撤离证据"):
            self.api["move_to"](self.point)
        self.base["move_to"].assert_not_called()
        self.runtime.scout.go.assert_not_called()

    def test_raise_clear_is_vertical_then_delegates_single_motion(self):
        self.api["raise_clear"]()
        args = self.runtime.scout.go.call_args.args
        self.assertEqual((args[0], args[2]), (0, 1.0))
        np.testing.assert_allclose(args[1], [.3, .2, .84])
        self.runtime.scout.go.assert_called_once()
        self.base["move_to"].assert_not_called()
        self.assertEqual(self.runtime.scout.events[0]["explicit_api"], "raise_clear")
        self.assertEqual(self.api["move_to"](self.point), .002)
        self.base["move_to"].assert_called_once_with(self.point)

    def test_holding_rejects_raise_clear_before_any_motion(self):
        self.arm.is_holding.return_value = True
        with self.assertRaisesRegex(RuntimeError, "要求空手"):
            self.api["raise_clear"]()
        self.runtime.scout.go.assert_not_called()
        self.base["move_to"].assert_not_called()
        self.assertEqual(self.runtime.scout.events, [])

    def test_clearance_is_not_shared_between_api_bindings(self):
        self.api["raise_clear"]()
        rebound = self.runtime.api(self.model)
        with self.assertRaisesRegex(RuntimeError, "显式竖直撤离证据"):
            rebound["move_to"](self.point)
        self.base["move_to"].assert_not_called()

    def test_failed_raise_clear_invalidates_previous_evidence(self):
        self.api["raise_clear"]()
        self.runtime.scout.go.side_effect = RuntimeError("控制器未完成")
        with self.assertRaisesRegex(RuntimeError, "控制器未完成"):
            self.api["raise_clear"]()
        with self.assertRaisesRegex(RuntimeError, "显式竖直撤离证据"):
            self.api["move_to"](self.point)
        self.base["move_to"].assert_not_called()


class HandoverInterfaceTests(unittest.TestCase):
    def setUp(self):
        # 仅替换硬件依赖；API 的校验、步骤证据和闭爪逻辑均执行原方法。
        self.point = (.4, .1, .7)
        self.runtime = handover.FineHandoverRuntime.__new__(handover.FineHandoverRuntime)
        self.arms = [SimpleNamespace(
            s=side, _grasp_enabled=False, _grip_cmd=1.0,
            get_ee_pose=Mock(return_value=(np.array(self.point), None)),
            _set_weld=Mock(),
        ) for side in ("_L", "_R")]

        def hold(grips, sub_steps):
            for arm in self.arms:
                if arm.s in grips:
                    arm._grip_cmd = grips[arm.s]

        r = self.runtime
        r.env = SimpleNamespace(arms=self.arms, d=SimpleNamespace(time=1.0),
                                hold_arms=Mock(side_effect=hold))
        r.skills = [SimpleNamespace(_tick=Mock()), SimpleNamespace(_tick=Mock())]
        r.audit = SimpleNamespace(phase=None, recent_handle_contacts=Mock(return_value=[]))
        r.home_dirty = [False, False]
        r.events = []
        r.h = .5
        r.p = {"approach_clearance": .1, "meet_center_xyz": list(self.point),
               "destination_center_xy": [.4, .1], "object_half_size": [.02, .02, .03]}
        r.collision_free = Mock(return_value=True)
        r.weld_constraint_active = Mock(return_value=False)
        r.weld_active = Mock(return_value=False)
        r.pos = Mock(return_value=np.array(self.point))
        r.handle_world = Mock(return_value=np.array(self.point))
        r._finger_target = Mock(return_value=np.array(self.point))
        r._move = Mock()
        r._move_object_slow = Mock()

    def bind(self, kind):
        arm = "1" if kind in ("receive", "receiver_place") else "0"
        return self.runtime.api({"program_kind": kind, "args": {"arm": arm, "obj": "baton"}})

    def assert_no_control(self):
        self.runtime._move.assert_not_called()
        self.runtime._move_object_slow.assert_not_called()
        self.runtime.env.hold_arms.assert_not_called()
        for arm in self.arms:
            arm._set_weld.assert_not_called()
        self.assertEqual(self.runtime.events, [])
        self.assertEqual(self.runtime.home_dirty, [False, False])

    def test_no_whole_pick_or_place_macro_in_docs_or_bound_api(self):
        for kind, docs in handover.DOCS.items():
            with self.subTest(kind=kind):
                api = self.bind(kind)
                self.assertEqual(set(api), set(docs))
                for macro in ("donor_pick", "receiver_place"):
                    self.assertNotIn(macro, api)
                    self.assertNotIn(macro, docs)
                    with self.assertRaisesRegex(PolicyRejected, "unsupported API attribute"):
                        PolicyProgram(f"def policy(api):\n    api.{macro}()", docs)
        self.assert_no_control()

    def test_attach_without_close_is_rejected_before_any_control(self):
        for kind in ("donor_pick", "receive"):
            with self.subTest(kind=kind):
                api = self.bind(kind)
                self.assertIn("close_gripper", api)
                self.assertIn("attach_contact_gated", api)
                self.assertIsNot(api["close_gripper"], api["attach_contact_gated"])
                with self.assertRaisesRegex(RuntimeError, "close_gripper"):
                    api["attach_contact_gated"]()
        self.assert_no_control()
        self.runtime.audit.recent_handle_contacts.assert_not_called()

    def test_close_without_descent_is_rejected_before_any_control(self):
        for kind in ("donor_pick", "receive"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(RuntimeError, "descend"):
                    self.bind(kind)["close_gripper"]()
        self.assert_no_control()

    def test_close_does_not_attach_and_attach_still_requires_contacts(self):
        for kind, index in (("donor_pick", 0), ("receive", 1)):
            with self.subTest(kind=kind):
                self.runtime.weld_active.side_effect = lambda i, receiver=index: receiver == 1 and i == 0
                api = self.bind(kind)
                p = api["grasp_pose"]()
                api["approach"](p)
                api["descend"](p)
                observation = api["close_gripper"]()
                self.assertEqual(self.arms[index]._grip_cmd, 0.0)
                self.assertFalse(observation["weld_constraint_active"])
                self.arms[index]._set_weld.assert_not_called()
                count = self.runtime.env.hold_arms.call_count
                with self.assertRaisesRegex(RuntimeError, "未过门控"):
                    api["attach_contact_gated"]()
                self.assertEqual(self.runtime.env.hold_arms.call_count, count)
                self.arms[index]._set_weld.assert_not_called()
        self.assertEqual(self.runtime.env.hold_arms.call_count, 60)
        self.assertEqual(self.runtime.events, [])

    def test_detach_without_open_is_rejected_before_any_control(self):
        for kind in ("donor_release", "receiver_place"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(RuntimeError, "open_gripper"):
                    self.bind(kind)["detach"]()
        self.assert_no_control()

    def test_unissued_coordinates_are_rejected_before_motion(self):
        for kind, method in (("donor_pick", "approach"), ("receive", "descend"),
                             ("present", "move_held"), ("receiver_place", "carry"),
                             ("receiver_place", "lower")):
            with self.subTest(kind=kind, method=method):
                with self.assertRaisesRegex(ValueError, "未经本次只读接口签发"):
                    self.bind(kind)[method](self.point)
        self.assert_no_control()

    def test_malformed_modified_and_nonfinite_targets_are_rejected(self):
        for kind, reader, methods in (
            ("donor_pick", "grasp_pose", ("approach", "descend")),
            ("receive", "grasp_pose", ("approach", "descend")),
            ("present", "target_pose", ("move_held",)),
            ("receiver_place", "target_pose", ("carry", "lower")),
        ):
            api = self.bind(kind)
            point = api[reader]()
            invalid = [None, (), point[:2], (*point, 0), np.array(point),
                       [str(point[0]), *point[1:]], [True, *point[1:]],
                       [float("nan"), *point[1:]], [float("inf"), *point[1:]],
                       [float("-inf"), *point[1:]], [point[0] + .01, *point[1:]]]
            for method in methods:
                for value in invalid:
                    with self.subTest(kind=kind, method=method, value=repr(value)):
                        with self.assertRaises(ValueError):
                            api[method](value)
        self.assert_no_control()

    def test_target_issuance_is_local_to_binding(self):
        first = self.bind("present")
        point = first["target_pose"]()
        second = self.bind("present")
        with self.assertRaisesRegex(ValueError, "未经本次只读接口签发"):
            second["move_held"](point)
        self.assert_no_control()

    def test_invalid_model_binding_is_rejected(self):
        invalid = [("donor_pick", {"arm": arm, "obj": "baton"})
                   for arm in (-1, 2, True, False, 0.0, None, "00", "2")]
        invalid += [("donor_pick", {"arm": "0", "obj": "other"}),
                    ("donor_pick", {"arm": "1", "obj": "baton"}),
                    ("receive", {"arm": "0", "obj": "baton"}),
                    ("receiver_place", {"arm": "0", "obj": "baton"}),
                    ("unknown", {"arm": "0", "obj": "baton"})]
        for kind, args in invalid:
            with self.subTest(kind=kind, args=args):
                with self.assertRaises(ValueError):
                    self.runtime.api({"program_kind": kind, "args": args})
        self.assert_no_control()


class BlocksFormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.source = blocks.transferred_models(PACKAGE)
        cls.document = json.loads(cls.source.read_text(encoding="utf-8"))
        cls.schemas = {schema["name"]: schema for schema in cls.document["schemas"]}

    def test_source_is_the_real_9b_schema_artifact(self):
        self.assertEqual(self.source, PACKAGE / "outputs/cover_cabto_fix/proposal_final_v2/models.json")
        self.assertEqual(self.document["proposal_model"], "mlx-community/Qwen3.5-9B-4bit")
        self.assertEqual(set(self.schemas), {"pick", "place", "home"})

    def test_exactly_five_unique_grounded_actions_validate(self):
        expected = {(kind, obj, support) for kind in ("pick", "place")
                    for obj, support in (("yellow", "green"), ("red", "blue"))}
        expected.add(("home", None, None))
        actual = {(m["name"], m["args"].get("object"), m["args"].get("support")) for m in self.models}
        self.assertEqual(len(self.models), 5)
        self.assertEqual(actual, expected)
        self.assertEqual(len({validate_model(m) for m in self.models}), 5)
        for model in self.models:
            self.assertEqual(model["program_kind"], model["name"])

    def test_grounded_effects_are_unmodified_schema_substitutions(self):
        for model in self.models:
            with self.subTest(action=action_id(model)):
                schema = self.schemas[model["name"]]
                for field in ("pre", "add", "del"):
                    expected = [atom.replace("OBJECT", model["args"]["object"])
                                for atom in schema[field]] if model["name"] != "home" else schema[field]
                    self.assertEqual(model[field], expected)
                    self.assertTrue(all("OBJECT" not in atom for atom in model[field]))
        self.assertEqual(next(m for m in self.models if m["name"] == "home")["args"], {})

    def test_declared_tasks_are_exactly_b1_b2_b3(self):
        tasks = blocks.task_set()
        self.assertEqual([task["id"] for task in tasks], ["B1", "B2", "B3"])
        for task, objects in zip(tasks, (("yellow",), ("red",), ("yellow", "red"))):
            self.assertEqual(set(task["initial"]), {"empty()", "at_source(yellow)", "at_source(red)"})
            self.assertEqual(set(task["goal"]), {"empty()", "home_completed()"} |
                             {f"at_target({obj})" for obj in objects})

    def check_task(self, task_id, objects):
        task = next(t for t in blocks.task_set() if t["id"] == task_id)
        start, goal = set(task["initial"]), set(task["goal"])
        before = deepcopy((self.models, start, goal))
        tree = ModelLibrary(self.models, bt_root=PACKAGE / "BTExpansion-demo").build(start, goal)
        result = symbolic_dry_run(tree, start, goal, max_steps=15)
        self.assertTrue(result["reached_goal"], result)
        self.assertEqual(result["scope"], "symbolic_only")
        self.assertLessEqual(result["steps"], 15)
        expected = {action_id(m) for m in self.models
                    if m["name"] == "home" or m["args"].get("object") in objects}
        self.assertEqual(set(result["action_ids"]), expected)
        self.assertEqual(result["steps"], len(result["action_ids"]))
        # 独立检查实际返回的动作轨迹及终态，不读取产物中的 complete/success 缓存。
        state = set(start)
        by_id = {action_id(m): m for m in self.models}
        for key in result["action_ids"]:
            model = by_id[key]
            self.assertLessEqual(set(model["pre"]), state)
            state = (state | set(model["add"])) - set(model["del"])
        self.assertEqual(set(result["final_state"]), state)
        self.assertLessEqual(goal, state)
        self.assertEqual(tree.tick(state), ("success", None))
        self.assertEqual((self.models, start, goal), before)

    def test_b1_reaches_goal_with_real_formal_dry_run(self):
        self.check_task("B1", {"yellow"})

    def test_b2_reaches_goal_with_real_formal_dry_run(self):
        self.check_task("B2", {"red"})

    def test_b3_reaches_goal_with_real_formal_dry_run(self):
        self.check_task("B3", {"yellow", "red"})

    def test_removing_any_action_breaks_b3_completeness(self):
        task = next(t for t in blocks.task_set() if t["id"] == "B3")
        start, goal = set(task["initial"]), set(task["goal"])
        for removed in self.models:
            with self.subTest(removed=action_id(removed)):
                reduced = [m for m in self.models if action_id(m) != action_id(removed)]
                try:
                    tree = ModelLibrary(reduced, bt_root=PACKAGE / "BTExpansion-demo").build(start, goal)
                except PlanningError:
                    continue
                result = symbolic_dry_run(tree, start, goal, max_steps=15)
                self.assertFalse(result["reached_goal"], result)
                self.assertFalse(goal <= set(result["final_state"]), result)


class SourceSafetyTests(unittest.TestCase):
    @staticmethod
    def qpos_writes(source):
        writes = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if any((isinstance(part, ast.Attribute) and part.attr == "qpos") or
                       (isinstance(part, ast.Name) and part.id == "qpos")
                       for part in ast.walk(target)):
                    writes.append(node.lineno)
        return writes

    def test_qpos_ast_guard_detects_assignment_forms_but_allows_reads(self):
        for source in ("env.d.qpos = value", "env.d.qpos[:] = value",
                       "env.d.qpos[0] += 1", "env.d.qpos: object = value",
                       "a, env.d.qpos[0] = values", "(qpos := value)"):
            with self.subTest(source=source):
                self.assertTrue(self.qpos_writes(source))
        self.assertEqual(self.qpos_writes("position = env.d.qpos.copy()\nnote = 'qpos = value'"), [])

    def test_all_three_api_sources_have_no_qpos_assignments(self):
        for module in (dual, handover, blocks):
            path = Path(module.__file__)
            with self.subTest(source=path.name):
                self.assertEqual(self.qpos_writes(path.read_text(encoding="utf-8")), [], str(path))

    def test_no_mlx_runtime_was_loaded(self):
        names = {"mlx", "mlx_vlm", "mlx_lm"}
        loaded = [name for name, module in sys.modules.items()
                  if module is not None and name.split(".")[0] in names]
        self.assertEqual(loaded, [])


class CoordinateProvenanceTests(unittest.TestCase):
    def validate(self,source):
        from run_remaining_cabto import waypoint_provenance
        names={"target_pose","safe_pose","grasp_pose","approach","descend_to","move_safe","move_above","lower","grasp","lift"}
        waypoint_provenance(PolicyProgram(source,names))
    def test_read_and_move(self):
        self.validate("def policy(api):\n    p=api.target_pose()\n    api.move_above(p)\n")
    def test_nested_read(self):
        self.validate("def policy(api):\n    api.move_safe(api.safe_pose())\n")
    def test_sliced_xy(self):
        self.validate("def policy(api):\n    p=api.target_pose()\n    api.descend_to(p[2],p[:2])\n")
    def test_literal_waypoint_rejected(self):
        with self.assertRaises(ValueError):self.validate("def policy(api):\n    api.approach((0.3,0.0,0.5))\n")
    def test_overwritten_waypoint_rejected(self):
        with self.assertRaises(ValueError):self.validate("def policy(api):\n    p=api.target_pose()\n    p=(0.3,0.0,0.5)\n    api.approach(p)\n")
    def test_mixed_literal_xy_rejected(self):
        with self.assertRaises(ValueError):self.validate("def policy(api):\n    p=api.target_pose()\n    api.descend_to(p[2],(p[0],0.1))\n")

if __name__ == "__main__":
    unittest.main()
