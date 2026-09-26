"""Lightweight tests for the NEW controller; no dependency on old result files."""
import ast
import inspect
import json
from pathlib import Path
import textwrap
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import mujoco
import numpy as np
from core.single_arm_rebuilt import Exp4Env, RebuiltSingleArm, ease, scene
ROOT=Path(__file__).resolve().parent


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs={
            "cover":json.loads((ROOT/"scenes/cover_kitchen_sort_v2.json").read_text()),
            "blocks":json.loads((ROOT/"scenes/blocks_rebuilt_v1.json").read_text()),
        }
        cls.scenes={t:scene(c) for t,c in cls.configs.items()}
        cls.roots={t:ET.fromstring(x[0]) for t,x in cls.scenes.items()}
    def test_both_scenes_compile(self):
        for xml,names in self.scenes.values():
            m=mujoco.MjModel.from_xml_string(xml)
            self.assertGreater(m.nq,9);self.assertEqual(len(names),m.njnt-9)
    def test_one_arm_only(self):
        for root in self.roots.values():
            self.assertEqual(len([n for n in root.iter("body") if n.get("name","").startswith("link0")]),1)
    def test_four_free_coloured_cubes(self):
        self.assertEqual(set(self.scenes["blocks"][1]),{"red","blue","yellow","green"})
    def test_cubes_start_without_stack(self):
        root=self.roots["blocks"]
        positions=[np.fromstring(root.find(f".//body[@name='{n}']").get("pos"),sep=" ") for n in self.scenes["blocks"][1]]
        for a in positions:self.assertAlmostEqual(a[2],.423)
        for i,a in enumerate(positions):
            for b in positions[i+1:]:self.assertGreater(np.linalg.norm(a[:2]-b[:2]),.15)
    def test_three_foods_are_free_and_not_pinned(self):
        root=self.roots["cover"]
        for name in ("shrimp","apple","potato"):
            self.assertIsNotNone(root.find(f".//body[@name='{name}']/freejoint"))
            self.assertFalse(any(e.get("body2")==name and e.get("active")=="true" for e in root.findall("./equality/weld")))
    def test_supports_are_fixed_in_a_back_row(self):
        root=self.roots["cover"]
        xs=[]
        for name in ("bowl","board","pan"):
            body=root.find(f".//body[@name='{name}']");self.assertIsNotNone(body)
            self.assertIsNone(body.find("freejoint"));pos=np.fromstring(body.get("pos"),sep=" ")
            xs.append(pos[0]);self.assertAlmostEqual(pos[1],.16)
        self.assertEqual(xs,[.31,.49,.67])
    def test_foods_form_a_front_row(self):
        root=self.roots["cover"];xs=[]
        for name in ("shrimp","apple","potato"):
            pos=np.fromstring(root.find(f".//body[@name='{name}']").get("pos"),sep=" ")
            xs.append(pos[0]);self.assertAlmostEqual(pos[1],-.19)
        self.assertEqual(xs,[.31,.49,.67])
    def test_bowl_is_upright_and_open(self):
        root=self.roots["cover"];body=root.find(".//body[@name='bowl']")
        self.assertIsNone(body.get("quat"));self.assertIsNotNone(body.find("geom[@name='bowl_floor']"))
        self.assertEqual(len([g for g in body.findall("geom") if g.get("name","").startswith("bowl_wall_")]),32)
        self.assertFalse(any(g.get("type")=="sphere" for g in body.findall("geom")))
    def test_explicit_food_to_support_mapping(self):
        self.assertEqual(self.configs["cover"]["pairs"],[["shrimp","bowl"],["apple","board"],["potato","pan"]])
    def test_cover_table_is_longer_than_blocks_table(self):
        self.assertGreater(self.configs["cover"]["table_half_size"][0],self.configs["blocks"]["table_half_size"][0])
    def test_seed_reproducibility(self):
        for c in self.configs.values():self.assertEqual(scene(c,3),scene(c,3))
    def test_seed_changes_initial_layout(self):
        for c in self.configs.values():self.assertNotEqual(scene(c,0)[0],scene(c,3)[0])
    def test_input_configuration_unchanged(self):
        for c in self.configs.values():
            before=json.dumps(c);scene(c,8);self.assertEqual(before,json.dumps(c))
    def test_strict_default(self):
        for c in self.configs.values():self.assertEqual(c["control"]["grasp_mode"],"strict")
    def test_gravity_compensation_is_robot_only(self):
        for root in self.roots.values():
            self.assertTrue(all(b.get("gravcomp")=="1" for b in root.find(".//body[@name='link0']").iter("body")))
            for body in root.findall(".//body"):
                if body.find("freejoint") is not None:self.assertIsNone(body.get("gravcomp"))
    def test_camera_axes_are_orthogonal(self):
        for root in self.roots.values():
            for camera in root.findall(".//camera"):
                a=np.fromstring(camera.get("xyaxes"),sep=" ").reshape(2,3)
                self.assertAlmostEqual(float(a[0]@a[1]),0,places=6)


class MotionAndSafetyTests(unittest.TestCase):
    def test_ease_endpoints(self):self.assertEqual(ease(0),0);self.assertEqual(ease(1),1)
    def test_ease_monotone(self):self.assertTrue(np.all(np.diff([ease(x) for x in np.linspace(0,1,101)])>=0))
    def test_ease_endpoint_slope(self):
        self.assertLess(ease(1e-5)/1e-5,1e-7);self.assertLess((1-ease(1-1e-5))/1e-5,1e-7)
    def test_execution_no_actual_qpos_or_qvel_assignment(self):
        tree=ast.parse(inspect.getsource(RebuiltSingleArm))
        for node in ast.walk(tree):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,(ast.AugAssign,ast.AnnAssign)) else []
            for t in targets:
                self.assertNotIn(".qpos",ast.unparse(t));self.assertNotIn(".qvel",ast.unparse(t))
    def test_step_joint_uses_actuators_not_state_writes(self):
        src=inspect.getsource(Exp4Env.step_joint)
        self.assertIn("self.d.ctrl",src);self.assertIn("mujoco.mj_step",src)
        self.assertNotIn("self.d.qpos[",src);self.assertNotIn("self.d.qvel[",src)
    def test_cover_support_contact_is_surface_height_gated(self):
        src=inspect.getsource(RebuiltSingleArm.assess_one)
        self.assertIn('support_surface_z',src);self.assertIn('support_surface_geom',src)
        self.assertIn('supporting_contact(n,support,surface_z,surface_geom)',src)
    def test_success_requires_hand_clearance_and_motion_gate(self):
        src=inspect.getsource(RebuiltSingleArm.assess)
        self.assertIn('released_clear',src);self.assertIn('motion_ok',src)
    def test_no_legacy_task_controller(self):
        source=inspect.getsource(RebuiltSingleArm)
        for name in ("ArmSkills","PrimitiveRunner","CoverRuntime","rehome("):self.assertNotIn(name,source)
    def test_release_checks_raw_constraints_and_actual_width(self):
        r=RebuiltSingleArm.__new__(RebuiltSingleArm);r.arm=Mock();r.arm._grip_cmd=1.;r.arm.get_gripper_width.return_value=.08;r.active_welds=Mock(return_value=[])
        self.assertTrue(r.released());r.active_welds.return_value=["cover"];self.assertFalse(r.released())
        r.active_welds.return_value=[];r.arm.get_gripper_width.return_value=.01;self.assertFalse(r.released())
    def test_pick_requires_simultaneous_finger_force(self):
        self.assertIn("min(self.force_now.values())<.05",inspect.getsource(RebuiltSingleArm.pick))
    def test_pose_observation_does_not_make_home_true(self):
        c=json.loads((ROOT/"scenes/blocks_rebuilt_v1.json").read_text());r=RebuiltSingleArm(c,render=False)
        try:
            self.assertFalse(r.assess()["final_home"]);self.assertFalse(r.assess()["success"])
            self.assertEqual(r.active_welds(),[])
        finally:r.close()
    def test_support_and_goal_monitor_present(self):
        self.assertIn("supporting_contact",inspect.getsource(RebuiltSingleArm.assess_one))
        self.assertIn("self.completed",inspect.getsource(RebuiltSingleArm.sample))
    def test_both_tasks_have_bounded_speed(self):
        paths=(ROOT/"test_fixtures/cover_rebuilt_v1.json",ROOT/"scenes/blocks_rebuilt_v1.json")
        for path in paths:
            c=json.loads(path.read_text())["control"]
            self.assertLessEqual(c["joint_speed_rad_s"],.85);self.assertLessEqual(c["cartesian_speed_m_s"],.20)


if __name__=="__main__":unittest.main()
