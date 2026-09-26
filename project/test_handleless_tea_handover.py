"""Regression tests for real handle removal, box-face grasping and scene isolation."""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path
import textwrap
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np
from core.handleless_tea_handover import build_box_scene, HandlelessTeaRuntime, BoxFaceAudit, box_domain
from core.smooth_handover_bridge import build_scene as old_scene

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/"scenes/handover_tea_box_v3.json").read_text())

class HandlelessSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml=build_box_scene(C);cls.root=ET.fromstring(cls.xml)
        cls.model=mujoco.MjModel.from_xml_string(cls.xml)
        cls.box=cls.root.find("worldbody/body[@name='baton']")

    def test_only_one_physical_box_geom(self):
        physical=[g for g in self.box.findall("geom") if g.get("contype")!="0"]
        self.assertEqual([g.get("name") for g in physical],["tea_box_core"])
        self.assertEqual(physical[0].get("type"),"box")

    def test_no_handle_or_stem_geometry_anywhere(self):
        self.assertFalse(any("handle" in g.get("name","") or "stem" in g.get("name","") for g in self.root.iter("geom")))

    def test_box_size_and_mass(self):
        geom=self.box.find("geom[@name='tea_box_core']")
        np.testing.assert_allclose(np.fromstring(geom.get("size"),sep=" "),C["handover"]["object_half_size"])
        bid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"baton")
        self.assertAlmostEqual(self.model.body_mass[bid],.15)

    def test_box_has_freejoint_not_fixed_world_attachment(self):
        self.assertIsNotNone(self.box.find("freejoint"))
        for w in self.root.findall("equality/weld"):
            if w.get("body2")=="baton":self.assertEqual(w.get("active"),"false")

    def test_no_box_gravity_compensation(self):
        bid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"baton")
        self.assertEqual(self.model.body_gravcomp[bid],0)

    def test_two_same_size_collision_enabled_fixed_display_boxes(self):
        centers=C["visual_style"]["box_display_centers_xy"]
        self.assertEqual(centers,[[-0.06,-0.22],[0.75,0.22]])
        self.assertEqual(len(centers),2)
        for i in range(len(centers)):
            body=self.root.find(f"worldbody/body[@name='tea_display_{i}']")
            self.assertIsNone(body.find("freejoint"))
            geom=body.find(f"geom[@name='tea_display_{i}_tea_box_core']")
            self.assertEqual(geom.get("contype"),"1")
            np.testing.assert_allclose(np.fromstring(geom.get("size"),sep=" "),C["handover"]["object_half_size"])
        self.assertIsNone(self.root.find("worldbody/body[@name='tea_display_2']"))

    def test_grasp_points_inside_visible_carton(self):
        half=np.array(C["handover"]["object_half_size"])
        for pt in C["handover"]["grasp_local_points"]:self.assertTrue(np.all(abs(np.array(pt))<half))

    def test_geometry_build_does_not_mutate_config(self):
        a=deepcopy(C);build_box_scene(a);self.assertEqual(a,C)

    def test_minimal_package_excludes_obsolete_handled_config(self):
        self.assertFalse((ROOT/"scenes/handover_green_tea_v2.json").exists())

class ControllerContractTests(unittest.TestCase):
    def test_bilateral_force_required_before_weld(self):
        src=inspect.getsource(HandlelessTeaRuntime._close_and_attach)
        self.assertLess(src.index("bilateral<2"),src.index('_set_weld("baton", True)'))
        self.assertIn('normal_force_n',src);self.assertIn('.03',src);self.assertIn('-.10',src)

    def test_attachment_requires_current_bilateral_contacts(self):
        src=inspect.getsource(HandlelessTeaRuntime._close_and_attach)
        self.assertIn('len(current_fingers)!=2',src)
        self.assertLess(src.index('len(current_fingers)!=2'),src.index('_set_weld("baton", True)'))

    def test_home_and_stability_have_their_own_collision_phases(self):
        self.assertIn('home_receiver_arm',inspect.getsource(HandlelessTeaRuntime._controlled_home))
        self.assertIn('stability',inspect.getsource(HandlelessTeaRuntime.stable_goal))

    def test_direct_box_face_contact_zone(self):
        src=inspect.getsource(BoxFaceAudit.sample)
        self.assertIn('other_geom == "tea_box_core"',src);self.assertIn('face and zone',src)
        self.assertIn('allowed_handle_penetration_m',src)

    def test_display_contacts_are_audited(self):
        self.assertIn('unexpected_display_contact',inspect.getsource(BoxFaceAudit.sample))

    def test_execution_has_no_qpos_or_qvel_assignment(self):
        for cls in (HandlelessTeaRuntime,):
            tree=ast.parse(textwrap.dedent(inspect.getsource(cls)))
            for node in ast.walk(tree):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,ast.AugAssign) else []
                for t in targets:self.assertNotIn('qpos',ast.unparse(t));self.assertNotIn('qvel',ast.unparse(t))

    def test_cartesian_controller_uses_slewed_joint_actuators(self):
        src=inspect.getsource(HandlelessTeaRuntime._move)
        self.assertIn('step_joint',src);self.assertIn('np.clip',src);self.assertIn('local_only=True',src)

    def test_table_support_is_physical_force_not_only_height(self):
        src=inspect.getsource(HandlelessTeaRuntime.placement_geometry)
        self.assertIn('mj_contactForce',src);self.assertIn('support',src)

    def test_no_old_handle_action_in_new_models(self):
        for m in box_domain():self.assertNotIn('handle',m['name'])

if __name__=="__main__":unittest.main()
