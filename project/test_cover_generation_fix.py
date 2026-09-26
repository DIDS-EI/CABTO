"""No model loading: fail-closed checks for guessed waypoints and source flow."""
import unittest
from core.cover_cabto_grounding import PolicyProgram
from core.cover_cabto_api import API_DOCS
from run_cover_cabto_method import validate_motion_provenance

class ProvenanceTests(unittest.TestCase):
    def check(self,source,kind="pick"):
        validate_motion_provenance(PolicyProgram(source,API_DOCS[kind]))
    def test_direct_perception_call(self):
        self.check("def policy(api):\n    api.approach(api.grasp_point())\n")
    def test_sensor_local(self):
        self.check("def policy(api):\n    p=api.grasp_point()\n    api.descend(p)\n")
    def test_destination_local(self):
        self.check("def policy(api):\n    p=api.destination_point()\n    api.carry(p)\n    api.lower(p)\n", "place")
    def test_reject_literal(self):
        with self.assertRaises(ValueError):self.check("def policy(api):\n    api.approach((0.3,0.1,0.4))\n")
    def test_reject_literal_local(self):
        with self.assertRaises(ValueError):self.check("def policy(api):\n    p=(0.3,0.1,0.4)\n    api.approach(p)\n")
    def test_reject_discarded_sensor_read(self):
        with self.assertRaises(ValueError):self.check("def policy(api):\n    api.grasp_point()\n    api.descend((0.3,0.1,0.4))\n")
    def test_reject_overwritten_sensor_local(self):
        with self.assertRaises(ValueError):self.check("def policy(api):\n    p=api.grasp_point()\n    p=(0.3,0.1,0.4)\n    api.approach(p)\n")
    def test_home_needs_no_waypoint(self):
        self.check("def policy(api):\n    api.return_home()\n","home")

if __name__=="__main__":unittest.main()
