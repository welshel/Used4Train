import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("p48_2_dynamic_suppression.py")
SPEC = importlib.util.spec_from_file_location("p482", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class DynamicSuppressionTests(unittest.TestCase):
    def test_risk_requires_near_large_curtain_candidate(self):
        ids = np.array([0, 1, 2])
        radii = np.array([1500, 1500, 1500])
        depths = np.array([0.08, 0.50, 0.08])
        opacity = np.array([0.2, 0.2, 0.2])
        curtain = np.array([True, True, False])
        score = MOD.frame_risk(3, ids, radii, depths, opacity, curtain)
        self.assertGreater(score[0], 0.5)
        self.assertEqual(score[1], 0.0)
        self.assertEqual(score[2], 0.0)

    def test_hysteresis_keeps_a_single_frame_floater_from_hard_popping(self):
        raw = np.zeros((8, 1), np.float32)
        raw[1, 0] = 1.0
        smooth = MOD.temporal_hysteresis(raw, enter=0.2, exit=0.05, decay=0.55)
        self.assertEqual(smooth[1, 0], 1.0)
        self.assertGreater(smooth[2, 0], 0.2)
        self.assertGreater(smooth[3, 0], 0.05)
        self.assertGreater(smooth[5, 0], 0.05)
        self.assertGreater(smooth[6, 0], 0.05)
        self.assertEqual(smooth[7, 0], 0.0)

    def test_full_and_appearance_modes_have_distinct_occlusion_behavior(self):
        opacity = np.array([0.8, 0.3], np.float32)
        rgb = np.array([[1.0, 1.0, 1.0], [0.4, 0.3, 0.2]], np.float32)
        weight = np.array([0.75, 0.0], np.float32)
        full_opacity, full_rgb = MOD.apply_suppression(opacity, rgb, weight, "full")
        appearance_opacity, appearance_rgb = MOD.apply_suppression(opacity, rgb, weight, "appearance_only")
        self.assertLess(full_opacity[0], opacity[0])
        self.assertEqual(appearance_opacity[0], opacity[0])
        self.assertLess(appearance_rgb[0, 0], rgb[0, 0])
        self.assertTrue(np.allclose(full_rgb, rgb))

    def test_smear_metric_counts_distributed_translucent_white_reduction(self):
        base = np.full((4, 4, 3), 0.60, np.float32)
        dynamic = np.full((4, 4, 3), 0.54, np.float32)
        metric = MOD.smear_metric(base, dynamic)
        self.assertGreater(metric["white_smear_reduced_fraction"], 0.99)


if __name__ == "__main__":
    unittest.main()
