import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).with_name("p48_1_tripo_cleanup.py")
SPEC = importlib.util.spec_from_file_location("p481", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class StaticCleanupTests(unittest.TestCase):
    def test_identity_quaternion_is_identity_rotation(self):
        R = MOD.quat_to_rot_wxyz(np.array([[1.0, 0.0, 0.0, 0.0]]))
        self.assertTrue(np.allclose(R[0], np.eye(3)))

    def test_camera_intersecting_splat_receives_higher_static_risk(self):
        cam = SimpleNamespace(c2w=np.eye(4, dtype=np.float32), K=np.diag([400., 400., 1.]).astype(np.float32), width=896, height=896)
        means = np.array([[0., 0., 0.02], [0., 0., 4.]], np.float32)
        scales = np.array([[.1, .1, .1], [.01, .01, .01]], np.float32)
        q = np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]], np.float32)
        risk = MOD.risk_components(means, scales, q, np.array([.5, .5], np.float32), [cam])
        self.assertLess(risk["min_mahalanobis"][0], 3.0)
        self.assertGreater(risk["risk"][0], risk["risk"][1])

    def test_cleaned_ply_mask_is_global_boolean_not_per_frame(self):
        risk = {"risk": np.array([0.99, 0.10, 0.05], np.float32), "min_mahalanobis": np.array([1., 9., 9.]),
                "max_projected_radius_px": np.array([1200., 2., 2.]), "near_camera_count": np.array([3, 0, 0]),
                "scale_outlier": np.array([1., 0., 0.])}
        masks = MOD.candidate_masks(risk)
        self.assertTrue(all(mask.shape == (3,) and mask.dtype == bool for mask in masks.values()))
        self.assertTrue(all(not mask[0] for mask in masks.values()))

    def test_projected_trajectory_scan_removes_only_near_large_splats(self):
        """A real projection scan must drive a static, not frame-local, mask."""
        risk = {
            "risk": np.array([0.1, 0.1, 0.1], np.float32),
            "min_mahalanobis": np.array([9., 9., 9.]),
            "max_projected_radius_px": np.array([2., 2., 2.]),
            "near_camera_count": np.array([0, 0, 0]),
            "scale_outlier": np.array([0., 0., 0.]),
        }
        scan = {
            "near_radius_px": np.array([499, 500, 2400], np.int32),
            "min_depth_m": np.array([0.10, 0.20, 0.21], np.float32),
        }
        masks = MOD.candidate_masks(risk, projected_scan=scan)
        # The aggressive tier includes the exact 0.20m / 500px boundary;
        # a giant splat just outside the depth guard remains.
        self.assertTrue(masks["aggressive"][0])
        self.assertFalse(masks["aggressive"][1])
        self.assertTrue(masks["aggressive"][2])

    def test_projected_cleanup_is_restricted_to_the_target_hard_views(self):
        risk = {
            "risk": np.zeros(2, np.float32), "min_mahalanobis": np.full(2, 9.),
            "max_projected_radius_px": np.zeros(2), "near_camera_count": np.zeros(2),
            "scale_outlier": np.zeros(2),
        }
        scan = {
            "near_radius_px": np.array([800, 800], np.int32),
            "min_depth_m": np.array([0.10, 0.10], np.float32),
            "target_near_radius_px": np.array([0, 800], np.int32),
            "target_min_depth_m": np.array([np.inf, 0.10], np.float32),
        }
        masks = MOD.candidate_masks(risk, projected_scan=scan)
        self.assertTrue(masks["aggressive"][0])
        self.assertFalse(masks["aggressive"][1])

    def test_full72_coverage_gate_rejects_large_mean_loss(self):
        self.assertTrue(MOD.passes_full72_coverage(0.8337, 0.8200))
        self.assertFalse(MOD.passes_full72_coverage(0.8337, 0.8100))

    def test_coverage_metric_compares_raw_and_cleaned_with_one_guard(self):
        before = {"min": .44, "p10": .66, "median": .74, "mean": .79, "max": 1.0}
        after = {"min": .44, "p10": .66, "median": .74, "mean": .785, "max": 1.0}
        metric = MOD.coverage_metric(before, after)
        self.assertAlmostEqual(metric["absolute_drop_raw_minus_cleaned"]["mean"], .005)
        self.assertTrue(metric["coverage_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
