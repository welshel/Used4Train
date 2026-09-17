import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("p48_3_dynamic_gaussian_cleanup.py")
SPEC = importlib.util.spec_from_file_location("p483", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class EllipsoidAwareDynamicCleanupTests(unittest.TestCase):
    def test_hard_view_board_covers_all_manually_discovered_veil_frames(self):
        # The final board must retain the original P48.3 targets and include
        # every frame whose residual veil required an attribution iteration.
        self.assertTrue({10, 11, 12, 13, 14, 16, 17, 18, 21, 24, 28, 29}.issubset(set(MOD.HARD_FRAMES)))

    def test_manual_qa_validation_requires_each_frame_and_allows_only_mild_or_better_occlusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.csv"
            with path.open("w", newline="") as f:
                writer = __import__("csv").DictWriter(f, fieldnames=MOD.QA_COLUMNS)
                writer.writeheader()
                for frame in range(72):
                    writer.writerow({
                        "frame": f"F{frame:02d}",
                        "white_foreground_occlusion": "MILD" if frame == 11 else "NONE",
                        "floater_present": "NO",
                        "new_hole": "NONE",
                        "important_content_removed": "NO",
                        "temporal_transition": "PASS",
                        "manual_status": "PASS",
                        "notes": "inspected",
                    })
            audit = MOD.validate_manual_qa_csv(path)
        self.assertTrue(audit["eligible_for_finalization"])
        self.assertEqual(audit["frames_checked"], 72)
        self.assertEqual(audit["white_foreground_occlusion"]["MILD"], 1)

    def test_manual_qa_validation_rejects_moderate_foreground_occlusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.csv"
            with path.open("w", newline="") as f:
                writer = __import__("csv").DictWriter(f, fieldnames=MOD.QA_COLUMNS)
                writer.writeheader()
                for frame in range(72):
                    writer.writerow({
                        "frame": f"F{frame:02d}",
                        "white_foreground_occlusion": "MODERATE" if frame == 12 else "NONE",
                        "floater_present": "NO",
                        "new_hole": "NONE",
                        "important_content_removed": "NO",
                        "temporal_transition": "PASS",
                        "manual_status": "PASS",
                        "notes": "inspected",
                    })
            audit = MOD.validate_manual_qa_csv(path)
        self.assertFalse(audit["eligible_for_finalization"])
        self.assertIn("F12", audit["blocking_frames"])
    def test_review_sheets_keep_each_dynamic_clean_pair_large_enough_for_manual_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            dynamic_dir, clean_dir = Path(tmp) / "dynamic", Path(tmp) / "clean"
            dynamic_dir.mkdir()
            clean_dir.mkdir()
            pixels = np.zeros((8, 8, 3), np.uint8)
            for frame in range(72):
                MOD.Image.fromarray(pixels).save(dynamic_dir / f"F{frame:02d}.png")
                MOD.Image.fromarray(pixels).save(clean_dir / f"F{frame:02d}.png")
            paths = MOD.make_review_sheets(Path(tmp), dynamic_dir, clean_dir)
            self.assertEqual(len(paths), 4)
            with MOD.Image.open(paths[0]) as sheet:
                self.assertEqual(sheet.size, (896, 34 + 18 * 448))

    def test_finalization_status_fails_closed_for_temporal_regression(self):
        qa = {"eligible_for_finalization": True, "white_foreground_occlusion": {"MILD": 0}}
        coverage = {"gate": {"mean_drop": 0.008, "hard": 0.02}}
        temporal = {"gate": {"pass": False}}
        self.assertEqual(MOD.finalization_status(qa, coverage, temporal), "P48_3_FAIL_TEMPORAL_POPPING")

    def test_attribution_verified_boost_is_continuous_and_never_touches_non_candidates(self):
        weights = np.array([[0.5, 0.7], [0.2, 0.3]], np.float32)
        candidates = np.array([True, False])
        boosted = MOD.apply_attribution_boost(weights, candidates, [0, 1], gamma=4.0)
        self.assertAlmostEqual(float(boosted[0, 0]), 0.9375, places=5)
        self.assertAlmostEqual(float(boosted[1, 0]), 1.0 - 0.8 ** 4, places=5)
        self.assertEqual(float(boosted[0, 1]), 0.0)

    def test_frame_envelope_boost_remains_continuous_and_candidate_scoped(self):
        weights = np.array([[0.5, 0.7], [0.2, 0.3]], np.float32)
        candidates = np.array([True, False])
        strength = np.array([0.0, 1.0], np.float32)
        boosted = MOD.apply_frame_envelope_boost(weights, candidates, strength, gamma=4.0)
        self.assertAlmostEqual(float(boosted[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(boosted[1, 0]), 1.0 - 0.8 ** 4, places=5)
        self.assertEqual(float(boosted[1, 1]), 0.0)

    def test_post_suppression_attribution_floor_is_ema_smoothed_and_candidate_scoped(self):
        weights = np.zeros((4, 2), np.float32)
        candidates = np.array([True, False])
        boosted = MOD.apply_post_suppression_attribution_floor(weights, candidates, {1: [0, 1]}, floor=0.92)
        self.assertAlmostEqual(float(boosted[1, 0]), 0.92, places=5)
        self.assertAlmostEqual(float(boosted[2, 0]), 0.92 * 0.35, places=5)
        self.assertEqual(float(boosted[1, 1]), 0.0)

    def test_ellipsoid_front_detects_long_axis_reaching_camera_when_center_is_not_near(self):
        # Identity camera and local-to-world quaternion.  Center z=0.50 is
        # not close by itself, but its 3-sigma z extent reaches z=0.05.
        means_cam = np.array([[0.0, 0.0, 0.50]], np.float32)
        scales = np.array([[0.01, 0.01, 0.15]], np.float32)
        quats = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
        front = MOD.ellipsoid_front_depth(means_cam, scales, quats, np.eye(3), k=3.0)
        self.assertAlmostEqual(float(front[0]), 0.05, places=5)

    def test_dynamic_risk_needs_candidate_front_extent_and_footprint(self):
        front = np.array([0.05, 0.05, 0.05], np.float32)
        area = np.array([0.20, 0.001, 0.20], np.float32)
        opacity = np.array([0.2, 0.2, 0.2], np.float32)
        candidate = np.array([True, True, False])
        risk = MOD.ellipsoid_dynamic_risk(front, area, opacity, candidate, footprint_lo=0.01, footprint_hi=0.10)
        self.assertGreater(risk[0], 0.5)
        self.assertEqual(risk[1], 0.0)
        self.assertEqual(risk[2], 0.0)

    def test_continuous_ema_preserves_fractional_weight_and_smooth_exit(self):
        raw = np.array([[0.0], [0.8], [0.4], [0.0], [0.0]], np.float32)
        smoothed = MOD.continuous_ema(raw, alpha=0.65)
        self.assertGreater(smoothed[1, 0], 0.5)
        self.assertGreater(smoothed[2, 0], 0.4)
        self.assertGreater(smoothed[3, 0], 0.1)
        self.assertLess(smoothed[3, 0], smoothed[2, 0])

    def test_candidate_restriction_prevents_non_candidates_from_suppression(self):
        weights = np.array([0.8, 0.7], np.float32)
        candidates = np.array([True, False])
        restricted = MOD.restrict_to_candidates(weights, candidates)
        self.assertAlmostEqual(float(restricted[0]), 0.8)
        self.assertEqual(float(restricted[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
