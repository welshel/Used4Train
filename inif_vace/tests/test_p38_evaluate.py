from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "p38_evaluate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p38_evaluate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvaluationTests(unittest.TestCase):
    def test_identical_video_metrics_are_perfect_and_temporally_zero(self):
        module = load_module()
        video = np.zeros((4, 32, 32, 3), dtype=np.float32)
        video[:, 8:24, 8:24] = 0.7

        result = module.compute_metrics(video, video)

        self.assertEqual(result["mse"], 0.0)
        self.assertGreater(result["ssim"], 0.9999)
        self.assertEqual(result["temporal_delta_error"], 0.0)
        self.assertEqual(result["edge_temporal_error"], 0.0)
        self.assertEqual(result["flow_warp_excess"], 0.0)

    def test_salient_object_mask_is_nonempty_and_excludes_flat_border(self):
        module = load_module()
        clean = np.zeros((5, 64, 64, 3), dtype=np.float32)
        condition = clean.copy()
        for index in range(5):
            clean[index, 20:44, 18 + index : 42 + index] = 1.0
            condition[index, 20:44, 18 + index : 42 + index] = 0.65

        mask, box = module.build_salient_object_mask(clean, condition)

        self.assertEqual(mask.shape, (64, 64))
        self.assertGreater(mask.mean(), 0.01)
        self.assertFalse(mask[0, 0])
        x0, y0, x1, y1 = box
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)
        self.assertLessEqual(x0, 22)
        self.assertLessEqual(22, x1)

    def test_clip_selection_uses_only_clean_and_condition_scores(self):
        module = load_module()
        scores = [
            {"start": 0, "edge_score": 0.2, "complex_score": 0.9},
            {"start": 17, "edge_score": 0.8, "complex_score": 0.3},
            {"start": 31, "edge_score": 0.5, "complex_score": 0.6},
        ]

        selection = module.choose_validation_roles(scores)

        self.assertEqual(selection, {"V_EDGE": 17, "V_COMPLEX": 0})

    def test_temporal_excess_is_zero_for_output_matching_clean(self):
        module = load_module()
        clean = np.zeros((5, 48, 48, 3), dtype=np.float32)
        for index in range(5):
            clean[index, 16:32, 10 + index : 26 + index] = 0.8
        condition = clean * 0.75
        mask, _ = module.build_salient_object_mask(clean, condition)

        value = module.object_temporal_excess(clean, clean, mask)

        self.assertLess(abs(value), 1e-8)

    def test_high_frequency_metrics_are_perfect_for_identical_output(self):
        module = load_module()
        clean = np.zeros((4, 64, 64, 3), dtype=np.float32)
        clean[:, 16:48, 16:48] = 1.0
        condition = clean * 0.7
        mask = module.build_high_frequency_mask(clean, condition)

        result = module.compute_high_frequency_metrics(clean, clean, mask)

        self.assertEqual(mask.shape, (4, 64, 64))
        self.assertGreater(mask.mean(), 0.01)
        self.assertEqual(result["hf_mse"], 0.0)
        self.assertGreater(result["hf_edge_f1"], 0.999)
        self.assertEqual(result["hf_gradient_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
