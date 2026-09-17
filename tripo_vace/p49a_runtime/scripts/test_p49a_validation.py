import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("p49a_validate.py")
SPEC = importlib.util.spec_from_file_location("p49a_validate", SCRIPT)
VAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VAL
assert SPEC.loader
SPEC.loader.exec_module(VAL)


class P49AValidationTests(unittest.TestCase):
    def test_cyclic_indices_preserve_exact_frame_order(self):
        self.assertEqual(VAL.clip_indices(31, 33)[0], 31)
        self.assertEqual(VAL.clip_indices(31, 33)[-1], 63)
        self.assertEqual(VAL.clip_indices(50, 33)[-1], 10)

    def test_pixel_metrics_are_identity_at_zero_error(self):
        frame = np.full((32, 32, 3), 127, dtype=np.uint8)
        metrics = VAL.compute_frame_metrics([frame], [frame.copy()])
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["psnr"], float("inf"))
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
