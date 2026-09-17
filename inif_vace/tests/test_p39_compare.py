import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p39_compare import (
    load_historical_metrics,
    parse_starts,
    resolution_comparison_title,
    validate_video_contract,
)


class P39CompareTests(unittest.TestCase):
    def test_accepts_672_fixed_validation_contract(self) -> None:
        video = np.zeros((33, 672, 672, 3), dtype=np.float32)

        validate_video_contract(video, fps=12.0, width=672, height=672)

    def test_rejects_wrong_resolution(self) -> None:
        video = np.zeros((33, 640, 640, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "video contract failed"):
            validate_video_contract(video, fps=12.0, width=672, height=672)

    def test_reads_p38_historical_metrics_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"historical_l33_800_512": {"mse": 0.1}}))

            result = load_historical_metrics(path)

        self.assertEqual(result, {"mse": 0.1})

    def test_parses_single_start_for_quick_checkpoint_screen(self) -> None:
        self.assertEqual(parse_starts("31"), (31,))

    def test_rejects_unknown_or_duplicate_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation starts"):
            parse_starts("31,31")
        with self.assertRaisesRegex(ValueError, "validation starts"):
            parse_starts("8")

    def test_resolution_board_title_uses_actual_inference_size(self) -> None:
        self.assertEqual(
            resolution_comparison_title("P3.8", "H704", 704),
            "P3.8 vs H704 | matched inference704",
        )


if __name__ == "__main__":
    unittest.main()
